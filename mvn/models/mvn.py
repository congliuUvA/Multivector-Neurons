import torch
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, global_add_pool, knn_graph
import torch.nn as nn
from algebra.cliffordalgebra import CliffordAlgebra
from cegnn_utils import MVLinear, MVLayerNorm

import torch
from torch import nn


class LinearFullyConnectedGPLayer(nn.Module):
    def __init__(self, in_vec_dims, hidden_vec_dims, out_vec_dims):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        self.linear_left = MVLinear(self.algebra, in_vec_dims, hidden_vec_dims, subspaces=False, bias=True)
        self.linear_right = MVLinear(self.algebra, in_vec_dims, hidden_vec_dims, subspaces=False, bias=True)
        self.linear_out =  MVLinear(self.algebra, hidden_vec_dims + in_vec_dims, out_vec_dims, subspaces=False, bias=True)
        self.vec_norm = MVLayerNorm(self.algebra, out_vec_dims)

    def forward(self, vec1, vec2=None):
        # normalization
        vec_right = self.linear_right(vec1)
        vec = vec1 if vec2 is None else vec2
        vec_left = self.linear_left(vec)

        # geometric product
        vec_out = self.algebra.geometric_product(vec_left, vec_right)
        vec_out = torch.cat([vec_out, vec1], dim=1)
        vec_out = self.linear_out(vec_out)
        vec_out = self.vec_norm(vec_out) 
        return vec_out
    
class CVPLayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims):
        super(CVPLayerNorm, self).__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        self.vec_norm = MVLayerNorm(self.algebra, self.v)
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        return self.scalar_norm(s), self.vec_norm(v)
    
class MVNMPNN(nn.Module):
    def __init__(self, hidden_features_s, hidden_features_v, use_subspaces=True):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        self.hidden_features_s = hidden_features_s
        self.hidden_features_v = hidden_features_v
        self.v_linear = MVLinear(self.algebra, 2*hidden_features_v, hidden_features_v)
        self.edge_model_v = MVMLP(hidden_features_v, hidden_features_v, hidden_features_v, 2)
        self.edge_model_s = nn.Sequential(
            nn.Linear(2*hidden_features_s + hidden_features_v + 1, hidden_features_s),
            nn.ReLU(),
            nn.Linear(hidden_features_s, hidden_features_s)
        )
        self.update_model_v = LinearFullyConnectedGPLayer(hidden_features_v, hidden_features_v, hidden_features_v)
        self.update_model_s = nn.Sequential(
            nn.Linear(hidden_features_s, hidden_features_s),
            nn.ReLU(),
            nn.Linear(hidden_features_s, hidden_features_s)
        )   
        factor = 1 if not use_subspaces else self.algebra.n_subspaces
        self.subspaces = use_subspaces
        self.pos_net = nn.Sequential(
            nn.Linear(hidden_features_s, hidden_features_s),
            nn.ReLU(),
            nn.Linear(hidden_features_s, hidden_features_v*factor)
        )
        self.mes_norm = CVPLayerNorm((hidden_features_s, hidden_features_v))
        self.update_norm_s = nn.LayerNorm(hidden_features_s)

    def forward(self, x, pos, edge_index):
        s, v = x
        i, j = edge_index

        s_mes, v_mes = self.message(s[i], s[j], v[i], v[j], pos[i], pos[j])
        s_mes, v_mes = self.mes_norm((s_mes, v_mes))
        num_messages = torch.bincount(j).unsqueeze(-1)
        s_mes = global_add_pool(s_mes, j) / torch.sqrt(num_messages)
        v_mes = (global_add_pool(v_mes.reshape(v_mes.shape[0], -1), j) / torch.sqrt(num_messages)).reshape(v.shape[0], -1, 2**self.algebra.dim)

        v_out = self.update_model_v(v_mes) 
        s_out = self.update_model_s(s_mes)
        s_out = self.update_norm_s(s_out)
        s_out = s_out + s
        v_out = v_out + v
        
        return (s_out, v_out)

    
    def message(self, s_i, s_j, v_i, v_j, pos_i, pos_j):
        v_ij = self.v_linear(torch.cat([v_i, v_j], dim=1))
        v_ij_inv = (v_ij * v_ij).sum(dim=-1)
        s_ij = torch.cat([s_i, s_j, (pos_i - pos_j).unsqueeze(-1), v_ij_inv], dim=-1)
        v_mes = self.edge_model_v(v_ij)
        s_mes = self.edge_model_s(s_ij)
        pos_message = self.pos_net(s_mes)
        v_mes = v_mes * (pos_message.reshape(pos_message.shape[0], -1, self.algebra.n_subspaces).repeat_interleave(self.algebra.subspaces, dim=-1)) if self.subspaces else v_ij * pos_message.unsqueeze(-1)  
        return s_mes, v_mes
        

class MVAct(nn.Module):
    def __init__(self, features_v):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        self.lin1 = MVLinear(self.algebra, features_v, features_v, subspaces=False, bias=False)
        self.lin2 = MVLinear(self.algebra, features_v, features_v, subspaces=False, bias=False)

    def forward(self, x):
        input = x
        q = self.lin1(input)
        k = self.lin2(input)

        qk = (q * k).sum(dim=-1)
        k_sq = (k ** 2).sum(dim=-1)
        inv = qk / k_sq
        c1 = q
        c2 = q - inv[..., None] * k
        gate = torch.maximum(torch.sign(qk)[..., None], torch.tensor(0.))

        return gate * c1 + (1 - gate) * c2

class MVMLP(nn.Module):
    def __init__(
        self, in_features_v, hidden_features_v, out_features_v, num_layers
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(MVLinear(self.algebra, in_features_v, hidden_features_v, subspaces=False))
            else:
                layers.append(MVLinear(self.algebra, hidden_features_v, hidden_features_v, subspaces=False))
            layers.append(MVAct(hidden_features_v))
        layers.append(MVLinear(self.algebra, hidden_features_v, out_features_v))

        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        return self.mlp(input)

class MVN(nn.Module):
    def __init__(
        self,
        in_features=1,
        hidden_features_s=64,
        hidden_features_v=16,
        out_features=1,
        num_layers=4,
        num_nearest_neighbors=16,
        num_tokens=20,
    ):
        super().__init__()
        self.num_nearest_neighbors = num_nearest_neighbors
        self.algebra = CliffordAlgebra((1,1,1))
        self.feature_embedding = MVLinear(self.algebra, in_features, hidden_features_v, subspaces=False)
        self.aa_embedding = nn.Embedding(num_tokens, hidden_features_s)
        layers = []
        for i in range(num_layers):
            layers.append(
                MVNMPNN(hidden_features_s, hidden_features_v)
            )

        self.projection = MVLinear(self.algebra, hidden_features_v, out_features, subspaces=False)

        self.model = nn.Sequential(*layers)

    def _featurization(self, seqs, coords, batch_idx):
        # invariant embeddings
        seqs = self.aa_embedding(seqs)

        # covariant embeddings
        coords_mean = global_mean_pool(coords, batch_idx)[batch_idx]
        coords_input = coords - coords_mean
        coords_input = self.algebra.embed_grade(coords_input.unsqueeze(1), 1)
        coords_output = self.feature_embedding(coords_input)
        return seqs, coords_output, coords_mean
    
    def _forward(self, seqs, coords, positions, edge_index, batch_idx):
        s, v, coords_mean = self._featurization(seqs, coords, batch_idx)
        x = (s, v)
        for layer in self.model:
            x = layer(x, positions, edge_index)
        s, v = x
        v = self.projection(v)[..., 0, 1:4]
        pred = v + coords_mean
        return s, pred
    
     
    def forward(self, batch, batch_idx, mode):

        coords, positions, seqs, masks = batch

        protein_lengths = masks.sum(dim=1)
        batch_idx = torch.arange(len(coords), device=coords.device).repeat_interleave(3 * protein_lengths)

        coords = coords[masks]
        positions = positions[masks].repeat_interleave(3)
        seqs = seqs[masks].repeat_interleave(3)

        coords = coords.reshape(-1, 3)

        noise = torch.randn_like(coords)
        noised_coords = coords + noise
        edge_index = knn_graph(noised_coords, k=self.num_nearest_neighbors, batch=batch_idx, loop=True)
        feats, denoised_coords = self._forward(seqs, noised_coords, positions, edge_index, batch_idx)
        loss = F.mse_loss(denoised_coords, coords, reduction="none").mean(dim=1)

        return (
            loss.mean(),
            {"loss":  loss,},
        )