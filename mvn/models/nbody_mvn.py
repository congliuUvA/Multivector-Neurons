import torch
import torch.nn.functional as F

from torch_geometric.nn import global_add_pool
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
            nn.Linear(2*hidden_features_s + hidden_features_v, hidden_features_s),
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

    def forward(self, x, edge_index):
        s, v = x
        i, j = edge_index

        s_mes, v_mes = self.message(s[i], s[j], v[i], v[j])
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
    
    
    def message(self, s_i, s_j, v_i, v_j):
        v_ij = self.v_linear(torch.cat([v_i, v_j], dim=1))
        v_ij_inv = (v_ij * v_ij).sum(dim=-1)
        s_ij = torch.cat([s_i, s_j, v_ij_inv], dim=-1)
        v_mes = self.edge_model_v(v_ij)
        s_mes = self.edge_model_s(s_ij)
        pos_message = self.pos_net(s_mes)
        # pos_message = pos_message * nn.Sigmoid()(pos_message)
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
        inv = qk / (k_sq + 1e-8) 
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

class NBody_MVN(nn.Module):
    def __init__(
        self,
        in_features_s=1,
        in_features_v=3,
        hidden_features_s=64,
        hidden_features_v=16,
        out_features=1,
        num_layers=4,
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        self.feature_embedding = MVLinear(self.algebra, in_features_v, hidden_features_v, subspaces=False)
        self.feature_embedding_s = nn.Linear(in_features_s, hidden_features_s)
        layers = []
        for i in range(num_layers):
            layers.append(
                MVNMPNN(hidden_features_s, hidden_features_v)
            )

        self.projection = MVLinear(self.algebra, hidden_features_v, out_features, subspaces=False)

        self.model = nn.Sequential(*layers)

    def _featurization(self, scalars, vectors):
        # invariant embeddings
        scalars = self.feature_embedding_s(scalars)

        # covariant embeddings
        vectors = self.feature_embedding(vectors)
        return scalars, vectors
    
    def _forward(self, scalars, vectors, edge_index):
        s, v = self._featurization(scalars, vectors)
        x = (s, v)
        for layer in self.model:
            x = layer(x, edge_index)
        s, v = x
        v = self.projection(v)[..., 0, 1:4]
        pred = v
        return s, pred
    
     
    def forward(self, batch, batch_idx, mode):
        batch = batch.to("cuda")
        batch_size = batch.ptr.shape[0] - 1
        coords = batch.loc.reshape(batch_size, -1, 3)

        mean_pos = coords.mean(dim=1, keepdims=True)
        input = coords - mean_pos
        num_nodes = input.shape[1]
        edge_index = batch.edge_index
        charges = batch.charges

        # covariant features
        input = input.reshape(batch_size*num_nodes, -1).unsqueeze(1)
        input = self.algebra.embed_grade(input, 1)
        vel = self.algebra.embed_grade(batch.vel.unsqueeze(1), 1)
        cl_charges = self.algebra.embed_grade(charges.unsqueeze(-1), 0)
        input = torch.cat([input, vel, cl_charges], dim=1)
        _, pred_v = self._forward(charges, input, edge_index)
        pred_v = pred_v.reshape(batch_size, -1, 3) + mean_pos
        loss = F.mse_loss(pred_v.reshape(-1, 3), batch.y, reduction="none").mean(dim=1)
        if torch.isnan(loss).any():
            breakpoint()

        return (
            loss.mean(),
            {"loss":  loss,},
        )