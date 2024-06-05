import torch
import torch.nn.functional as F

from torch_geometric.nn import global_mean_pool, global_add_pool, knn_graph
import torch.nn as nn
from algebra.cliffordalgebra import CliffordAlgebra
# from einops import rearrange


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    """
    L2 norm of tensor clamped above a minimum value `eps`.
    :param sqrt: if `False`, returns the square of the L2 norm
    """
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out

    
class LinearFullyConnectedDotProductLayer(nn.Module):
    def __init__(self, in_vec_dims, hidden_vec_dims, out_scalar_dims, residual=False):
        super().__init__()
        self.linear_left = nn.Linear(in_vec_dims, hidden_vec_dims, bias=False)
        self.linear_right = nn.Linear(in_vec_dims, hidden_vec_dims, bias=False)
        self.linear_out = nn.Linear(hidden_vec_dims, out_scalar_dims)
        self.residual = residual 

    def forward(self, vec):
        # normalization
        vec_right = self.linear_right(vec)
        vec_left = self.linear_left(vec)

        # dot product
        dot = (vec_left * vec_right).sum(dim=1)

        if self.residual:
            vec_norm = _norm_no_nan(vec, axis=-2, keepdims=True)
            dot += vec_norm
        dot = self.linear_out(dot)
        return dot

class LinearFullyConnectedCrossProductLayer(nn.Module):
    def __init__(self, in_vec_dims, hidden_vec_dims, out_scalar_dims):
        super().__init__()
        self.linear_left = nn.Linear(in_vec_dims, hidden_vec_dims, bias=False)
        self.linear_right = nn.Linear(in_vec_dims, hidden_vec_dims, bias=False)
        self.linear_out = nn.Linear(hidden_vec_dims, out_scalar_dims, bias=False)

    def forward(self, vec):
        # normalization
        vec_right = self.linear_right(vec)
        vec_left = self.linear_left(vec)

        # cross product
        cross_vec = torch.cross(vec_left, vec_right, dim=1)

        # dot cross product
        cross_dot = (vec_left * vec_right).sum(dim=1)
        
        vec_out = self.linear_out(cross_vec)
        # return vec_out, cross_dot
        return cross_dot

class GVPLinear(nn.Module):
    """
    Geometric Vector Perceptron. See manuscript and README.md
    for more details.

    :param in_dims: tuple (n_scalar, n_vector)
    :param out_dims: tuple (n_scalar, n_vector)
    :param h_dim: intermediate number of vector channels, optional
    :param activations: tuple of functions (scalar_act, vector_act)
    :param vector_gate: whether to use vector gating.
                        (vector_act will be used as sigma^+ in vector gating if `True`)
    """

    def __init__(
        self,
        in_dims,
        out_dims,
        h_dim=None,
        activations=(F.relu, torch.sigmoid),
        vector_gate=False,
    ):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.dot_prod = LinearFullyConnectedDotProductLayer(self.vi, self.h_dim, self.h_dim)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        """
        :param x: tuple (s, V) of `torch.Tensor`,
                  or (if vectors_in is 0), a single `torch.Tensor`
        :return: tuple (s, V) of `torch.Tensor`,
                 or (if vectors_out is 0), a single `torch.Tensor`
        """
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vn = self.dot_prod(v)
            s = self.ws(torch.cat([s, vn], -1))
            vh = self.wh(v)
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 5, device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s

class GVPLayerNorm(nn.Module):
    '''
    Combined LayerNorm for tuples (s, V).
    Takes tuples (s, V) as input and as output.
    '''
    def __init__(self, dims):
        super(GVPLayerNorm, self).__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        
    def forward(self, x):
        '''
        :param x: tuple (s, V) of `torch.Tensor`,
                  or single `torch.Tensor` 
                  (will be assumed to be scalar channels)
        '''
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn


class GVPMPNN(nn.Module):
    def __init__(
        self,
        in_features_v,
        in_features_s,
        hidden_features_v,
        hidden_features_s,
        out_features_v,
        out_features_s
    ):
        super().__init__()
        self.edge_model = nn.Sequential(
            GVPLinear(
                        (in_features_s*2 + 1, in_features_v*2),
                        (hidden_features_s, hidden_features_v),
                    ),
            GVPLayerNorm((hidden_features_s, hidden_features_v))
        )

        self.node_model = nn.Sequential(
            GVPLinear(
                        (hidden_features_s*2, hidden_features_v*2),
                        (out_features_s, out_features_v),
                    ),
            GVPLayerNorm((hidden_features_s, hidden_features_v))
        )

    def message(self, x_i, x_j, pos_i, pos_j):
        s_rec, v_rec = x_i[0], x_i[1]
        s_send, v_send = x_j[0], x_j[1]
        s_input = torch.cat((s_rec, s_send, (pos_i - pos_j).unsqueeze(-1)), dim=-1)
        v_input = torch.cat((v_rec, v_send), dim=1)
        input = (s_input, v_input)
        h_msg = self.edge_model(input)
        return h_msg

    def update(self, h_agg, h):
        s_agg, v_agg = h_agg[0], h_agg[1]
        s, v = h[0], h[1]
        input_s = torch.cat([s, s_agg], dim=-1)
        input_v = torch.cat([v, v_agg], dim=1)
        input = (input_s, input_v)
        out_h = self.node_model(input)

        out_h_s, out_h_v = h[0] + out_h[0], h[1] + out_h[1]
        return (out_h_s, out_h_v)

    def forward(self, input, positions, edge_index):
        s, v = input
        s_send, v_send = s[edge_index[0]], v[edge_index[0]]
        s_rec, v_rec = s[edge_index[1]], v[edge_index[1]]
        x_j = (s_send, v_send)
        x_i = (s_rec, v_rec)
        pos_i, pos_j = positions[edge_index[0]], positions[edge_index[1]]

        num_messages = torch.bincount(edge_index[1]).unsqueeze(-1)
        h_msg = self.message(x_i, x_j, pos_i, pos_j)
        h_msg_s = global_add_pool(h_msg[0], edge_index[1]) / torch.sqrt(num_messages)
        h_msg_v = global_add_pool(h_msg[1].reshape(h_msg[1].shape[0], -1), edge_index[1]) / torch.sqrt(num_messages)
        h_msg_v = h_msg_v.reshape(h_msg_v.shape[0], -1, 3)
        h_agg = (h_msg_s, h_msg_v)

        out_h = self.update(h_agg, input)
        return out_h


class GVP(nn.Module):
    def __init__(
        self,
        in_features=1,
        hidden_features=64,
        hidden_features_v=16,
        out_features=1,
        num_layers=4,
        num_nearest_neighbors=16,
        num_tokens=20,
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        self.num_nearest_neighbors = num_nearest_neighbors
        self.feature_embedding = nn.Linear(in_features, hidden_features_v, bias=False)
        self.aa_embedding = nn.Embedding(num_tokens, hidden_features)
        self.inv_feature_embedding = nn.Linear(1, hidden_features)
        layers = []
        for i in range(num_layers):
            layers.append(
                GVPMPNN(hidden_features_v, hidden_features, hidden_features_v, hidden_features, hidden_features_v, hidden_features)
            )

        self.projection = GVPLinear(
                (hidden_features, hidden_features_v),
                (out_features, out_features),
                activations=(None, None),
            )

        self.model = nn.Sequential(*layers)

    def _featurization(self, seqs, coords, batch_idx):
        # invariant embeddings
        seqs = self.aa_embedding(seqs)

        # covariant embeddings
        coords_mean = global_mean_pool(coords, batch_idx)[batch_idx]
        coords_input = coords - coords_mean
        coords_output = self.feature_embedding(coords_input.unsqueeze(1).permute(0, 2, 1)).permute(0, 2, 1)
        return seqs, coords_output, coords_mean
    
    def _forward(self, seqs, coords, positions, edge_index, batch_idx):
        s, v, coords_mean = self._featurization(seqs, coords, batch_idx)
        x = (s, v)
        for layer in self.model:
            x = layer(x, positions, edge_index)
        s, v = self.projection(x)
        pred = v.squeeze(1) + coords_mean
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