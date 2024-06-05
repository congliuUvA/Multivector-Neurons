import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from algebra.cliffordalgebra import CliffordAlgebra
from cegnn_utils import MVLinear, MVLayerNorm
import math
EPS = 1e-8

class MVGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.des = "MVGELU"
    
    def forward(self, x):
        scalar = x[..., 0]
        gates = nn.Sigmoid()(math.sqrt(2 / math.pi) * (2 * (scalar + 0.044715 * torch.pow(scalar, 3))))
        return gates.unsqueeze(-1) * x
    
class LinearFullyConnectedDotProductLayer(nn.Module):
    def __init__(self, in_vec_dims, hidden_vec_dims, in_scalar_dims, out_scalar_dims, residual=False):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        self.linear_left = MVLinear(self.algebra, in_vec_dims, hidden_vec_dims, subspaces=False, bias=True)
        self.linear_right = MVLinear(self.algebra, in_vec_dims, hidden_vec_dims, subspaces=False, bias=True)
        self.linear_out = nn.Linear(hidden_vec_dims, out_scalar_dims, bias=False)
        self.hidden_vec_dims = hidden_vec_dims
        self.residual = residual 

    def get_invariants(self, input, algebra):
        norms = algebra.qs(input, grades=algebra.grades[1:])
        return torch.cat([input[..., :1], *norms], dim=-1)
    
    def forward(self, vec):
        # normalization
        vec_right = self.linear_right(vec)
        vec_left = self.linear_left(vec)
        # dot product
        dot = self.algebra.b(vec_left, vec_right).squeeze(dim=-1)
        dot = self.linear_out(dot)
        return dot
    
class LinearFullyConnectedGPLayer(nn.Module):
    def __init__(self, in_vec_dims, hidden_vec_dims, out_vec_dims):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        self.linear_left = MVLinear(self.algebra, in_vec_dims, hidden_vec_dims, subspaces=False, bias=True)
        self.linear_right = MVLinear(self.algebra, in_vec_dims, hidden_vec_dims, subspaces=False, bias=True)
        self.linear_out =  MVLinear(self.algebra, hidden_vec_dims, out_vec_dims, subspaces=False, bias=True)
        self.vec_norm = MVLayerNorm(self.algebra, out_vec_dims)

    def forward(self, vec1, vec2=None):
        # normalization
        vec_right = self.linear_right(vec1)
        vec = vec1 if vec2 is None else vec2
        vec_left = self.linear_left(vec)

        # geometric product
        vec_out = self.algebra.geometric_product(vec_left, vec_right)
        vec_out = self.linear_out(vec_out)
        vec_out = self.vec_norm(vec_out) 
        return vec_out

class CVPLinear(nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        h_dim=None,
        activations=(F.relu, torch.sigmoid),
        vector_gate=False,
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1,))
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate
        self.mvgelu = MVGELU()
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.dot_prod = LinearFullyConnectedDotProductLayer(self.vi, self.h_dim, self.si, self.h_dim)
            self.wh = MVLinear(self.algebra, self.vi, self.h_dim, subspaces=False, bias=True)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = MVLinear(self.algebra, self.h_dim, self.vo, subspaces=False, bias=True)
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
            vn = self.dot_prod(v)
            s = self.ws(torch.cat([s, vn], -1))
            vh = self.wh(v)
            if self.vo:
                v = self.wv(vh)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v
                    # v = self.mvgelu(v)
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, self.algebra.dim, device=self.dummy_param.device)
        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s

class CVPGeometricProductLayer(nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        h_dim=None,
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1,))
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.h_dim = h_dim or max(self.vi, self.vo)
        self.gp_prod = LinearFullyConnectedGPLayer(self.vi, self.h_dim, self.vo)
        self.mvgelu = MVGELU()

        self.wv = MVLinear(self.algebra, self.h_dim, self.vo, subspaces=False, bias=True)
        self.s2v = nn.Linear(self.si + self.h_dim, self.vo)

    def forward(self, x):
        s, v = x
        v_gp = self.gp_prod(v)

        # add first layer
        v = v_gp + v
        v = self.mvgelu(v)
        v = self.wv(v)

        # scalar vector mixture
        scalar = torch.cat([v[..., 0], s], dim=-1)
        scalar = self.s2v(scalar)
        v[..., 0] = scalar

        return (s, v) if self.vo else s

class CVPLayerNorm(nn.Module):
    def __init__(self, dims):
        super(CVPLayerNorm, self).__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)
        self.vec_norm = MVLayerNorm(self.algebra, self.v)
        
    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        return self.scalar_norm(s), self.vec_norm(v)


class CVPMPNN(nn.Module):
    def __init__(
        self,
        in_features_v,
        in_features_s,
        hidden_features_v,
        hidden_features_s,
        out_features_v,
        out_features_s,
        use_gp_layers=True
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1,))
        self.edge_model = nn.Sequential(
            CVPLinear(
                        (in_features_s*2, in_features_v*2),
                        (hidden_features_s, hidden_features_v),
                        vector_gate=True,
                    ),
            CVPGeometricProductLayer(
                        (hidden_features_s, hidden_features_v),
                        (hidden_features_s, hidden_features_v),
            ) if use_gp_layers else nn.Identity(),
            CVPLayerNorm((hidden_features_s, hidden_features_v)) if use_gp_layers else nn.Identity(),
        )

        self.node_model = nn.Sequential(
            CVPLinear(
                        (hidden_features_s*2, hidden_features_v*2),
                        (hidden_features_s, hidden_features_v),
                        vector_gate=True,
                    ),
            CVPGeometricProductLayer(
                        (hidden_features_s, hidden_features_v),
                        (out_features_s, out_features_v),
            ),
            CVPLayerNorm((out_features_s, out_features_v))
        )

    def message(self, x_i, x_j):
        s_send, v_send = x_i[0], x_i[1]
        s_rec, v_rec = x_j[0], x_j[1]
        s_input = torch.cat((s_rec, s_send), dim=-1)
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

    def forward(self, input, edge_index):
        s, v = input
        s_send, v_send = s[edge_index[0]], v[edge_index[0]]
        s_rec, v_rec = s[edge_index[1]], v[edge_index[1]]
        x_i = (s_send, v_send)
        x_j = (s_rec, v_rec)
        
        s_msg, v_msg = self.message(x_i, x_j)
        h_msg = (s_msg, v_msg)
        num_messages = torch.bincount(edge_index[1]).unsqueeze(-1)
        h_msg_s = global_add_pool(h_msg[0], edge_index[1]) / torch.sqrt(num_messages)
        h_msg_v = global_add_pool(self.algebra.flatten(h_msg[1]), edge_index[1]) / torch.sqrt(num_messages)
        h_msg_v = self.algebra.split(h_msg_v)
        h_agg = (h_msg_s, h_msg_v)
        
        out_s, out_v = self.update(h_agg, input)

        return (out_s, out_v)



class NbodyCVP(nn.Module):
    def __init__(
        self,
        in_features_s=1,
        in_features_v=3,
        hidden_features_s=64,
        hidden_features_v=16,
        out_features_s=1,
        out_features_v=1,
        num_layers=3,
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        self.feature_embedding = CVPLinear(
                (in_features_s, in_features_v),
                (hidden_features_s, hidden_features_v),
                activations=(None, None),
            )
        layers = []
        for i in range(num_layers):
            layers.append(
                CVPMPNN(hidden_features_v, hidden_features_s, hidden_features_v, hidden_features_s, hidden_features_v, hidden_features_s)
            )

        self.projection = CVPLinear(
                (hidden_features_s, hidden_features_v),
                (out_features_s, out_features_v),
                activations=(None, None),
            )

        self.model = nn.Sequential(*layers)


    def _forward(self, x, edge_index, batch):
        x = self.feature_embedding(x)
        for layer in self.model:
            x = layer(x, edge_index)
        x = self.projection(x)
        s, v = x
        return v
    
    def forward(self, batch, step, mode):
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
        cl_charge = self.algebra.embed_grade(charges.unsqueeze(-1), 0)
        input = torch.cat([input, vel, cl_charge], dim=1)

        pred_pos = self._forward((charges, input), edge_index, batch.batch)
        pred_pos = pred_pos[:, 0, 1:4]
        pred_pos = pred_pos.reshape(batch_size, -1, 3) + coords
        pred_pos = pred_pos.reshape(-1,3)
        loss = F.mse_loss(pred_pos, batch.y.reshape(-1, 3), reduction="none").mean(dim=1)

        return (
            loss.mean(),
            {"loss":  loss,},
        )