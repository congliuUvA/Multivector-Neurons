import torch
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
        self.linear_left = MVLinear(self.algebra, in_vec_dims, hidden_vec_dims, subspaces=True, bias=True)
        self.linear_right = MVLinear(self.algebra, in_vec_dims, hidden_vec_dims, subspaces=True, bias=True)
        self.linear_out =  MVLinear(self.algebra, hidden_vec_dims + in_vec_dims, out_vec_dims, subspaces=True, bias=True)
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


class EGNN_C_Block(nn.Module):
    """ E(n)-equivariant Message Passing Layer """
    def __init__(self, node_features_s, hidden_features_s, node_features_v, edge_features, subspaces=False):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1,))
        self.node_features_s = node_features_s
        self.node_features_v = node_features_v
        self.edge_features = edge_features
        self.hidden_features_s = hidden_features_s
        self.factor = self.algebra.n_subspaces if subspaces==True else 1
        self.subspaces = subspaces
        self.message_net = nn.Sequential(nn.Linear(2 * node_features_s + edge_features, hidden_features_s),
                                         nn.ReLU(),
                                         nn.Linear(hidden_features_s, hidden_features_s))
    
        self.update_net = nn.Sequential(nn.Linear(node_features_s + hidden_features_s, hidden_features_s),
                                        nn.ReLU(),
                                        nn.Linear(hidden_features_s, hidden_features_s))
        
        self.pos_net = nn.Sequential(nn.Linear(hidden_features_s, hidden_features_s),
                                     nn.ReLU(),
                                     nn.Linear(hidden_features_s, node_features_v * self.factor))

        self.v_update = LinearFullyConnectedGPLayer(node_features_v, node_features_v, node_features_v)
        self.s_layernorm = nn.LayerNorm(node_features_v * self.factor)
        self.v = MVLinear(self.algebra, node_features_v, node_features_v)


    def get_invariants(self, input, algebra):
        norms = algebra.qs(input, grades=algebra.grades[1:]) 
        norms = torch.sqrt(torch.cat(norms, dim=-1) + 1e-8)
        return torch.cat([input[..., :1], norms], dim=-1)

    def forward(self, s, v, edge_index):
        send_idx, rec_idx = edge_index
        s_i, s_j = s[send_idx], s[rec_idx]
        v_i, v_j = v[send_idx], v[rec_idx]  
        message, pos_message = self.message(s_i, s_j, v_i, v_j)
        num_messages = torch.bincount(send_idx).unsqueeze(-1)
        message_aggr = global_add_pool(message, rec_idx)
        message_aggr = message_aggr / torch.sqrt(num_messages)
        pos_message_aggr = self.algebra.split(global_add_pool(self.algebra.flatten(pos_message), rec_idx))
        pos_message_aggr = self.algebra.split(self.algebra.flatten(pos_message_aggr) / torch.sqrt(num_messages))
        s, v = self.update(message_aggr, pos_message_aggr, s, v)
        return s, v

    def message(self, s_i, s_j, v_i, v_j):
        """ Create messages """
        v_ij = self.v(v_j - v_i)
        edge_attr = (v_ij * v_ij).sum(dim=-1)
        input = [s_i, s_j, edge_attr]
        input = torch.cat(input, dim=-1)
        message = self.message_net(input)
        pos_message = self.pos_net(message)
        pos_message = v_ij * (pos_message.reshape(pos_message.shape[0], -1, self.algebra.n_subspaces).repeat_interleave(self.algebra.subspaces, dim=-1)) if self.subspaces else v_ij * pos_message.unsqueeze(-1)
        return message, pos_message

    def update(self, message, pos_message, s, v):
        # Update node features
        input = torch.cat((s, message), dim=-1)
        output = self.update_net(input)
        update = s + output
        # Update positions
        v_update = self.v_update(pos_message) + v
        return update, v_update


class NBodyEGNN_C(nn.Module):
    def __init__(
        self,
        in_features=2,
        hidden_features=64,
        hidden_features_v=16,
        out_features=1,
        num_layers=4,
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1,1,1))
        self.feature_embedding = MVLinear(self.algebra, in_features, hidden_features_v, subspaces=False, bias=True)
        self.inv_feature_embedding = nn.Linear(1, hidden_features)
        layers = []
        for i in range(num_layers):
            layers.append(
                EGNN_C_Block(hidden_features, hidden_features, hidden_features_v, hidden_features_v, subspaces=True)
            )

        self.projection =  MVLinear(self.algebra, hidden_features_v, out_features, subspaces=False, bias=True)

        self.model = nn.Sequential(*layers)

    def _forward(self, s, x, edge_index, batch):
        x = self.feature_embedding(x)
        for layer in self.model:
            s, x = layer(s, x, edge_index)
        x = self.projection(x)
        return x
    
    def forward(self, batch, step, mode):
        batch = batch.to("cuda")
        batch_size = batch.ptr.shape[0] - 1
        coords = batch.loc.reshape(batch_size, -1, 3)

        mean_pos = coords.mean(dim=1, keepdims=True)
        input_pos = coords - mean_pos
        num_nodes = input_pos.shape[1]
        edge_index = batch.edge_index
        charges = batch.charges

        # invariant features
        s = self.inv_feature_embedding(charges)

        # covariant features
        input_pos = input_pos.reshape(batch_size*num_nodes, -1).unsqueeze(1)
        input_pos = self.algebra.embed_grade(input_pos, 1)
        vel = self.algebra.embed_grade(batch.vel.unsqueeze(1), 1)
        charges = self.algebra.embed_grade(charges.unsqueeze(-1), 0)
        x = torch.cat([input_pos, vel], dim=1)
        
        pred_pos = self._forward(s, x, edge_index, batch.batch)
        pred_pos = pred_pos[:, 0, 1:4]
        pred_pos = pred_pos.reshape(batch_size, -1, 3) + coords
        pred_pos = pred_pos.reshape(-1,3)
        loss = F.mse_loss(pred_pos, batch.y.reshape(-1, 3), reduction="none").mean(dim=1)

        return (
            loss.mean(),
            {"loss":  loss,},
        )


