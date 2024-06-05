import torch
from torch import nn

from algebra.cliffordalgebra import CliffordAlgebra
from cegnn_utils import MVLayerNorm, MVLinear, MVSiLU, SteerableGeometricProductLayer
from torch_geometric.nn import global_add_pool

import torch.nn.functional as F


class CEBlock(nn.Module):
    def __init__(self, algebra, in_features, out_features, normalization_init=0):
        super().__init__()

        self.algebra = algebra
        self.in_features = in_features
        self.out_features = out_features

        self.block = nn.Sequential(
            MVLinear(self.algebra, in_features, out_features),
            MVSiLU(self.algebra, out_features),
            SteerableGeometricProductLayer(
                self.algebra,
                out_features,
                normalization_init=normalization_init,
            ),
            MVLayerNorm(self.algebra, out_features),
            MVSiLU(self.algebra, out_features),
        )

    def forward(self, input):
        return self.block(input)


class CEMLP(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        n_layers=2,
        normalization_init=0,
    ):
        super().__init__()
        self.algebra = algebra
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.n_layers = n_layers

        layers = []

        # Add geometric product layers.
        for i in range(n_layers - 1):
            layers.append(
                CEBlock(
                    self.algebra,
                    in_features,
                    hidden_features,
                    normalization_init=normalization_init,
                )
            )
            in_features = hidden_features

        # Add final layer.
        layers.append(
            CEBlock(
                self.algebra,
                in_features,
                out_features,
                normalization_init=normalization_init,
            )
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class EGCL(nn.Module):
    def __init__(
        self,
        algebra,
        in_features,
        hidden_features,
        out_features,
        edge_attr_features=0,
        node_attr_features=0,
        residual=True,
        normalization_init=0,
    ):
        super().__init__()
        self.residual = residual
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.edge_attr_features = edge_attr_features
        self.node_attr_features = node_attr_features

        self.edge_model = CEMLP(
            algebra,
            self.in_features + self.edge_attr_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )

        self.node_model = CEMLP(
            algebra,
            self.in_features + self.out_features + node_attr_features,
            self.hidden_features,
            self.out_features,
            normalization_init=normalization_init,
        )
        self.algebra = algebra

    def message(self, h_i, h_j, edge_attr=None):
        h_i, h_j = self.algebra.split(h_i), self.algebra.split(h_j)
        if edge_attr is None:
            input = h_i - h_j
        else:
            input = torch.cat([h_i - h_j, edge_attr], dim=1)
        h_msg = self.edge_model(input)
        return h_msg

    def update(self, h_agg, h, node_attr):
        h_agg, h = self.algebra.split(h_agg), self.algebra.split(h)
        if node_attr is not None:
            input_h = torch.cat([h, h_agg, node_attr], dim=1)
        else:
            input_h = torch.cat([h, h_agg], dim=1)
        out_h = self.node_model(input_h)

        if self.residual:
            out_h = h + out_h
        return out_h

    def forward(self, h, edge_index, edge_attr=None, node_attr=None):
        h = self.algebra.flatten(h)
        h_mes = self.message(h[edge_index[0]], h[edge_index[1]], edge_attr)
        num_messages = torch.bincount(edge_index[0]).unsqueeze(-1)
        h_aggr = global_add_pool(self.algebra.flatten(h_mes), edge_index[0]) / num_messages
        h_aggr = self.algebra.split(h_aggr)
        h = self.update(h_aggr, h, node_attr)

        return h


class NBodyCliffordMPNN(nn.Module):
    def __init__(
        self,
        in_features=3,
        hidden_features=28,
        out_features=1,
        edge_features_in=1,
        num_layers=4,
        normalization_init=0,
        residual=True,
    ):
        super().__init__()
        self.algebra = CliffordAlgebra((1.0, 1.0, 1.0))
        self.hidden_features = hidden_features
        self.n_layers = num_layers

        self.embedding = MVLinear(
            self.algebra, in_features, hidden_features, subspaces=False
        )

        layers = []

        for i in range(0, num_layers):
            layers.append(
                EGCL(
                    self.algebra,
                    hidden_features,
                    hidden_features,
                    hidden_features,
                    edge_features_in,
                    residual=residual,
                    normalization_init=normalization_init,
                )
            )

        self.projection = nn.Sequential(
            MVLinear(self.algebra, hidden_features, out_features),
        )

        self.layers = nn.Sequential(*layers)

    def _forward(self, h, edges, edge_attr):
        h = self.embedding(h)
        for layer in self.layers:
            h = layer(h, edges, edge_attr=edge_attr)

        h = self.projection(h)
        return h

    def forward(self, batch, step, mode):
        batch = batch.to("cuda")
        batch_size = batch.ptr.shape[0] - 1
        loc = batch.loc.reshape(batch_size, -1, 3)
        loc_mean = (loc - loc.mean(dim=1, keepdim=True)).reshape(-1, 3)
        vel = batch.vel

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        edge_attr = self.algebra.embed(edge_attr[..., None], (0,))  # type: ignore

        invariants = batch.charges
        invariants = self.algebra.embed(invariants, (0,))  # type: ignore

        xv = torch.stack([loc_mean, vel], dim=1)
        covariants = self.algebra.embed(xv, (1, 2, 3))  # type: ignore

        input = torch.cat([invariants[:, None], covariants], dim=1)

        loc_pred = self._forward(input, edge_index, edge_attr)
        loc_pred = loc_pred[..., 0, 1:4]
        loc = loc.reshape(-1, 3)
        loc_pred = loc + loc_pred

        targets = batch.y.view(-1, 3)
        loss = F.mse_loss(loc_pred, targets, reduction="none").mean(dim=1)

        backprop_loss = loss.mean()  # []

        return backprop_loss, {"loss": loss}
