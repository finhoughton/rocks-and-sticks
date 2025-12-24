from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # type: ignore
from torch_geometric.nn import GINEConv, global_mean_pool  # type: ignore

from game import Game
from gnn.encode import encode_game_to_graph

INIT_MEAN = 0.0
INIT_STD = 0.1

def _init_parameters(module: nn.Module, mean: float = INIT_MEAN, std: float = INIT_STD) -> None:
    for p in module.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, mean=mean, std=std)



class GNNEval(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        global_feat_dim: int,
        hidden: int = 256,
        num_hidden_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__() # type: ignore
        self.dropout_p = dropout

        def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

        dims = [node_feat_dim] + [hidden] * num_hidden_layers
        edge_dim = 2  # [orth, diag]
        self.convs = nn.ModuleList([GINEConv(mlp(dims[i], dims[i + 1]), edge_dim=edge_dim) for i in range(num_hidden_layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(dims[i + 1]) for i in range(num_hidden_layers)])

        self.head = nn.Sequential(
            nn.Linear(hidden + global_feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        _init_parameters(self)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr, batch, g = data.x, data.edge_index, data.edge_attr, data.batch, data.global_feats
        h: torch.Tensor = x # type: ignore
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)
            if h.shape == h_in.shape:
                h = h + h_in
        pooled = global_mean_pool(h, batch)  # type: ignore
        logits = self.head(torch.cat([pooled, g], dim=-1))
        return logits.squeeze(-1)


_model: GNNEval | None = None
_device: torch.device = torch.device("cpu")


def init_random_model(
    node_feat_dim: int,
    global_feat_dim: int,
    device: str | torch.device = "cpu",
    mean: float = INIT_MEAN,
    std: float = INIT_STD,
) -> None:
    """Create a fresh GNNEval with weights sampled from N(mean, std)."""
    global _model, _device
    _device = torch.device(device)
    model = GNNEval(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim)
    _init_parameters(model, mean=mean, std=std)
    model.to(_device)
    model.eval()
    _model = model


def load_model(path: str, node_feat_dim: int, global_feat_dim: int, device: str | torch.device = "cpu") -> None:
    """Load a saved state_dict into a GNNEval instance."""
    global _model, _device
    _device = torch.device(device)
    model = GNNEval(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim)
    state = torch.load(path, map_location=_device)
    model.load_state_dict(state)
    model.to(_device)
    model.eval()
    _model = model


def evaluate_game(game: Game) -> float:
    """Return a heuristic score: probability current player eventually wins."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model(...) first.")

    encoded = encode_game_to_graph(game)
    data = encoded.data.to(_device)
    with torch.no_grad():
        logit = _model(data)
        prob = torch.sigmoid(logit).item()
    return prob
