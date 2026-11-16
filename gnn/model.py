from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as Data
from torch_geometric.nn import GINEConv, global_mean_pool  # type: ignore
from torch_geometric.nn.norm import GraphNorm  # type: ignore

from game import Game

if TYPE_CHECKING:
    from game import GameProtocol as Game
from gnn.encode import EncodedGraph, encode_game_to_graph

INIT_MEAN = 0.0
INIT_STD = 0.4
MOVE_FEAT_DIM = 16  # [x, y, is_pass, is_rock, end_x, end_y, dx, dy, dir_onehot(8)]

def _init_parameters(module: nn.Module, mean: float = INIT_MEAN, std: float = INIT_STD) -> None:
    for p in module.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, mean=mean, std=std)



class GNNEval(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        global_feat_dim: int,
        hidden: int = 384,
        num_hidden_layers: int = 5,
        dropout: float = 0.1,
        move_feat_dim: int = MOVE_FEAT_DIM,
    ) -> None:
        super().__init__() # type: ignore
        self.dropout_p = dropout
        self.move_feat_dim = move_feat_dim

        def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

        dims = [node_feat_dim] + [hidden] * num_hidden_layers
        edge_dim = 2  # [orth, diag]
        self.convs = nn.ModuleList([GINEConv(mlp(dims[i], dims[i + 1]), edge_dim=edge_dim) for i in range(num_hidden_layers)])
        self.norms = nn.ModuleList([GraphNorm(dims[i + 1]) for i in range(num_hidden_layers)])

        self.head = nn.Sequential(
            nn.Linear(hidden + global_feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        # Policy head: scores a single move given the backbone's pooled embedding.
        # Input: cat([pooled(hidden), global_feats(gfd), move_feat(mfd)])
        policy_in = hidden + global_feat_dim + move_feat_dim
        self.policy_mlp = nn.Sequential(
            nn.Linear(policy_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        _init_parameters(self)

    def backbone_forward(self, data: Data.Data) -> torch.Tensor:
        """Run GNN backbone and return cat([pooled, global_feats])."""
        x, edge_index, edge_attr, batch, g = data.x, data.edge_index, data.edge_attr, data.batch, data.global_feats
        h: torch.Tensor = x  # type: ignore
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = self.norms[i](h, batch)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)
            if h.shape == h_in.shape:
                h = h + h_in
        pooled = global_mean_pool(h, batch)  # type: ignore
        return torch.cat([pooled, g], dim=-1)  # [B, hidden + gfd]

    def forward(self, data: Data.Data) -> torch.Tensor:
        pooled_cat = self.backbone_forward(data)
        logits = self.head(pooled_cat)
        return logits.squeeze(-1)

    def policy_forward(self, pooled_cat: torch.Tensor, move_feats: torch.Tensor) -> torch.Tensor:
        """Score moves given backbone output and per-move features.

        Args:
            pooled_cat: [B, hidden + gfd] from backbone_forward
            move_feats: [B, K, move_feat_dim] features for K candidate moves

        Returns:
            logits: [B, K] raw scores (apply softmax for probabilities)
        """
        B, K, _ = move_feats.shape
        # Expand pooled to match each move: [B, 1, hidden+gfd] -> [B, K, hidden+gfd]
        expanded = pooled_cat.unsqueeze(1).expand(B, K, -1)
        # cat along feature dim: [B, K, hidden+gfd+mfd]
        combined = torch.cat([expanded, move_feats], dim=-1)
        return self.policy_mlp(combined).squeeze(-1)  # [B, K]


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
    return evaluate_encoding(encode_game_to_graph(game))

def evaluate_encoding(enc: EncodedGraph) -> float:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model(...) first.")
    data = enc.data.to(_device)
    with torch.no_grad():
        logit = _model(data)
        prob = torch.sigmoid(logit).item()
    return prob

def evaluate_encodings(encs: list[EncodedGraph]) -> list[float]:
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model(...) first.")
    if not encs:
        return []
    datas = []
    for e in encs:
        try:
            d = e.data
        except Exception:
            d = e
        datas.append(d)
    if not datas:
        return []
    batch = Data.Batch.from_data_list(datas).to(_device) # type: ignore
    with torch.no_grad():
        logits = _model(batch)
        probs = torch.sigmoid(logits).cpu().tolist()
    return probs
