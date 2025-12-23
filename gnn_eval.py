from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # type: ignore
from torch_geometric.nn import GINConv, global_mean_pool  # type: ignore

if TYPE_CHECKING:
    from game import Game
    from models import Node


INIT_MEAN = 0.0
INIT_STD = 0.1

def _init_parameters(module: nn.Module, mean: float = INIT_MEAN, std: float = INIT_STD) -> None:
    for p in module.parameters():
        if p.requires_grad:
            torch.nn.init.normal_(p, mean=mean, std=std)


@dataclass
class EncodedGraph:
    data: Data
    perspective: int  # current player index for which the label/score is defined


def _node_feature(node: Node, num_players: int) -> list[float]:
    owner_one_hot = [0.0] * (num_players + 1)
    owner_idx = (node.rocked_by.number + 1) if node.rocked_by is not None else 0
    owner_one_hot[owner_idx] = 1.0
    deg = float(node.neighbour_count) / 8.0  # max degree is 8 (including diagonals)
    is_leaf = float(node.neighbour_count == 1)
    x = float(node.x)
    y = float(node.y)
    r2 = x * x + y * y
    return [*owner_one_hot, deg, is_leaf, x, y, r2]


def _edge_index_from_points(points: Iterable[Node]) -> torch.Tensor:
    edges: set[tuple[int, int]] = set()
    nodes = list(points)
    idx_map = {p: i for i, p in enumerate(nodes)}
    for p in nodes:
        for nbr in p.neighbours:
            if nbr is None:
                continue
            a = idx_map[p]
            b = idx_map.get(nbr)
            if b is None:
                continue
            edges.add((a, b))
            edges.add((b, a))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    src, dst = zip(*sorted(edges))
    return torch.tensor([src, dst], dtype=torch.long)


def encode_game_to_graph(game: Game) -> EncodedGraph:
    """Encode the visible board graph and global features for the current player."""
    # Limit to connected points plus rock locations to keep the graph compact.
    point_set = set(game.connected_points)
    point_set.update(game.rocks)

    nodes = sorted(point_set, key=lambda n: n.c)
    node_feats = [_node_feature(n, game.num_players) for n in nodes]
    x = torch.tensor(node_feats, dtype=torch.float32)
    edge_index = _edge_index_from_points(nodes)

    turn = float(game.turn_number)
    cur_one_hot = [1.0 if i == game.current_player else 0.0 for i in range(game.num_players)]
    scores = [float(s) for s in game.players_scores]
    rocks_left = [float(r) for r in game.num_rocks]
    rocks_placed = [float(sum(1 for p in game.rocks if p.rocked_by == game.players[i])) for i in range(game.num_players)]
    max_r2 = max((float(n.x * n.x + n.y * n.y) for n in nodes), default=0.0)
    global_feats = torch.tensor([turn, *cur_one_hot, *scores, *rocks_left, *rocks_placed, max_r2], dtype=torch.float32).unsqueeze(0)

    data = Data(
        x=x,
        edge_index=edge_index,
        batch=torch.zeros(len(nodes), dtype=torch.long),
        global_feats=global_feats,
    )
    return EncodedGraph(data=data, perspective=game.current_player)


class GNNEval(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        global_feat_dim: int,
        hidden: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__() # type: ignore
        self.dropout_p = dropout

        def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

        dims = [node_feat_dim] + [hidden] * num_layers
        self.convs = nn.ModuleList([GINConv(mlp(dims[i], dims[i + 1]), train_eps=True) for i in range(num_layers)])

        self.head = nn.Sequential(
            nn.Linear(hidden + global_feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

        _init_parameters(self)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch, g = data.x, data.edge_index, data.batch, data.global_feats
        h: torch.Tensor = x # type: ignore
        for conv in self.convs:
            h_in = h
            h = conv(h, edge_index)
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
