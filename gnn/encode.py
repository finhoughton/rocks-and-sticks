from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

import torch
from torch_geometric.data import Data

from game import Game
from players import RandomPlayer  # type: ignore

if TYPE_CHECKING:
    from models import Node

@dataclass
class EncodedGraph:
    data: Data
    perspective: int  # current player index for which the label/score is defined


def _node_feature(node: 'Node', num_players: int) -> list[float]:
    owner_one_hot = [0.0] * (num_players + 1)
    owner_idx = (node.rocked_by.number + 1) if node.rocked_by is not None else 0
    owner_one_hot[owner_idx] = 1.0
    deg = float(node.neighbour_count) / 8.0  # max degree is 8 (including diagonals)
    is_leaf = float(node.neighbour_count == 1)
    x = float(node.x)
    y = float(node.y)
    r2 = x * x + y * y
    return [*owner_one_hot, deg, is_leaf, x, y, r2]


def _edge_index_and_attr_from_points(points: Iterable['Node']) -> tuple[torch.Tensor, torch.Tensor]:
    edges: list[tuple[int, int]] = []
    attrs: list[list[float]] = []
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
            dx = float(nbr.x - p.x)
            dy = float(nbr.y - p.y)
            is_diag = 1.0 if abs(dx) == 1.0 and abs(dy) == 1.0 else 0.0
            orth = 1.0 - is_diag
            edges.append((a, b))
            attrs.append([orth, is_diag])
            edges.append((b, a))
            attrs.append([orth, is_diag])
    if not edges:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 2), dtype=torch.float32)
    src, dst = zip(*edges)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    return edge_index, edge_attr


def encode_game_to_graph(game: Game) -> EncodedGraph:
    """Encode the visible board graph and global features for the current player."""
    point_set = set(game.connected_points)
    point_set.update(game.rocks)
    nodes = sorted(point_set, key=lambda n: n.c)
    node_feats = [_node_feature(n, game.num_players) for n in nodes]
    x = torch.tensor(node_feats, dtype=torch.float32)
    edge_index, edge_attr = _edge_index_and_attr_from_points(nodes)
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
        edge_attr=edge_attr,
        batch=torch.zeros(len(nodes), dtype=torch.long),
        global_feats=global_feats,
    )
    # include node coordinates so downstream code can map moves -> node indices
    coords = torch.tensor([n.c for n in nodes], dtype=torch.long)
    data.node_coords = coords
    return EncodedGraph(data=data, perspective=game.current_player)

g = Game([RandomPlayer(0), RandomPlayer(1)])
SAMPLE_ENC = encode_game_to_graph(g)
del g
