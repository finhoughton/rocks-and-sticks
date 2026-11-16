"""Symmetry augmentation must transform the (x, y) node features, not (y, r2)."""

from __future__ import annotations

import torch

from game import Game
from gnn.augment import _augment_symmetries
from gnn.encode import encode_game_to_graph
from models import Move
from players import RandomPlayer


def test_augmentation_transforms_xy_and_keeps_r2() -> None:
    g = Game([RandomPlayer(0), RandomPlayer(1)])
    g.do_move(0, Move(0, 1, "E"))
    g.do_move(1, Move(1, 1, "NE"))
    enc = encode_game_to_graph(g)
    x = enc.data.x
    xs, ys, r2 = x[:, 5], x[:, 6], x[:, 7]
    assert torch.equal(r2, xs * xs + ys * ys)

    outs = _augment_symmetries(enc)
    assert len(outs) == 8
    expected = [(xs, ys), (-xs, ys), (xs, -ys), (-xs, -ys), (ys, xs), (-ys, xs), (ys, -xs), (-ys, -xs)]
    for data, (ex, ey) in zip(outs, expected):
        assert torch.equal(data.x[:, 5], ex)
        assert torch.equal(data.x[:, 6], ey)
        assert torch.equal(data.x[:, 7], r2)
        assert torch.equal(data.x[:, :5], x[:, :5])
