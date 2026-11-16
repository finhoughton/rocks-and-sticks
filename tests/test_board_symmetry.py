"""The rules are symmetric under the four symmetries of the starting stick (0,0)-(0,1):
left-right (x -> -x), top-bottom (y -> 1 - y) and both (a half turn). The opening book relies
on this, so check it: play random games next to their images and compare everything."""

from __future__ import annotations

import random

import pytest

from game import Game
from models import PASS
from players.base import RandomPlayer
from players.rs_book import SYMMETRIES, transform_move, to_move
from tests.test_rs_engine_parity import _pick


def _moves(game: Game) -> set[tuple[int, int, str]]:
    return {(0, 0, "P") if m is PASS else (m.c[0], m.c[1], m.t) for m in game.get_possible_moves(game.current_player)}


@pytest.mark.parametrize("sym", SYMMETRIES[1:])
@pytest.mark.parametrize("style", [0, 1, 2])
def test_games_and_their_images_play_out_identically(sym: tuple[bool, bool], style: int) -> None:
    for g in range(15):
        rng = random.Random(7000 + 100 * style + g)
        a = Game([RandomPlayer(0), RandomPlayer(1)])
        b = Game([RandomPlayer(0), RandomPlayer(1)])
        for ply in range(40):
            if a.winner is not None:
                break
            ma = {transform_move(m, sym) for m in _moves(a)}
            assert ma == _moves(b), f"{sym} style {style} game {g} ply {ply}: legal moves differ"
            m = _pick(a, rng, style)
            t = (0, 0, "P") if m is PASS else (m.c[0], m.c[1], m.t)
            a.do_move(a.current_player, m)
            b.do_move(b.current_player, to_move(transform_move(t, sym)))
            assert a.players_scores == b.players_scores, f"{sym} style {style} game {g} ply {ply}"
            assert a.num_rocks == b.num_rocks and a.winner == b.winner
