"""The rules treat every direction and every place on the grid alike: a position mirrored,
turned or shifted is the same position. The opening book relies on this, so check it: play
random games next to their images and compare everything."""

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


# (symmetry, moves played first in the game, moves played first in its image): the mirrors that
# leave the starting stick where it is, then a diagonal mirror and a shift, which need a first
# move each so that both games can start from the usual starting stick
CASES = [(sym, [], []) for sym in SYMMETRIES[1:]] + [
    ((True, False, False, -1, 0), [(0, 1, "E")], [(0, 0, "W")]),
    ((False, False, False, 0, -1), [(0, 1, "N")], [(0, 0, "S")]),
]


@pytest.mark.parametrize("sym,first_a,first_b", CASES)
@pytest.mark.parametrize("style", [0, 1, 2])
def test_games_and_their_images_play_out_identically(sym, first_a, first_b, style: int) -> None:
    for g in range(15):
        rng = random.Random(7000 + 100 * style + g)
        a = Game([RandomPlayer(0), RandomPlayer(1)])
        b = Game([RandomPlayer(0), RandomPlayer(1)])
        for game, first in ((a, first_a), (b, first_b)):
            for m in first:
                game.do_move(game.current_player, to_move(m))
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
