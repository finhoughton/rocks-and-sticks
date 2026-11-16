"""The rs_engine C++ state must follow game.py exactly: fuzz them against each other."""

from __future__ import annotations

import random

import pytest

rs_engine = pytest.importorskip("rs_engine")

from game import Game
from models import PASS, D, Move
from players.base import RandomPlayer
from players.rs_player import sync_engine


def _py_moves(game: Game) -> set[tuple[int, int, str]]:
    out = set()
    for m in game.get_possible_moves(game.current_player):
        out.add((0, 0, "P") if m is PASS else (m.c[0], m.c[1], m.t))
    return out


def _assert_same(game: Game, eng, ctx: str) -> None:
    assert eng.current_player == game.current_player, ctx
    assert eng.turn == game.turn_number, ctx
    assert list(eng.scores) == list(game.players_scores), ctx
    assert list(eng.num_rocks) == list(game.num_rocks), ctx
    assert eng.winner == (-1 if game.winner is None else game.winner), ctx
    em = eng.legal_moves(False)
    es = set(em)
    assert len(em) == len(es), f"duplicate engine moves {ctx}"
    ps = _py_moves(game)
    assert es == ps, f"{ctx}: only_engine={sorted(es - ps)[:8]} only_py={sorted(ps - es)[:8]}"
    # the deduplicated list covers the same edges and rocks
    dd = set(eng.legal_moves(True))
    assert dd <= es


def _assert_closures(game: Game, eng, ctx: str) -> None:
    """The engine's closure list must predict exactly what game.py scores for each of those
    sticks, whichever end the stick is placed from."""
    for x, y, d, area2, _can0, _can1 in eng.closures():
        dx, dy = D[d].delta
        for m in (Move(x, y, d), Move(x + dx, y + dy, D[d].reversed.name)):
            for player in (0, 1):
                if not game.valid_move(m, player):
                    continue
                before = game.players_scores[player]
                game.do_move(player, m)
                gained = game.players_scores[player] - before
                game.undo_move()
                assert gained == (area2 if area2 >= 2 else 0), f"{ctx}: closure {m} engine area {area2} python gain {gained}"


def _pick(game: Game, rng: random.Random, style: int) -> Move:
    moves = list(game.get_possible_moves(game.current_player))
    sticks = [m for m in moves if m is not PASS and m.t != "R"]
    rocks = [m for m in moves if m is not PASS and m.t == "R"]
    joins = []
    for m in sticks:
        end = (m.c[0] + D[m.t].delta[0], m.c[1] + D[m.t].delta[1])
        node = game.points.get(end)
        if node is not None and node.connected:
            joins.append(m)
    r = rng.random()
    if style == 0:  # mostly uniform
        if r < 0.03:
            return PASS
        return rng.choice([m for m in moves if m is not PASS] or [PASS])
    if style == 1:  # dense: prefer sticks that join existing vertices (many cycles)
        if joins and r < 0.6:
            return rng.choice(joins)
        if sticks and r < 0.9:
            return rng.choice(sticks)
        return rng.choice(rocks or sticks or [PASS])
    # style 2: compact structure near the origin, to reach high-degree vertices
    near = [m for m in sticks if abs(m.c[0]) <= 1 and abs(m.c[1]) <= 1]
    if near and r < 0.8:
        return rng.choice(near)
    return rng.choice(sticks or rocks or [PASS])


@pytest.mark.parametrize("style", [0, 1, 2])
def test_fuzz_parity(style: int) -> None:
    n_games = 60 if style == 0 else 40
    for g in range(n_games):
        rng = random.Random(1000 * style + g)
        game = Game([RandomPlayer(0), RandomPlayer(1)])
        eng = rs_engine.Engine(10)
        _assert_same(game, eng, f"style {style} game {g} start")
        for ply in range(45):
            m = _pick(game, rng, style)
            t = "P" if m is PASS else m.t
            game.do_move(game.current_player, m)
            eng.do_move(m.c[0], m.c[1], t)
            _assert_same(game, eng, f"style {style} game {g} ply {ply} after {m}")
            if rng.random() < 0.3:
                _assert_closures(game, eng, f"style {style} game {g} ply {ply}")
            if rng.random() < 0.1:
                # a fresh engine synced from the Python position agrees too
                e2 = rs_engine.Engine(10)
                sync_engine(e2, game)
                _assert_same(game, e2, f"style {style} game {g} ply {ply} resync")


def test_undo_restores_state() -> None:
    rng = random.Random(7)
    game = Game([RandomPlayer(0), RandomPlayer(1)])
    eng = rs_engine.Engine(10)
    for ply in range(30):
        m = _pick(game, rng, 1)
        t = "P" if m is PASS else m.t
        before = (eng.legal_moves(False), list(eng.scores), eng.winner, eng.turn, eng.current_player)
        eng.do_move(m.c[0], m.c[1], t)
        eng.undo_move()
        after = (eng.legal_moves(False), list(eng.scores), eng.winner, eng.turn, eng.current_player)
        assert before == after
        game.do_move(game.current_player, m)
        eng.do_move(m.c[0], m.c[1], t)


def test_claimed_vertex_star() -> None:
    """A vertex with sticks to all eight neighbours becomes unreachable in game.py's flood fill."""
    game = Game([RandomPlayer(0), RandomPlayer(1)])
    eng = rs_engine.Engine(10)
    # spokes from (0,1): the start stick already gives the S spoke
    for name in ["N", "E", "W", "NE", "NW", "SE", "SW"]:
        m = Move(0, 1, name)
        assert game.valid_move(m, game.current_player)
        game.do_move(game.current_player, m)
        eng.do_move(0, 1, name)
        _assert_same(game, eng, f"star after {name}")
    assert game.coord_in_claimed_region((0, 1))
