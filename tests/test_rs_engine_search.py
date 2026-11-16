"""Tactical sanity checks for the rs engine search."""

from __future__ import annotations

import pytest

rs_engine = pytest.importorskip("rs_engine")

from game import Game
from models import PASS, Move
from players.base import RandomPlayer
from players.rs_player import RSPlayer, sync_engine

MATE_BOUND = 29000


def _game(moves: list[Move]) -> Game:
    g = Game([RandomPlayer(0), RandomPlayer(1)])
    for m in moves:
        assert g.valid_move(m, g.current_player), m
        g.do_move(g.current_player, m)
    return g


def _search(g: Game, nodes: int = 50_000) -> dict:
    e = rs_engine.Engine(16)
    sync_engine(e, g)
    return e.search(0, 0, nodes)


def test_player0_takes_the_square() -> None:
    # three sides of a unit square exist; player 0 closes it and player 1 cannot answer
    g = _game([Move(0, 1, "E"), Move(1, 1, "S")])
    r = _search(g)
    assert r["score"] > MATE_BOUND
    x, y, t = r["move"]
    g.do_move(0, Move(x, y, t))
    assert g.players_scores[0] == 2
    g.do_move(1, PASS)
    assert g.winner == 0


def test_player1_wins_at_once_when_it_can_close() -> None:
    g = _game([Move(0, 1, "E"), Move(1, 1, "S"), PASS])
    r = _search(g)
    assert r["score"] > MATE_BOUND
    x, y, t = r["move"]
    g.do_move(1, Move(x, y, t))
    assert g.winner == 1


def test_does_not_set_up_the_opponent() -> None:
    # player 1 must not add the third side of the square (player 0 would close it)
    g = _game([Move(0, 1, "E")])
    r = _search(g)
    assert r["score"] > -MATE_BOUND
    x, y, t = r["move"]
    g.do_move(1, Move(x, y, t) if t != "P" else PASS)
    assert _search(g)["score"] < MATE_BOUND


def test_exclusive_closure_cannot_be_stolen() -> None:
    # Player 0 rocks both ends of the last side of a square, then adds the third side: only
    # player 0 can close, and the engine must see it wins.
    g = _game(
        [
            Move(0, 1, "E"),   # P0: (0,1)-(1,1)
            Move(0, 1, "W"),   # P1: harmless
            Move(0, 0, "R"),   # P0 rocks (0,0)
            Move(-1, 1, "W"),  # P1: harmless
            Move(1, 0, "R"),   # P0 rocks (1,0): the edge (0,0)-(1,0) is now exclusive
            Move(-2, 1, "W"),  # P1: harmless
        ]
    )
    r = _search(g, 200_000)
    assert r["score"] > MATE_BOUND


def test_full_game_through_python_rules() -> None:
    players = [RSPlayer(0, time_ms=0, max_nodes=3000, seed=1, root_margin=10), RSPlayer(1, time_ms=0, max_nodes=3000, seed=2, root_margin=10)]
    g = Game(players)
    n = 0
    while g.winner is None and n < 120:
        cp = g.current_player
        m = players[cp].get_move(g)
        assert g.valid_move(m, cp)
        g.do_move(cp, m)
        n += 1
    assert g.winner is not None


def test_legal_move_in_large_random_positions() -> None:
    """Long, sprawling positions (far bigger than real games) must not upset the engine."""
    import random

    from models import D

    found = 0
    for seed in range(40):
        rng = random.Random(seed)
        g = Game([RandomPlayer(0), RandomPlayer(1)])
        for _ in range(50):
            moves = []
            for m in g.get_possible_moves(g.current_player):
                if m is PASS:
                    continue
                if m.t != "R":
                    end = g.points.get((m.c[0] + D[m.t].delta[0], m.c[1] + D[m.t].delta[1]))
                    if end is not None and end.connected:
                        continue  # keep it a tree so nobody scores
                moves.append(m)
            g.do_move(g.current_player, rng.choice(moves))
        assert g.winner is None
        player = RSPlayer(g.current_player, time_ms=0, max_nodes=20000, threads=2)
        m = player.get_move(g)
        assert g.valid_move(m, g.current_player)
        found += 1
        if found >= 6:
            break


def test_keeps_searching_for_a_win_it_already_announced() -> None:
    """If one move announces a forced win and the next search does not find it again, the
    player searches longer instead of settling for an ordinary move."""

    class FakeEngine:
        def __init__(self, real) -> None:
            self.real = real
            self.scores = [29990, 500, 400, 29992]  # win, lost sight of it twice, found again
            self.calls = 0

        def set_position(self, *a) -> None:
            self.real.set_position(*a)

        def search(self, time_ms, max_depth, max_nodes):
            info = dict(self.real.search(0, 2, 0))
            info["score"] = self.scores[min(self.calls, len(self.scores) - 1)]
            self.calls += 1
            return info

    g = Game([RandomPlayer(0), RandomPlayer(1)])
    player = RSPlayer(0, time_ms=10, book="")
    player.engine = FakeEngine(player.engine)
    g.do_move(0, player.get_move(g))
    assert player.engine.calls == 1 and player.last_info["score"] == 29990
    g.do_move(1, PASS)
    g.do_move(0, player.get_move(g))
    assert player.engine.calls == 4  # one normal search, then retries until the win reappeared
    assert player.last_info["score"] == 29992
