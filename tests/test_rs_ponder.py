"""Pondering: the engine searches in the background while the opponent thinks."""

from __future__ import annotations

import time

import pytest

rs_engine = pytest.importorskip("rs_engine")

from game import Game  # noqa: E402
from players.base import RandomPlayer  # noqa: E402
from players.rs_book import to_move  # noqa: E402
from players.rs_player import RSPlayer  # noqa: E402


def test_stop_ends_a_background_search_quickly_and_is_never_lost() -> None:
    eng = rs_engine.Engine(16, 2)
    eng.clear_stop()
    eng.stop()  # sent before the search starts: it must still stop at once
    t0 = time.time()
    eng.search(60_000, 0, 0, True)
    assert time.time() - t0 < 1.0
    # a normal search clears the flag again
    assert eng.search(0, 4, 0)["depth"] == 4


def test_pondering_player_plays_legal_moves_and_stops_cleanly() -> None:
    bot = RSPlayer(1, time_ms=150, threads=2, tt_bits=18, ponder=True, book="")
    game = Game([RandomPlayer(0, seed=3), bot])
    while game.winner is None and len(game.moves) < 16:
        p = game.current_player
        if p == 1:
            m = bot.get_move(game)
        else:
            time.sleep(0.05)  # the "human" thinks while the bot ponders
            m = game.players[0].get_move(game)
        assert game.valid_move(m, p)
        game.do_move(p, m)
    bot.stop_pondering()
    assert bot._ponder_thread is None


def test_pondering_gives_the_next_search_a_head_start() -> None:
    depths = []
    for ponder in (False, True):
        bot = RSPlayer(1, time_ms=200, threads=1, tt_bits=20, ponder=ponder, book="")
        g = Game([RandomPlayer(0), bot])
        g.do_move(0, to_move((0, 0, "S")))
        g.do_move(1, bot.get_move(g))  # pondering starts here when enabled
        time.sleep(1.5)
        reply = to_move((0, 1, "N"))
        if not g.valid_move(reply, 0):
            reply = to_move((0, -1, "S"))
        g.do_move(0, reply)
        bot.get_move(g)
        depths.append(bot.last_info["depth"])
        bot.stop_pondering()
    assert depths[1] >= depths[0]
