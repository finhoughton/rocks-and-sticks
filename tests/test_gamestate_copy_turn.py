"""Copying the C++ GameState must keep whose turn it is.

randomize_start() places setup moves out of turn and then forces player 0 to move. The copy
constructor used to rebuild the state by replaying moves only, so after an odd number of setup
moves every copy (and the MCTS engine copies the state for each simulation) had the wrong
player to move and the search statistics landed under a different position key.
"""

from __future__ import annotations

import random

import players_ext

from game import Game
from gnn.game_generation import randomize_start
from players.base import RandomPlayer
from players.game_total import GameTotal
from players.mcts_cpp import MCTSPlayerCPP


def test_mcts_root_gets_visits_after_any_randomized_start() -> None:
    seen_odd = False
    for seed in range(12):
        random.seed(seed)
        game = GameTotal(Game(players=[RandomPlayer(0), RandomPlayer(1)]), players_ext.GameState())
        log: list = []
        randomize_start(game, max_sticks=5, move_log=log)
        seen_odd = seen_odd or len(log) % 2 == 1
        mcts = MCTSPlayerCPP(0, n_rollouts=60, seed=1, use_nn_value=False, use_heuristic_rollout=True)
        mcts.engine.choose_move(game.cpp, 60)
        visits = sum(s["visits"] for s in mcts.engine.get_root_visit_stats(game.cpp))
        assert visits == 60, f"seed {seed}: {len(log)} setup moves, root visits {visits}"
    assert seen_odd
