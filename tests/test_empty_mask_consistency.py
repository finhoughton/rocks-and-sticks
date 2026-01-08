import random

import mcts_ext
import pytest

import game
from gnn.encode import SAMPLE_ENC
from gnn.game_generation import randomize_start
from gnn.model import load_model
from players.game_total import GameTotal
from players.mcts_cpp import MCTSPlayerCPP


def _assert_empty_mask_consistent(cpp_state) -> None:
    drift = cpp_state.debug_empty_mask_drift_count()
    assert drift == 0, f"empty_mask drifted from neighbours: {drift} mismatches"


@pytest.mark.parametrize("steps", [200])
def test_empty_mask_does_not_drift(cpp_state, steps: int):
    rng = random.Random(0)
    applied = 0

    _assert_empty_mask_consistent(cpp_state)

    for i in range(steps):
        if applied > 0 and rng.random() < 0.30:
            cpp_state.undo_move()
            applied -= 1
        else:
            player = int(cpp_state.current_player)
            moves = cpp_state.get_possible_moves(player)
            assert moves, "C++ engine returned no moves"
            mv = rng.choice(moves)
            cpp_state.do_move(mv, player)
            applied += 1

        if i % 20 == 0:
            _assert_empty_mask_consistent(cpp_state)

    _assert_empty_mask_consistent(cpp_state)


def test_empty_mask_does_not_drift_after_mctsplayercpp_get_move():
    node_dim = SAMPLE_ENC.data.x.size(1)  # type: ignore
    global_dim = SAMPLE_ENC.data.global_feats.size(1)
    load_model("checkpoints/gnn_eval_balanced.pt", node_dim, global_dim, device="cpu")

    pcpp = MCTSPlayerCPP(player_number=0, n_rollouts=10)

    for _ in range(50):
        py_game = game.Game()
        cpp_state = mcts_ext.GameState()
        total = GameTotal(py_game, cpp_state)
        randomize_start(total, 3, 1)

        _assert_empty_mask_consistent(cpp_state)
        mv = pcpp.get_move(total)
        _assert_empty_mask_consistent(cpp_state)

        total.do_move(py_game.current_player, mv)
        _assert_empty_mask_consistent(cpp_state)
        total.undo_move()
        _assert_empty_mask_consistent(cpp_state)
