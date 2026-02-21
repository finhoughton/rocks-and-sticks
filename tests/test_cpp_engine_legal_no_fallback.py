import tempfile

import players_ext
import torch

import game
from gnn.encode import SAMPLE_ENC
from gnn.game_generation import randomize_start
from gnn.model import init_random_model
from models import move_key
from players.game_total import GameTotal
from players.move_utils import to_py_move
from rl.train import PolicyValueNet


def _load_eval() -> None:
    # Tests shouldn't depend on a specific on-disk checkpoint format.
    # Use a deterministic randomly-initialized evaluator (GraphNorm-compatible).
    torch.manual_seed(0)
    node_dim = SAMPLE_ENC.data.x.size(1)  # type: ignore
    global_dim = SAMPLE_ENC.data.global_feats.size(1)
    init_random_model(node_dim, global_dim, device="cpu")


def test_cpp_engine_choose_move_is_python_legal_through_play():
    _load_eval()

    py_game = game.Game()
    cpp_state = players_ext.GameState()
    total = GameTotal(py_game, cpp_state)

    randomize_start(total, 50, 20)

    e0 = players_ext.MCTSEngine(123)
    e1 = players_ext.MCTSEngine(456)
    # For this unit test we don't need NN leaf values; disable to avoid loading a value model.
    e0.set_use_nn_value(False)
    e1.set_use_nn_value(False)

    # Create and load a temporary policy checkpoint so the C++ engine can compute priors.
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
        node_dim = SAMPLE_ENC.data.x.size(1)
        global_dim = SAMPLE_ENC.data.global_feats.size(1)
        pol = PolicyValueNet(node_feat_dim=node_dim, global_feat_dim=global_dim, move_feat_dim=16)
        torch.save(pol.state_dict(), tf.name)
        e0.set_policy_checkpoint(tf.name, "cpu")
        e1.set_policy_checkpoint(tf.name, "cpu")
        # Also write and load a GNNEval checkpoint for the engine's value model.
        from gnn.model import GNNEval
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as mf:
            model = GNNEval(node_feat_dim=node_dim, global_feat_dim=global_dim)  # type: ignore[arg-type]
            torch.save(model.state_dict(), mf.name)
            e0.set_model_checkpoint(mf.name, "cpu")
            e1.set_model_checkpoint(mf.name, "cpu")

    # Keep this test unit-speed: low rollouts, bounded moves.
    for _ in range(120):
        if py_game.winner is not None:
            break

        player_idx = py_game.current_player
        engine = e0 if player_idx == 0 else e1

        cpp_mv = engine.choose_move(total.cpp, 60)
        py_mv = to_py_move(cpp_mv)

        assert py_game.valid_move(py_mv, player_idx), f"C++ engine returned illegal move {py_mv}"

        total.do_move(player_idx, py_mv)

        # Keep both trees aligned with the advanced game state.
        e0.advance_root(total.cpp)
        e1.advance_root(total.cpp)


def test_cpp_possible_moves_match_python_after_randomize_start():
    _load_eval()

    for _ in range(50):
        py_game = game.Game()
        cpp_state = players_ext.GameState()
        total = GameTotal(py_game, cpp_state)

        randomize_start(total, 50, 20)

        player_idx = py_game.current_player

        py_moves = sorted(list(py_game.get_possible_moves(player_idx)), key=move_key)
        cpp_moves = [to_py_move(m) for m in total.cpp.get_possible_moves(player_idx)]
        cpp_moves = sorted(cpp_moves, key=move_key)

        assert py_moves == cpp_moves, "Python/C++ get_possible_moves disagree"
