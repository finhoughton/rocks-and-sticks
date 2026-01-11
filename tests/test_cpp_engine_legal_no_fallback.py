import players_ext

import game
from gnn.encode import SAMPLE_ENC
from gnn.game_generation import randomize_start
from gnn.model import load_model
from models import move_key
from players.game_total import GameTotal
from players.move_utils import to_py_move


def _load_eval() -> None:
    node_dim = SAMPLE_ENC.data.x.size(1)  # type: ignore
    global_dim = SAMPLE_ENC.data.global_feats.size(1)
    load_model("checkpoints/gnn_eval_balanced.pt", node_dim, global_dim, device="cpu")


def test_cpp_engine_choose_move_is_python_legal_through_play():
    _load_eval()

    py_game = game.Game()
    cpp_state = players_ext.GameState()
    total = GameTotal(py_game, cpp_state)

    randomize_start(total, 50, 20)

    e0 = players_ext.MCTSEngine(123)
    e1 = players_ext.MCTSEngine(456)

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
