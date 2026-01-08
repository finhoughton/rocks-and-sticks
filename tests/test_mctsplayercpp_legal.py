import mcts_ext

import game
from gnn.encode import SAMPLE_ENC
from gnn.game_generation import randomize_start
from gnn.model import load_model
from players.game_total import GameTotal
from players.mcts_cpp import MCTSPlayerCPP


def test_mctsplayercpp_move_is_legal_from_random_positions():
    node_dim = SAMPLE_ENC.data.x.size(1)  # type: ignore
    global_dim = SAMPLE_ENC.data.global_feats.size(1)
    load_model("checkpoints/gnn_eval_balanced.pt", node_dim, global_dim, device="cpu")

    # Keep rollouts small so this stays unit-test fast.
    pcpp = MCTSPlayerCPP(player_number=0, n_rollouts=50)

    for _ in range(250):
        py_game = game.Game()
        cpp_state = mcts_ext.GameState()
        total = GameTotal(py_game, cpp_state)
        randomize_start(total, 50, 20)

        player_num = py_game.current_player
        mv = pcpp.get_move(total)

        assert py_game.valid_move(mv, player_num), f"MCTSPlayerCPP returned illegal move {mv}"

        # Also ensure the move is among generated legal moves.
        legal = set(py_game.get_possible_moves(player_num))
        assert mv in legal, f"MCTSPlayerCPP move {mv} not in get_possible_moves()"

        # Sanity: applying the move should succeed for both backends.
        total.do_move(player_num, mv)
        total.undo_move()
 