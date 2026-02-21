import tempfile

import players_ext
import torch

import game
from gnn.encode import SAMPLE_ENC
from gnn.game_generation import randomize_start
from gnn.model import init_random_model
from players.game_total import GameTotal
from players.mcts_cpp import MCTSPlayerCPP
from rl.train import PolicyValueNet


def test_mctsplayercpp_move_is_legal_from_random_positions():
    torch.manual_seed(0)
    node_dim = SAMPLE_ENC.data.x.size(1)  # type: ignore
    global_dim = SAMPLE_ENC.data.global_feats.size(1)
    init_random_model(node_dim, global_dim, device="cpu")

    # Keep rollouts small so this stays unit-test fast.
    pcpp = MCTSPlayerCPP(player_number=0, n_rollouts=50)
    # This test doesn't require NN leaf value evaluation; disable to avoid loading a value model.
    pcpp.engine.set_use_nn_value(False)

    # Create and load a temporary policy checkpoint for the player's engine.
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
        pol_node_dim = SAMPLE_ENC.data.x.size(1)
        pol_global_dim = SAMPLE_ENC.data.global_feats.size(1)
        pol = PolicyValueNet(node_feat_dim=pol_node_dim, global_feat_dim=pol_global_dim, move_feat_dim=16)
        torch.save(pol.state_dict(), tf.name)
        pcpp.set_policy_checkpoint(tf.name, "cpu")
        from gnn.model import GNNEval
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as mf:
            model = GNNEval(node_feat_dim=pol_node_dim, global_feat_dim=pol_global_dim)  # type: ignore[arg-type]
            torch.save(model.state_dict(), mf.name)
            pcpp.set_model_checkpoint(mf.name, "cpu")

    for _ in range(250):
        py_game = game.Game()
        cpp_state = players_ext.GameState()
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
 