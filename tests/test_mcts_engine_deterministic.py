import random
import tempfile

import players_ext
import torch

from gnn.encode import SAMPLE_ENC
from gnn.model import init_random_model
from rl.train import PolicyValueNet


def test_mcts_engine_deterministic():
    torch.manual_seed(0)
    node_dim = SAMPLE_ENC.data.x.size(1)  # type: ignore
    global_dim = SAMPLE_ENC.data.global_feats.size(1)
    init_random_model(node_dim, global_dim, device="cpu")

    start = random.randint(0, 2**30 - 1)
    state1 = players_ext.GameState()
    state2 = players_ext.GameState()
    for i in range(10):
        e1 = players_ext.MCTSEngine(start + i)
        e2 = players_ext.MCTSEngine(start + i)
        # Deterministic test doesn't need NN leaf values; disable NN value eval.
        e1.set_use_nn_value(False)
        e2.set_use_nn_value(False)
        # create and load temporary policy checkpoint
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
            node_dim = SAMPLE_ENC.data.x.size(1)
            global_dim = SAMPLE_ENC.data.global_feats.size(1)
            pol = PolicyValueNet(node_feat_dim=node_dim, global_feat_dim=global_dim, move_feat_dim=16)
            torch.save(pol.state_dict(), tf.name)
            e1.set_policy_checkpoint(tf.name, "cpu")
            e2.set_policy_checkpoint(tf.name, "cpu")
            from gnn.model import GNNEval
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as mf:
                model = GNNEval(node_feat_dim=node_dim, global_feat_dim=global_dim)  # type: ignore[arg-type]
                torch.save(model.state_dict(), mf.name)
                e1.set_model_checkpoint(mf.name, "cpu")
                e2.set_model_checkpoint(mf.name, "cpu")
        m1 = e1.choose_move(state1, 100)
        m2 = e2.choose_move(state2, 100)

        assert (m1.x, m1.y, m1.t) == (m2.x, m2.y, m2.t)
