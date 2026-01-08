import random

import mcts_ext

from gnn.encode import SAMPLE_ENC
from gnn.model import load_model


def test_mcts_engine_deterministic():
    node_dim = SAMPLE_ENC.data.x.size(1)  # type: ignore
    global_dim = SAMPLE_ENC.data.global_feats.size(1)
    load_model("checkpoints/gnn_eval_balanced.pt", node_dim, global_dim, device="cpu")

    start = random.randint(0, 2**30 - 1)
    state1 = mcts_ext.GameState()
    state2 = mcts_ext.GameState()
    for i in range(10):
        e1 = mcts_ext.MCTSEngine(start + i)
        e2 = mcts_ext.MCTSEngine(start + i)
        m1 = e1.choose_move(state1, 100)
        m2 = e2.choose_move(state2, 100)

        assert (m1.x, m1.y, m1.t) == (m2.x, m2.y, m2.t)
