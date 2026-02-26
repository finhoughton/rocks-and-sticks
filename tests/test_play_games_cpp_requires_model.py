from pathlib import Path

import pytest


def _write_test_eval_checkpoint(tmp_path: Path) -> Path:
    import torch

    from gnn.encode import SAMPLE_ENC
    from gnn.model import GNNEval

    torch.manual_seed(0)
    node_feat_dim = int(SAMPLE_ENC.data.x.shape[1])  # type: ignore
    global_feat_dim = int(SAMPLE_ENC.data.global_feats.shape[1])

    model = GNNEval(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim)
    model.eval()

    out = tmp_path / "gnn_eval_test.pt"
    torch.save(model.state_dict(), out)
    return out


def test_play_games_cpp_requires_model_allows_when_provided(tmp_path):
    pytest.importorskip("players_ext")

    ckpt = _write_test_eval_checkpoint(tmp_path)

    from rl.play_games import play_self_play_games

    play_self_play_games(
        num_games=1,
        mcts_rollouts=2,
        mcts_time_limit=None,
        save_games_dir=str(tmp_path),
        model_path=str(ckpt),
        device="cpu",
        backend="cpp",
        cpp_verbose=False,
        cpp_use_nn_value=False,
        max_moves=2,
        seed_base=123,
    )


def test_play_games_cpp_raises_without_model(tmp_path):
    pytest.importorskip("players_ext")

    from rl.play_games import play_self_play_games

    with pytest.raises(ValueError):
        play_self_play_games(
            num_games=1,
            mcts_rollouts=2,
            mcts_time_limit=None,
            save_games_dir=str(tmp_path),
            model_path=None,
            device="cpu",
            backend="cpp",
            cpp_verbose=False,
            cpp_use_nn_value=False,
            max_moves=2,
            seed_base=123,
        )
