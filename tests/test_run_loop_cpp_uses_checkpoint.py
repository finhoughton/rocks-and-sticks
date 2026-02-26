import os
from pathlib import Path

import pytest

from gnn.encode import SAMPLE_ENC


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


def test_run_loop_cpp_eval_records_nn_usage(tmp_path):
    pytest.importorskip("players_ext")

    ckpt = _write_test_eval_checkpoint(tmp_path)
    # Also write a temporary policy checkpoint to enable policy priors in C++ engine.
    import torch

    from rl.train import PolicyValueNet
    pol = PolicyValueNet(node_feat_dim=int(SAMPLE_ENC.data.x.shape[1]), global_feat_dim=int(SAMPLE_ENC.data.global_feats.shape[1]), move_feat_dim=16)
    pol_ckpt = tmp_path / "policy_test.pt"
    torch.save(pol.state_dict(), pol_ckpt)

    from rl.run_loop import _evaluate_vs_random

    # Keep it tiny so it runs fast in unit tests.
    res = _evaluate_vs_random(
        backend="cpp",
        device="cpu",
        eval_games=1,
        eval_rollouts=10,
        eval_max_moves=4,
        eval_seed=12345,
        eval_randomize_start=False,
        iteration=0,
        model_path=str(ckpt),
        policy_path=str(pol_ckpt),
        cpp_verbose=False,
        cpp_use_nn_value=True,
    )

    cpp_prof = res.get("cpp_profile") or {}
    # Newer C++ profiling reports policy-prior activity under
    # `policy_prior_calls`; keep backward compatibility with
    # legacy `prior_model_calls`.
    prior_calls = float(cpp_prof.get("prior_model_calls", 0.0))
    policy_prior_calls = float(cpp_prof.get("policy_prior_calls", 0.0))
    assert (prior_calls + policy_prior_calls) > 0.0, cpp_prof

    assert float(cpp_prof.get("value_model_calls", 0.0)) > 0.0, cpp_prof


@pytest.mark.skipif(os.environ.get("CI") == "true", reason="Extra safety on CI")
def test_run_loop_cpp_eval_disables_exploration_noise(tmp_path):
    pytest.importorskip("players_ext")

    ckpt = _write_test_eval_checkpoint(tmp_path)
    # also write and pass a policy checkpoint
    import torch

    from rl.train import PolicyValueNet
    pol = PolicyValueNet(node_feat_dim=int(SAMPLE_ENC.data.x.shape[1]), global_feat_dim=int(SAMPLE_ENC.data.global_feats.shape[1]), move_feat_dim=16)
    pol_ckpt = tmp_path / "policy_test2.pt"
    torch.save(pol.state_dict(), pol_ckpt)

    from rl.run_loop import _evaluate_vs_random

    r1 = _evaluate_vs_random(
        backend="cpp",
        device="cpu",
        eval_games=2,
        eval_rollouts=10,
        eval_max_moves=16,
        eval_seed=999,
        eval_randomize_start=False,
        iteration=0,
        model_path=str(ckpt),
        policy_path=str(pol_ckpt),
        cpp_verbose=False,
        cpp_use_nn_value=False,
    )

    r2 = _evaluate_vs_random(
        backend="cpp",
        device="cpu",
        eval_games=2,
        eval_rollouts=10,
        eval_max_moves=16,
        eval_seed=999,
        eval_randomize_start=False,
        iteration=0,
        model_path=str(ckpt),
        policy_path=str(pol_ckpt),
        cpp_verbose=False,
        cpp_use_nn_value=False,
    )

    # With the same seed/settings and exploration disabled, results should match exactly.
    assert (r1["wins"], r1["losses"], r1["draws"]) == (r2["wins"], r2["losses"], r2["draws"])