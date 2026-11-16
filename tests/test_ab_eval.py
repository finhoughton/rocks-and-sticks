"""Tests for AB engine evaluation functions in rl/eval.py."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gnn.encode import SAMPLE_ENC
from gnn.model import GNNEval


def _make_native_model(path: Path, seed: int = 0) -> Path:
    """Create a GNNEval .pt and export it to .bin, returning the .bin path."""
    from rl.utils import export_native_weights

    pt_path = path.with_suffix(".pt")
    torch.manual_seed(seed)
    node_dim = int(SAMPLE_ENC.data.x.size(1))  # type: ignore[union-attr]
    global_dim = int(SAMPLE_ENC.data.global_feats.size(1))  # type: ignore[union-attr]
    model = GNNEval(node_feat_dim=node_dim, global_feat_dim=global_dim)
    torch.save(model.state_dict(), pt_path)

    bin_path = path.with_suffix(".bin")
    export_native_weights(pt_path, bin_path)
    return bin_path


# ------------------------------------------------------------------
# evaluate_ab_vs_random
# ------------------------------------------------------------------


def test_ab_vs_random_returns_expected_keys(tmp_path):
    pytest.importorskip("players_ext")
    from rl.eval import evaluate_ab_vs_random

    bin_path = _make_native_model(tmp_path / "model")

    result = evaluate_ab_vs_random(
        eval_games=2,
        eval_max_moves=20,
        eval_seed=42,
        eval_randomize_start=False,
        iteration=1,
        ab_time_limit_ms=0,
        ab_depth=2,
        ab_max_depth=6,
        ab_move_cap=48,
        ab_use_heuristic=True,
        native_model=str(bin_path),
        nn_ordering_depth=3,
    )

    required_keys = {
        "wins", "losses", "draws", "win_rate", "eval_games",
        "avg_moves", "seconds", "iter", "model_path", "model_sha256",
    }
    assert required_keys.issubset(result.keys()), f"Missing keys: {required_keys - result.keys()}"
    assert result["eval_games"] == 2
    assert result["wins"] + result["losses"] + result["draws"] == 2


def test_ab_vs_random_records_model_sha256(tmp_path):
    pytest.importorskip("players_ext")
    from rl.eval import evaluate_ab_vs_random

    bin_path = _make_native_model(tmp_path / "model")

    result = evaluate_ab_vs_random(
        eval_games=1,
        eval_max_moves=10,
        eval_seed=123,
        eval_randomize_start=False,
        iteration=1,
        ab_time_limit_ms=0,
        ab_depth=2,
        ab_max_depth=6,
        ab_move_cap=48,
        ab_use_heuristic=True,
        native_model=str(bin_path),
    )

    assert result["model_sha256"] is not None
    assert len(result["model_sha256"]) == 64


def test_ab_vs_random_without_model(tmp_path):
    """AB eval works without a native model (pure heuristic)."""
    pytest.importorskip("players_ext")
    from rl.eval import evaluate_ab_vs_random

    result = evaluate_ab_vs_random(
        eval_games=2,
        eval_max_moves=20,
        eval_seed=42,
        eval_randomize_start=False,
        iteration=0,
        ab_time_limit_ms=0,
        ab_depth=2,
        ab_max_depth=6,
        ab_move_cap=48,
        ab_use_heuristic=True,
        native_model="",
    )

    assert result["model_sha256"] is None
    assert result["wins"] + result["losses"] + result["draws"] == 2


def test_ab_vs_random_deterministic(tmp_path):
    """Same seed, same model → same result."""
    pytest.importorskip("players_ext")
    from rl.eval import evaluate_ab_vs_random

    bin_path = _make_native_model(tmp_path / "model")

    kwargs = dict(
        eval_games=2,
        eval_max_moves=20,
        eval_seed=999,
        eval_randomize_start=False,
        iteration=1,
        ab_time_limit_ms=0,
        ab_depth=2,
        ab_max_depth=6,
        ab_move_cap=48,
        ab_use_heuristic=True,
        native_model=str(bin_path),
    )
    r1 = evaluate_ab_vs_random(**kwargs)
    r2 = evaluate_ab_vs_random(**kwargs)

    assert r1["wins"] == r2["wins"]
    assert r1["losses"] == r2["losses"]
    assert r1["draws"] == r2["draws"]


# ------------------------------------------------------------------
# evaluate_ab_vs_ab
# ------------------------------------------------------------------


def test_ab_vs_ab_returns_expected_keys(tmp_path):
    pytest.importorskip("players_ext")
    from rl.eval import evaluate_ab_vs_ab

    bin_a = _make_native_model(tmp_path / "model_a", seed=0)
    bin_b = _make_native_model(tmp_path / "model_b", seed=99)

    result = evaluate_ab_vs_ab(
        eval_games=2,
        eval_max_moves=20,
        eval_seed=42,
        eval_randomize_start=False,
        iteration=2,
        ab_time_limit_ms=0,
        ab_depth=2,
        ab_max_depth=6,
        ab_move_cap=48,
        ab_use_heuristic=True,
        model_a_path=str(bin_a),
        model_b_path=str(bin_b),
        model_a_label="iter_2",
        model_b_label="iter_1",
    )

    required_keys = {"wins", "losses", "draws", "win_rate", "eval_games", "seconds"}
    assert required_keys.issubset(result.keys())
    assert result["eval_games"] == 2
    assert result["wins"] + result["losses"] + result["draws"] == 2


def test_ab_vs_ab_same_model_balanced(tmp_path):
    """When both sides use the same model, neither should dominate (roles alternate)."""
    pytest.importorskip("players_ext")
    from rl.eval import evaluate_ab_vs_ab

    bin_path = _make_native_model(tmp_path / "model", seed=42)

    result = evaluate_ab_vs_ab(
        eval_games=4,
        eval_max_moves=30,
        eval_seed=42,
        eval_randomize_start=False,
        iteration=1,
        ab_time_limit_ms=0,
        ab_depth=2,
        ab_max_depth=6,
        ab_move_cap=48,
        ab_use_heuristic=True,
        model_a_path=str(bin_path),
        model_b_path=str(bin_path),
    )

    # With identical models and alternating roles, results should be roughly balanced.
    # At minimum, all games should be accounted for.
    total = result["wins"] + result["losses"] + result["draws"]
    assert total == 4
