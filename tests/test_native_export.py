"""Tests for native binary export (GNNEval .pt → NativeNNEval .bin)."""
from __future__ import annotations

import struct
from pathlib import Path

import pytest
import torch

from gnn.encode import SAMPLE_ENC
from gnn.model import GNNEval


def _make_gnn_eval_checkpoint(path: Path, seed: int = 0) -> Path:
    """Create a deterministic GNNEval checkpoint at *path*."""
    torch.manual_seed(seed)
    node_dim = int(SAMPLE_ENC.data.x.size(1))  # type: ignore[union-attr]
    global_dim = int(SAMPLE_ENC.data.global_feats.size(1))  # type: ignore[union-attr]
    model = GNNEval(node_feat_dim=node_dim, global_feat_dim=global_dim)
    torch.save(model.state_dict(), path)
    return path


# ------------------------------------------------------------------
# export_native_weights
# ------------------------------------------------------------------


def test_export_native_weights_creates_file(tmp_path):
    from rl.utils import export_native_weights

    pt = _make_gnn_eval_checkpoint(tmp_path / "model.pt")
    bin_path = tmp_path / "model.bin"
    export_native_weights(pt, bin_path)

    assert bin_path.exists()
    assert bin_path.stat().st_size > 0


def test_export_native_weights_magic_header(tmp_path):
    from rl.utils import export_native_weights

    pt = _make_gnn_eval_checkpoint(tmp_path / "model.pt")
    bin_path = tmp_path / "model.bin"
    export_native_weights(pt, bin_path)

    with open(bin_path, "rb") as f:
        magic = f.read(4)
    assert magic == b"NNWT", f"Expected NNWT magic, got {magic!r}"


def test_export_native_weights_header_values(tmp_path):
    from rl.utils import export_native_weights

    pt = _make_gnn_eval_checkpoint(tmp_path / "model.pt")
    bin_path = tmp_path / "model.bin"
    export_native_weights(pt, bin_path)

    with open(bin_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"NNWT"
        nfd, gfd, hidden, nlayers, edge_dim = struct.unpack("<5i", f.read(20))

    node_dim = int(SAMPLE_ENC.data.x.size(1))  # type: ignore[union-attr]
    global_dim = int(SAMPLE_ENC.data.global_feats.size(1))  # type: ignore[union-attr]

    assert nfd == node_dim
    assert gfd == global_dim
    assert hidden == 384  # GNNEval default
    assert nlayers == 5  # GNNEval default
    assert edge_dim == 2  # GNNEval default


def test_export_native_weights_cpp_loads(tmp_path):
    """The C++ AlphaBetaEngine can load the exported binary."""
    pytest.importorskip("players_ext")
    import players_ext

    from rl.utils import export_native_weights

    pt = _make_gnn_eval_checkpoint(tmp_path / "model.pt")
    bin_path = tmp_path / "model.bin"
    export_native_weights(pt, bin_path)

    engine = players_ext.AlphaBetaEngine(42, 3.0)
    assert engine.load_native_model(str(bin_path)) is True


def test_export_two_different_seeds_differ(tmp_path):
    """Checkpoints from different seeds produce different binaries."""
    from rl.utils import export_native_weights

    pt_a = _make_gnn_eval_checkpoint(tmp_path / "a.pt", seed=0)
    pt_b = _make_gnn_eval_checkpoint(tmp_path / "b.pt", seed=99)
    bin_a = tmp_path / "a.bin"
    bin_b = tmp_path / "b.bin"
    export_native_weights(pt_a, bin_a)
    export_native_weights(pt_b, bin_b)

    assert bin_a.read_bytes() != bin_b.read_bytes()


def test_export_same_seed_deterministic(tmp_path):
    """Same checkpoint exported twice produces identical binaries."""
    from rl.utils import export_native_weights

    pt = _make_gnn_eval_checkpoint(tmp_path / "model.pt", seed=42)
    bin_a = tmp_path / "a.bin"
    bin_b = tmp_path / "b.bin"
    export_native_weights(pt, bin_a)
    export_native_weights(pt, bin_b)

    assert bin_a.read_bytes() == bin_b.read_bytes()


# ------------------------------------------------------------------
# ensure_native_bin
# ------------------------------------------------------------------


def test_ensure_native_bin_creates_bin(tmp_path):
    from rl.utils import ensure_native_bin

    pt = _make_gnn_eval_checkpoint(tmp_path / "model.pt")
    result = ensure_native_bin(pt)

    assert result == tmp_path / "model.bin"
    assert result.exists()


def test_ensure_native_bin_caches(tmp_path):
    """Second call doesn't re-export when the .bin is already up to date."""
    from rl.utils import ensure_native_bin

    pt = _make_gnn_eval_checkpoint(tmp_path / "model.pt")
    bin1 = ensure_native_bin(pt)
    mtime1 = bin1.stat().st_mtime

    # Touch with different content would change mtime, but ensure_native_bin
    # should skip because bin is already >= pt mtime.
    import time
    time.sleep(0.05)

    bin2 = ensure_native_bin(pt)
    mtime2 = bin2.stat().st_mtime

    assert bin1 == bin2
    assert mtime1 == mtime2


def test_ensure_native_bin_re_exports_when_stale(tmp_path):
    """Re-exports if the .pt is newer than the .bin."""
    import os
    import time

    from rl.utils import ensure_native_bin

    pt = _make_gnn_eval_checkpoint(tmp_path / "model.pt", seed=0)
    bin_path = ensure_native_bin(pt)
    old_bytes = bin_path.read_bytes()

    # Re-create the .pt with a different seed so the content changes.
    time.sleep(0.05)
    _make_gnn_eval_checkpoint(tmp_path / "model.pt", seed=99)

    bin_path2 = ensure_native_bin(pt)
    new_bytes = bin_path2.read_bytes()

    assert bin_path == bin_path2
    assert old_bytes != new_bytes, "Should have re-exported with different weights"
