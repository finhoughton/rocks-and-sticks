"""Small utilities shared across the rl package."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Scheduling helper
# ---------------------------------------------------------------------------

def scheduled_int(
    *,
    cur_iter: int,
    start_iter: int,
    end_iter: int,
    start_val: int,
    end_val: int,
    curve: str,
) -> int:
    """Compute a scheduled integer value (linear or cosine) for the current iteration."""
    cur_iter = int(cur_iter)
    start_iter = int(start_iter)
    end_iter = int(end_iter)
    start_val = int(start_val)
    end_val = int(end_val)
    curve = str(curve or "linear").lower()

    if end_iter <= start_iter:
        return int(end_val if cur_iter >= end_iter else start_val)

    t = (float(cur_iter) - float(start_iter)) / float(end_iter - start_iter)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)

    if curve in ("cos", "cosine"):
        t = 0.5 - 0.5 * math.cos(math.pi * t)
    elif curve in ("linear", "lin"):
        pass
    else:
        raise ValueError(f"Unknown schedule curve: {curve}")

    v = float(start_val) + (float(end_val) - float(start_val)) * t
    return int(round(v))


# ---------------------------------------------------------------------------
# Checkpoint export
# ---------------------------------------------------------------------------

def export_gnn_eval_from_policy(policy_ckpt: Path, out_path: Path) -> None:
    """Load policy+value checkpoint and copy conv/norm weights into GNNEval then save."""
    from gnn.encode import SAMPLE_ENC
    from gnn.model import GNNEval

    device = torch.device("cpu")
    ck = torch.load(policy_ckpt, map_location=device)
    state = ck if isinstance(ck, dict) and "state_dict" not in ck else ck.get("state_dict", ck)

    node_feat_dim = int(SAMPLE_ENC.data.x.shape[1])  # type: ignore
    global_feat_dim = int(SAMPLE_ENC.data.global_feats.shape[1])
    model = GNNEval(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim)
    model_sd = model.state_dict()

    new_sd: dict[str, torch.Tensor] = {}
    for k in model_sd.keys():
        if k in state:
            new_sd[k] = state[k]

    # Map PolicyValueNet value head → GNNEval head.
    # PolicyValueNet: value_mlp = [Linear, ReLU, Linear]
    # GNNEval:        head      = [Linear, ReLU, Dropout, Linear]
    if isinstance(state, dict) and any(str(k).startswith("value_mlp.") for k in state.keys()):
        mapping = {
            "value_mlp.0.weight": "head.0.weight",
            "value_mlp.0.bias": "head.0.bias",
            "value_mlp.2.weight": "head.3.weight",
            "value_mlp.2.bias": "head.3.bias",
        }
        for src, dst in mapping.items():
            if src in state and dst in model_sd:
                new_sd[dst] = state[src]

    model_sd.update(new_sd)
    model.load_state_dict(model_sd)
    torch.save(model.state_dict(), out_path)
    print(f"Saved GNNEval checkpoint to {out_path}")


# ---------------------------------------------------------------------------
# Native binary export (for NativeNNEval in C++)
# ---------------------------------------------------------------------------

def _write_tensor(f, t: "torch.Tensor") -> None:
    """Write a tensor as contiguous row-major float32."""
    data = t.detach().cpu().contiguous().float().numpy()
    f.write(data.tobytes())


def export_native_weights(pt_path: str | Path, bin_path: str | Path) -> None:
    """Convert a GNNEval .pt state-dict to the C++ NativeNNEval binary format.

    The binary format has magic ``NNWT`` and is read by
    ``NativeNNEval::load_weights()`` in ``nn_eval.cpp``.
    """
    import torch as _torch

    sd = _torch.load(str(pt_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # Detect architecture from weight shapes.
    lin0_w = sd["convs.0.lin.weight"]
    node_feat_dim = int(lin0_w.shape[0])
    edge_dim = int(lin0_w.shape[1])
    hidden = int(sd["convs.0.nn.0.weight"].shape[0])

    num_layers = 0
    while f"convs.{num_layers}.eps" in sd:
        num_layers += 1

    head0_w = sd["head.0.weight"]
    global_feat_dim = int(head0_w.shape[1]) - hidden

    print(
        f"Exporting native weights: nfd={node_feat_dim} gfd={global_feat_dim} "
        f"hidden={hidden} layers={num_layers} edge_dim={edge_dim}"
    )

    with open(str(bin_path), "wb") as f:
        # Magic
        f.write(b"NNWT")
        # Header
        for v in (node_feat_dim, global_feat_dim, hidden, num_layers, edge_dim):
            f.write(struct.pack("<i", v))

        # Conv layers
        for i in range(num_layers):
            lin_w = sd[f"convs.{i}.lin.weight"]
            lin_b = sd[f"convs.{i}.lin.bias"]
            nn0_w = sd[f"convs.{i}.nn.0.weight"]
            nn0_b = sd[f"convs.{i}.nn.0.bias"]
            nn2_w = sd[f"convs.{i}.nn.2.weight"]
            nn2_b = sd[f"convs.{i}.nn.2.bias"]
            eps = sd[f"convs.{i}.eps"]

            in_ch = int(lin_w.shape[0])
            out_ch = int(nn0_w.shape[0])
            ed = int(lin_w.shape[1])

            f.write(struct.pack("<i", in_ch))
            f.write(struct.pack("<i", out_ch))
            f.write(struct.pack("<i", ed))
            f.write(struct.pack("<f", float(eps.item())))

            _write_tensor(f, lin_w)
            _write_tensor(f, lin_b)
            _write_tensor(f, nn0_w)
            _write_tensor(f, nn0_b)
            _write_tensor(f, nn2_w)
            _write_tensor(f, nn2_b)

        # Norm layers
        for i in range(num_layers):
            mean_scale = sd[f"norms.{i}.mean_scale"]
            weight = sd[f"norms.{i}.weight"]
            bias = sd[f"norms.{i}.bias"]
            ch = int(mean_scale.shape[0])

            f.write(struct.pack("<i", ch))
            _write_tensor(f, mean_scale)
            _write_tensor(f, weight)
            _write_tensor(f, bias)

        # Head
        _write_tensor(f, sd["head.0.weight"])
        _write_tensor(f, sd["head.0.bias"])
        _write_tensor(f, sd["head.3.weight"])
        f.write(struct.pack("<f", float(sd["head.3.bias"].item())))

        # Policy head (optional — write if weights exist in state dict)
        if "policy_mlp.0.weight" in sd:
            pol0_w = sd["policy_mlp.0.weight"]
            pol0_b = sd["policy_mlp.0.bias"]
            pol2_w = sd["policy_mlp.2.weight"]
            pol2_b = sd["policy_mlp.2.bias"]
            policy_in = int(pol0_w.shape[1])
            move_feat_dim = policy_in - hidden - global_feat_dim
            f.write(b"PLCY")
            f.write(struct.pack("<i", move_feat_dim))
            _write_tensor(f, pol0_w)    # [hidden, policy_in]
            _write_tensor(f, pol0_b)    # [hidden]
            _write_tensor(f, pol2_w)    # [1, hidden] — stored as [hidden]
            f.write(struct.pack("<f", float(pol2_b.item())))
            print(f"  Policy head exported: move_feat_dim={move_feat_dim}")

    print(f"Exported native binary to {bin_path}")


def ensure_native_bin(pt_path: Path) -> Path:
    """Return the ``.bin`` sibling of *pt_path*, exporting it if necessary.

    If the ``.bin`` file already exists and is at least as new as the
    ``.pt`` file it is returned immediately.
    """
    bin_path = pt_path.with_suffix(".bin")
    if bin_path.exists() and bin_path.stat().st_mtime >= pt_path.stat().st_mtime:
        return bin_path
    export_native_weights(pt_path, bin_path)
    return bin_path


# ---------------------------------------------------------------------------
# Dataset cleanup
# ---------------------------------------------------------------------------

def cleanup_old_datasets(data_dir: Path, *, keep_last: int, current_iter: int) -> None:
    """Delete old iteration datasets (pt + shards) to save disk space."""
    keep_last = max(0, int(keep_last))
    cutoff = current_iter - keep_last
    if cutoff <= 0:
        return

    for k in range(1, cutoff + 1):
        pt_path = data_dir / f"alpha_dataset_iter_{k}.pt"
        shards_dir = data_dir / f"alpha_dataset_iter_{k}.pt.shards"
        try:
            if shards_dir.exists() and shards_dir.is_dir():
                shutil.rmtree(shards_dir)
        except Exception as e:
            print(f"Warning: failed to delete {shards_dir}: {e}")
        try:
            if pt_path.exists() and pt_path.is_file():
                pt_path.unlink()
        except Exception as e:
            print(f"Warning: failed to delete {pt_path}: {e}")


# ---------------------------------------------------------------------------
# JSONL append
# ---------------------------------------------------------------------------

def append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def merge_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    """Merge two count dicts by summing values."""
    out = dict(a)
    for k, v in b.items():
        out[k] = int(out.get(k, 0)) + int(v)
    return out


def split_indices(n: int, parts: int) -> list[list[int]]:
    """Split ``range(n)`` into *parts* interleaved chunks."""
    n = int(n)
    parts = max(1, int(parts))
    idxs = list(range(n))
    if parts <= 1 or n <= 1:
        return [idxs]
    out: list[list[int]] = [[] for _ in range(parts)]
    for j, i in enumerate(idxs):
        out[j % parts].append(i)
    return [c for c in out if c]


def sha256_file(path: Path) -> str | None:
    """Return hex SHA-256 of a file, or None on error."""
    try:
        if path.exists() and path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
    except Exception:
        pass
    return None
