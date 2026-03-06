"""Small utilities shared across the rl package."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
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
