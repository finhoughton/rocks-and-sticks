from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from gnn.encode import SAMPLE_ENC
from rl.train import PolicyValueNet


def _infer_policy_dims_from_state(state: dict) -> tuple[int, int]:
    """Infer (hidden, move_feat_dim) from a PolicyValueNet state_dict."""
    w = state.get("policy_mlp.0.weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 2:
        # Fallback to historical defaults.
        return 256, 5
    hidden = int(w.size(0))
    in_dim = int(w.size(1))
    move_feat_dim = max(1, int(in_dim - hidden))
    return hidden, move_feat_dim


def _infer_num_layers_from_state(state: dict) -> int:
    idxs: set[int] = set()
    for k in state.keys():
        if not isinstance(k, str) or not k.startswith("convs."):
            continue
        parts = k.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            idxs.add(int(parts[1]))
    return max(1, len(idxs)) if idxs else 3


def load_policy_model(path: str, device: str = "cpu") -> PolicyValueNet:
    """Load a PolicyValueNet (policy+value) from a state_dict checkpoint.

    The checkpoint at `path` is expected to be produced by `rl.train` and contain
    a plain `state_dict`.
    """

    dev = torch.device(device)
    node_dim = int(SAMPLE_ENC.data.x.size(1))  # type: ignore[attr-defined]
    global_dim = int(SAMPLE_ENC.data.global_feats.size(1))  # type: ignore[attr-defined]

    state = torch.load(str(path), map_location=dev)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Policy checkpoint did not contain a state_dict: {type(state)}")

    hidden, move_feat_dim = _infer_policy_dims_from_state(state)
    num_layers = _infer_num_layers_from_state(state)

    model = PolicyValueNet(
        node_feat_dim=node_dim,
        global_feat_dim=global_dim,
        move_feat_dim=move_feat_dim,
        hidden=hidden,
        num_layers=num_layers,
    ).to(dev)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.no_grad()
def policy_priors_from_enc_and_moves(
    model: PolicyValueNet,
    enc_obj,
    move_feats: Sequence[Sequence[float]],
    device: str = "cpu",
) -> list[float]:
    """Return a softmax distribution over `move_feats` for a single state.

    `enc_obj` is the object returned by `players_ext_internal.encode_state_common`,
    which is a SimpleNamespace containing `.data`.

    `move_feats` is a list of feature rows matching the policy checkpoint's
    move feature dimension.
    """

    if not move_feats:
        return []

    dev = torch.device(device)
    data = enc_obj.data
    data = data.to(dev)

    pooled, _g_flat = model.encode_graph(data)

    mf = torch.as_tensor(move_feats, dtype=torch.float32, device=dev)
    expected_dim: int | None = None
    try:
        lin0 = model.policy_mlp[0]
        if isinstance(lin0, nn.Linear):
            expected_dim = int(lin0.in_features - lin0.out_features)
    except Exception:
        expected_dim = None

    if mf.dim() != 2:
        if expected_dim is None:
            expected_dim = int(mf.numel())
        mf = mf.view(-1, int(expected_dim))

    if expected_dim is not None and int(mf.size(1)) != int(expected_dim):
        raise ValueError(
            f"move_feat dim mismatch: got {int(mf.size(1))} expected {int(expected_dim)}"
        )
    move_owner = torch.zeros((mf.size(0),), dtype=torch.long, device=dev)

    logits = model.policy_logits_grouped(pooled, mf, move_owner)
    probs = torch.softmax(logits, dim=0)

    return probs.detach().cpu().tolist()
