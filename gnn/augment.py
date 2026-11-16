from typing import Callable

import torch
from torch_geometric.data import Data

from gnn.encode import EncodedGraph


def _augment_symmetries(enc: EncodedGraph) -> list[Data]:
    data = enc.data
    x = data.x
    assert x is not None, "data.x must not be None"
    # node features end with [x, y, r2]; r2 is unchanged by every symmetry
    feats = x[:, :-3]
    coords = x[:, -3:-1]
    r2 = x[:, -1:]
    transforms: list[Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = [
        lambda a, b: (a, b),
        lambda a, b: (-a, b),
        lambda a, b: (a, -b),
        lambda a, b: (-a, -b),
        lambda a, b: (b, a),
        lambda a, b: (-b, a),
        lambda a, b: (b, -a),
        lambda a, b: (-b, -a),
    ]
    out: list[Data] = []
    for tf in transforms:
        a, b = coords[:, 0], coords[:, 1]
        tx, ty = tf(a, b)
        coords_tf = torch.stack((tx, ty), dim=1)
        x_tf = torch.cat((feats, coords_tf, r2), dim=1)
        out.append(
            Data(
                x=x_tf,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=data.batch,
                global_feats=data.global_feats,
            )
        )
    return out

def samples_to_data(samples: list[tuple[EncodedGraph, float, float]], augment_sym: bool) -> list[Data]:
    out: list[Data] = []
    for enc, label, weight in samples:
        datas = _augment_symmetries(enc) if augment_sym else [enc.data]
        for data in datas:
            data.y = torch.tensor([label], dtype=torch.float32)
            data.weight = torch.tensor([weight], dtype=torch.float32)
            out.append(data)
    return out


def policy_samples_to_data(
    samples: list,  # list[PolicySample]
    augment_sym: bool,
) -> list[Data]:
    """Convert PolicySamples to Data objects.

    Each Data object gets:
        - y: value label (float)
        - weight: sample weight (float)
        - move_feats: [K, 16] move features or empty (for policy head)
        - policy_target: [K] policy distribution or empty
        - has_policy: 1.0 if policy targets present, 0.0 otherwise
    """
    out: list[Data] = []
    for sample in samples:
        enc, label, weight, move_feats, policy_probs = sample
        # For simplicity, skip augmentation of policy samples
        # (augmenting move features would require coordinate transforms)
        datas = [enc.data] if not augment_sym else _augment_symmetries(enc)
        for data in datas:
            data.y = torch.tensor([label], dtype=torch.float32)
            data.weight = torch.tensor([weight], dtype=torch.float32)
            if move_feats is not None and policy_probs is not None:
                data.move_feats = move_feats           # [K, 16]
                data.policy_target = policy_probs      # [K]
                data.has_policy = torch.tensor([1.0], dtype=torch.float32)
            else:
                data.move_feats = torch.zeros(0, 16, dtype=torch.float32)
                data.policy_target = torch.zeros(0, dtype=torch.float32)
                data.has_policy = torch.tensor([0.0], dtype=torch.float32)
            out.append(data)
    return out
