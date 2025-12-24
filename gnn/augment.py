from typing import Callable

import torch
from torch_geometric.data import Data

from gnn.encode import EncodedGraph


def _augment_symmetries(enc: EncodedGraph) -> list[Data]:
    data = enc.data
    x = data.x
    assert x is not None, "data.x must not be None"
    feats = x[:, :-2]
    coords = x[:, -2:]
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
        x_tf = torch.cat((feats, coords_tf), dim=1)
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
