import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data


def make_sample(num_nodes=6, node_feat_dim=4, global_feat_dim=2, K=4, move_feat_dim=5, value=0.0, peak=0):
    x = torch.randn((num_nodes, node_feat_dim), dtype=torch.float32)
    # simple chain edges
    edges = []
    for i in range(num_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    E = edge_index.size(1)
    # edge features: two-dim
    edge_attr = torch.zeros((E, 2), dtype=torch.float32)
    # random global feats
    global_feats = torch.randn((1, global_feat_dim), dtype=torch.float32)
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, global_feats=global_feats)

    # moves
    move_feat = torch.randn((K, move_feat_dim), dtype=torch.float32)
    policy = torch.zeros((K,), dtype=torch.float32)
    policy[peak % K] = 1.0

    return {
        "graph": graph,
        "move_feat": move_feat,
        "policy": policy,
        "value": float(value),
    }

def main():
    out = Path('tests/fixtures')
    out.mkdir(parents=True, exist_ok=True)
    samples = []
    # create 8 samples, alternating values and peaks
    for i in range(8):
        v = 1.0 if i < 4 else 0.0
        peak = i % 4
        s = make_sample(value=v, peak=peak)
        samples.append(s)
    torch.save(samples, out / 'tiny_dataset.pt')
    print('Wrote', out / 'tiny_dataset.pt')

if __name__ == '__main__':
    main()
