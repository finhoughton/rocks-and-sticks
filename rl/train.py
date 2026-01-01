"""Train a policy+value GNN on AlphaZero-style (state, move, policy_prob, value) samples.

Expects a dataset saved by `alpha_zero_convert.py` (a list of torch_geometric.Data).
Each Data must include: x, edge_index, edge_attr, batch, global_feats, node_coords,
and added fields: move_feat (tensor), y (policy prob), value (scalar).
"""

from __future__ import annotations

import argparse
import random
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

from gnn.model import GNNEval


class PolicyValueNet(nn.Module):
    def __init__(self, node_feat_dim: int, global_feat_dim: int, move_feat_dim: int = 5, hidden: int = 256, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()

        base = GNNEval(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim, hidden=hidden, num_hidden_layers=num_layers, dropout=dropout)
        self.convs = base.convs
        self.norms = base.norms
        self.dropout_p = base.dropout_p

        self.policy_mlp = nn.Sequential(nn.Linear(hidden + move_feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value_mlp = nn.Sequential(nn.Linear(hidden + global_feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, data):
        x, edge_index, edge_attr, batch, g = data.x, data.edge_index, data.edge_attr, data.batch, data.global_feats
        h = x
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)
            if h.shape == h_in.shape:
                h = h + h_in
        pooled = global_mean_pool(h, batch)

        g_flat = g.view(g.size(0), -1)
        v_in = torch.cat([pooled, g_flat], dim=-1)
        value = self.value_mlp(v_in).squeeze(-1)

        move_feat = data.move_feat
        bsize = g.size(0)
        if move_feat.dim() == 1:
            move_feat = move_feat.view(bsize, -1)
        elif move_feat.dim() == 2 and move_feat.size(0) != bsize:
            move_feat = move_feat.view(bsize, -1)
        p_in = torch.cat([pooled, move_feat], dim=-1)
        policy_logit = self.policy_mlp(p_in).squeeze(-1)
        return policy_logit, value


def train(dataset_path: str, epochs: int, batch_size: int, lr: float, device: str, out_path: str | None):
    device_t = torch.device(device)
    try:
        samples: List = torch.load(dataset_path)
    except Exception:
        samples: List = torch.load(dataset_path, weights_only=False)
    if not samples:
        raise ValueError("Empty dataset")
    samples = [s for s in samples if hasattr(s, "move_feat")]

    random.shuffle(samples)
    split = max(1, int(len(samples) * 0.95))
    train_s = samples[:split]
    val_s = samples[split:]

    sample0 = train_s[0]
    node_feat_dim = sample0.x.size(1)
    global_feat_dim = sample0.global_feats.size(1)
    move_feat_dim = sample0.move_feat.view(-1).size(0)

    model = PolicyValueNet(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim, move_feat_dim=move_feat_dim).to(device_t)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    policy_crit = nn.BCEWithLogitsLoss()
    value_crit = nn.MSELoss()

    train_loader = DataLoader(train_s, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_s, batch_size=batch_size, shuffle=False) if val_s else None

    best_val = None
    for epoch in range(epochs):
        model.train()
        tot_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device_t)
            p_logit, v = model(batch)
            p_target = batch.y.view(-1)
            v_target = batch.value.view(-1)
            loss_p = policy_crit(p_logit, p_target)
            loss_v = value_crit(torch.sigmoid(v), v_target)
            loss = loss_p + loss_v
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss.item())

        avg_loss = tot_loss / max(1, len(train_loader))
        val_loss = None
        if val_loader:
            model.eval()
            v_tot = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device_t)
                    p_logit, v = model(batch)
                    p_target = batch.y.view(-1)
                    v_target = batch.value.view(-1)
                    loss_p = policy_crit(p_logit, p_target)
                    loss_v = value_crit(torch.sigmoid(v), v_target)
                    v_tot += float((loss_p + loss_v).item())
            val_loss = v_tot / max(1, len(val_loader))

        print(f"epoch {epoch+1}/{epochs} train_loss={avg_loss:.4f} val_loss={val_loss if val_loss is not None else 'NA'}")
        if out_path and val_loss is not None:
            if best_val is None or val_loss < best_val:
                torch.save(model.state_dict(), out_path)
                best_val = val_loss
                print(f"Saved best model to {out_path} (val_loss={best_val:.6f})")


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero policy+value GNN")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    train(args.dataset, args.epochs, args.batch_size, args.lr, args.device, args.out)


if __name__ == "__main__":
    main()
