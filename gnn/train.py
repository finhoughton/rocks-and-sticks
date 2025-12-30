from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gnn.model import GNNEval


def train(
    train_dataset: list[Data],
    val_dataset: list[Data],
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    best_model_out: Optional[str] = None,
) -> tuple[GNNEval, list[float], list[float]]:

    if not train_dataset:
        raise ValueError("Dataset is empty; generate samples first.")
    device_t = torch.device(device)
    sample = train_dataset[0]
    node_feat_dim = sample.x.size(1) # type: ignore
    global_feat_dim = sample.global_feats.size(1)
    model = GNNEval(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim).to(device_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Use per-sample loss so we can apply the same weighting on train and val
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    train_losses: list[float] = []
    val_losses: list[float] = []

    best_val: float | None = None
    best_epoch: int | None = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        train_correct_w = 0.0
        train_pos_prob_sum = 0.0
        train_neg_prob_sum = 0.0
        train_pos_w = 0.0
        train_neg_w = 0.0
        for batch in train_loader:
            batch = batch.to(device_t)
            logits = model(batch).view(-1)
            labels = batch.y.view(-1).float()
            weights = batch.weight.view(-1).float()
            per_sample = criterion(logits, labels)
            # weighted mean loss
            loss = (per_sample * weights).sum() / max(1e-12, weights.sum())
            opt.zero_grad()
            loss.backward()
            opt.step()  # type: ignore

            total_loss += float(loss.item()) * 1.0
            total_weight += float(weights.sum().item())

            probs = torch.sigmoid(logits.detach())
            preds = (probs >= 0.5).float()
            train_correct_w += float(((preds == labels).float() * weights).sum().item())
            pos_mask = (labels == 1.0).float()
            neg_mask = (labels == 0.0).float()
            train_pos_prob_sum += float((probs * pos_mask * weights).sum().item())
            train_pos_w += float((pos_mask * weights).sum().item())
            train_neg_prob_sum += float((probs * neg_mask * weights).sum().item())
            train_neg_w += float((neg_mask * weights).sum().item())

        # epoch-level weighted averages
        avg_loss = total_loss / max(1, len(train_loader))
        train_acc = train_correct_w / max(1e-12, total_weight)
        train_pos_prob = train_pos_prob_sum / max(1e-12, train_pos_w)
        train_neg_prob = train_neg_prob_sum / max(1e-12, train_neg_w)

        val_loss = None
        val_acc = float('nan')
        val_pos_prob = float('nan')
        val_neg_prob = float('nan')
        if val_loader is not None:
            model.eval()
            v_weighted_loss = 0.0
            v_total_w = 0.0
            v_correct_w = 0.0
            v_pos_prob_sum = 0.0
            v_neg_prob_sum = 0.0
            v_pos_w = 0.0
            v_neg_w = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device_t)
                    logits = model(batch).view(-1)
                    labels = batch.y.view(-1).float()
                    weights = batch.weight.view(-1).float()
                    per_sample = criterion(logits, labels)
                    v_weighted_loss += float((per_sample * weights).sum().item())
                    v_total_w += float(weights.sum().item())

                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    v_correct_w += float(((preds == labels).float() * weights).sum().item())
                    pos_mask = (labels == 1.0).float()
                    neg_mask = (labels == 0.0).float()
                    v_pos_prob_sum += float((probs * pos_mask * weights).sum().item())
                    v_pos_w += float((pos_mask * weights).sum().item())
                    v_neg_prob_sum += float((probs * neg_mask * weights).sum().item())
                    v_neg_w += float((neg_mask * weights).sum().item())

            if v_total_w > 0:
                val_loss = v_weighted_loss / v_total_w
                val_acc = v_correct_w / v_total_w
                val_pos_prob = v_pos_prob_sum / max(1e-12, v_pos_w)
                val_neg_prob = v_neg_prob_sum / max(1e-12, v_neg_w)
                if best_model_out is not None:
                    if (best_val is None) or (val_loss < best_val):
                        try:
                            torch.save(model.state_dict(), best_model_out)
                            best_val = float(val_loss)
                            best_epoch = epoch + 1
                            print(f"Saved best model to {best_model_out} at epoch {best_epoch} (val_loss={best_val:.6f})")
                        except Exception as e:
                            print(f"Failed to save best model to {best_model_out}: {e}")

        train_losses.append(avg_loss)
        val_losses.append(val_loss if val_loss is not None else float("nan"))

        if val_loss is None:
            print(f"epoch {epoch+1}/{epochs} train loss={avg_loss:.4f} train_acc={train_acc:.4f} pos_p={train_pos_prob:.4f} neg_p={train_neg_prob:.4f}")
        else:
            print(f"epoch {epoch+1}/{epochs} train loss={avg_loss:.4f} train_acc={train_acc:.4f} test loss={val_loss:.4f} test_acc={val_acc:.4f} pos_p={val_pos_prob:.4f} neg_p={val_neg_prob:.4f}")
    model.eval()
    return model, train_losses, val_losses
