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
    init_from: Optional[str] = None,
    policy_weight: float = 0.0,
) -> tuple[GNNEval, list[float], list[float]]:

    if not train_dataset:
        raise ValueError("Dataset is empty; generate samples first.")
    device_t = torch.device(device)
    sample = train_dataset[0]
    node_feat_dim = sample.x.size(1) # type: ignore
    global_feat_dim = sample.global_feats.size(1)
    model = GNNEval(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim).to(device_t)
    if init_from:
        try:
            ckpt = torch.load(init_from, map_location=device_t)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            model.load_state_dict(ckpt, strict=False)
            print(f"Warm-started model from {init_from}")
        except Exception as e:
            print(f"Warning: failed to warm-start from {init_from}: {e}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Use per-sample loss so we can apply the same weighting on train and val
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    use_policy = policy_weight > 0.0

    train_losses: list[float] = []
    val_losses: list[float] = []

    best_val: float | None = None
    best_epoch: int | None = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_weight = 0.0
        total_policy_loss = 0.0
        n_policy_batches = 0
        train_correct_w = 0.0
        train_pos_prob_sum = 0.0
        train_neg_prob_sum = 0.0
        train_pos_w = 0.0
        train_neg_w = 0.0
        for batch in train_loader:
            batch = batch.to(device_t)
            labels = batch.y.view(-1).float()
            weights = batch.weight.view(-1).float()

            if use_policy:
                pooled_cat = model.backbone_forward(batch)
                logits = model.head(pooled_cat).squeeze(-1)
            else:
                logits = model(batch).view(-1)

            per_sample = criterion(logits, labels)
            value_loss = (per_sample * weights).sum() / max(1e-12, weights.sum())

            # Policy loss (per-sample, unbatched for variable-length move lists)
            policy_loss = torch.tensor(0.0, device=device_t)
            if use_policy and hasattr(batch, 'has_policy'):
                has_pol = batch.has_policy.view(-1)
                pol_mask = has_pol > 0.5
                if pol_mask.any():
                    p_losses = []
                    B = int(pooled_cat.shape[0])
                    # move_feats and policy_target are variable-size; with batch_size=1 this is trivial
                    if B == 1 and pol_mask[0]:
                        mf = batch.move_feats  # [K, 16]
                        pt = batch.policy_target  # [K]
                        if mf.dim() == 2 and mf.shape[0] > 0 and pt.dim() == 1 and pt.shape[0] > 0:
                            pc = pooled_cat  # [1, dim]
                            mf_3d = mf.unsqueeze(0)  # [1, K, 16]
                            pol_logits = model.policy_forward(pc, mf_3d)  # [1, K]
                            log_probs = torch.log_softmax(pol_logits, dim=-1)
                            p_loss = -(pt * log_probs).sum()
                            p_losses.append(p_loss)
                    if p_losses:
                        policy_loss = torch.stack(p_losses).mean()
                        total_policy_loss += float(policy_loss.item())
                        n_policy_batches += 1

            loss = value_loss + policy_weight * policy_loss
            opt.zero_grad()
            loss.backward()
            opt.step()  # type: ignore

            total_loss += float(value_loss.item())
            total_weight += float(weights.sum().item())

            probs = torch.sigmoid(logits.detach())
            preds = (probs >= 0.5).float()
            label_class = (labels >= 0.5).float()
            train_correct_w += float(((preds == label_class).float() * weights).sum().item())
            pos_mask = (labels >= 0.5).float()
            neg_mask = (labels < 0.5).float()
            train_pos_prob_sum += float((probs * pos_mask * weights).sum().item())
            train_pos_w += float((pos_mask * weights).sum().item())
            train_neg_prob_sum += float((probs * neg_mask * weights).sum().item())
            train_neg_w += float((neg_mask * weights).sum().item())

        # epoch-level weighted averages
        avg_loss = total_loss / max(1, len(train_loader))
        avg_policy = total_policy_loss / max(1, n_policy_batches) if n_policy_batches > 0 else 0.0
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
                    label_class = (labels >= 0.5).float()
                    v_correct_w += float(((preds == label_class).float() * weights).sum().item())
                    pos_mask = (labels >= 0.5).float()
                    neg_mask = (labels < 0.5).float()
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
                    if (best_val is None) or (val_loss * 0.96 < best_val):
                        try:
                            torch.save(model.state_dict(), best_model_out)
                            best_val = float(val_loss)
                            best_epoch = epoch + 1
                            print(f"Saved best model to {best_model_out} at epoch {best_epoch} (val_loss={best_val:.6f})")
                        except Exception as e:
                            print(f"Failed to save best model to {best_model_out}: {e}")

        train_losses.append(avg_loss)
        val_losses.append(val_loss if val_loss is not None else float("nan"))

        pol_str = f" policy_loss={avg_policy:.4f}" if use_policy else ""
        if val_loss is None:
            print(f"epoch {epoch+1}/{epochs} train loss={avg_loss:.4f} train_acc={train_acc:.4f} pos_p={train_pos_prob:.4f} neg_p={train_neg_prob:.4f}{pol_str}")
        else:
            print(f"epoch {epoch+1}/{epochs} train loss={avg_loss:.4f} train_acc={train_acc:.4f} test loss={val_loss:.4f} test_acc={val_acc:.4f} pos_p={val_pos_prob:.4f} neg_p={val_neg_prob:.4f}{pol_str}")
    model.eval()
    return model, train_losses, val_losses
