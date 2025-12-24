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
    criterion = torch.nn.BCEWithLogitsLoss()

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device_t)
            logits = model(batch)
            loss = criterion(logits, batch.y.view_as(logits))
            loss = (loss * batch.weight.view_as(loss)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step() # type: ignore
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_loader))
        val_loss = None
        if val_loader is not None:
            model.eval()
            v_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device_t)
                    logits = model(batch)
                    loss = criterion(logits, batch.y.view_as(logits))
                    v_total += float(loss.item())
            val_loss = v_total / max(1, len(val_loader))
        train_losses.append(avg_loss)
        val_losses.append(val_loss if val_loss is not None else float("nan"))

        if val_loss is None:
            print(f"epoch {epoch+1}/{epochs} train loss={avg_loss:.4f}")
        else:
            print(f"epoch {epoch+1}/{epochs} train loss={avg_loss:.4f} test loss={val_loss:.4f}")
    model.eval()
    return model, train_losses, val_losses
