from typing import Any, cast

import torch


def plot_losses(train_losses: list[float], val_losses: list[float], out_path: str) -> None:
    epochs = list(range(1, len(train_losses) + 1))
    import matplotlib.pyplot as plt
    plt = cast(Any, plt)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="train")
    if any(torch.isfinite(torch.tensor(val_losses))):
        plt.plot(epochs, val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.title("GNN eval loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"saved loss plot to {out_path}")
