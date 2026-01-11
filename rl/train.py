from __future__ import annotations

import argparse
import gc
import math
import os
import random
import resource
from contextlib import nullcontext
from typing import Any, Dict, Iterator, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax

from gnn.model import GNNEval


def _clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


class _ShardIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        shard_paths: Sequence[str],
        shard_counts: Sequence[int] | None,
        *,
        shuffle: bool,
        seed: int,
    ) -> None:
        super().__init__()
        self.shard_paths = list(shard_paths)
        self._length = int(sum(shard_counts)) if shard_counts is not None else None
        self.shuffle = shuffle
        self.seed = seed

    def __len__(self) -> int:
        return int(self._length or 0)

    def __iter__(self) -> Iterator[Any]:
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1

        rng = random.Random(self.seed + worker_id)
        shard_paths = list(self.shard_paths)
        if self.shuffle:
            rng.shuffle(shard_paths)

        # split shards across workers to avoid duplication
        shard_paths = shard_paths[worker_id::num_workers]

        for shard_path in shard_paths:
            try:
                shard = torch.load(shard_path)
            except Exception:
                shard = torch.load(shard_path, weights_only=False)
            if isinstance(shard, dict) and shard.get("format") == "sharded-v1":
                raise ValueError(f"Nested sharded dataset manifest in shard file: {shard_path}")
            if not isinstance(shard, list):
                continue
            if self.shuffle:
                rng.shuffle(shard)
            for s in shard:
                if hasattr(s, "move_feat"):
                    yield s
            del shard
            gc.collect()


def _load_dataset_obj(dataset_path: str) -> Any:
    try:
        return torch.load(dataset_path)
    except Exception as e:
        # If the manifest write was interrupted, the .pt can be corrupt.
        # Attempt to reconstruct a sharded manifest from disk.
        shard_dir = dataset_path + ".shards"
        if os.path.isdir(shard_dir):
            shard_files = sorted(
                f for f in os.listdir(shard_dir)
                if f.startswith("shard_") and f.endswith(".pt")
            )
            if shard_files:
                shard_paths = [os.path.join(shard_dir, f) for f in shard_files]
                counts: List[int] = []
                for sp in shard_paths:
                    try:
                        shard = torch.load(sp)
                    except Exception:
                        shard = torch.load(sp, weights_only=False)
                    if isinstance(shard, list):
                        counts.append(len(shard))
                    else:
                        counts.append(0)
                return {
                    "format": "sharded-v1",
                    "shard_dir": os.path.basename(shard_dir),
                    "shards": shard_files,
                    "counts": counts,
                    "total": int(sum(counts)),
                    "recovered": True,
                    "load_error": repr(e),
                }
        # Fall back to the older pickle path.
        return torch.load(dataset_path, weights_only=False)


def _maybe_sharded_manifest(obj: Any) -> Dict[str, Any] | None:
    if isinstance(obj, dict) and obj.get("format") == "sharded-v1":
        return obj  # type: ignore[return-value]
    return None


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


def train(
    dataset_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    out_path: str | None,
    amp: bool = False,
    clear_cache_interval: int | None = 200,
    mps_cache_ratio: float | None = None,
    rss_log_interval: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    steps_per_epoch: int | None = None,
):
    device_t = torch.device(device)
    train_loader = None
    val_loader = None
    dataset_obj = _load_dataset_obj(dataset_path)
    manifest = _maybe_sharded_manifest(dataset_obj)

    use_amp = bool(amp) and device_t.type in ("cuda", "mps")
    amp_dtype = torch.bfloat16 if device_t.type == "mps" else torch.float16
    amp_ctx = torch.autocast(device_type=device_t.type, dtype=amp_dtype) if use_amp else nullcontext()

    if device_t.type == "mps" and mps_cache_ratio is not None:
        try:
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = str(max(0.1, float(mps_cache_ratio)))
            print(f"Set PYTORCH_MPS_HIGH_WATERMARK_RATIO={os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']}")
        except Exception as e:
            print(f"Warning: failed to set PYTORCH_MPS_HIGH_WATERMARK_RATIO: {e}")

    train_s: Any
    val_s: Any
    train_len: int

    if manifest is None:
        samples: List = dataset_obj
        if not samples:
            raise ValueError("Empty dataset")
        samples = [s for s in samples if hasattr(s, "move_feat")]

        random.shuffle(samples)
        split = max(1, int(len(samples) * 0.95))
        train_s = samples[:split]
        val_s = samples[split:]
        train_len = len(train_s)
        del samples
        sample0 = train_s[0]
    else:
        base_dir = os.path.dirname(dataset_path) or "."
        shard_dir = os.path.join(base_dir, str(manifest.get("shard_dir")))
        shard_names = list(manifest.get("shards", []))
        shard_counts = list(manifest.get("counts", []))
        if not shard_names:
            raise ValueError("Sharded dataset manifest has no shards")
        shard_paths = [os.path.join(shard_dir, s) for s in shard_names]

        shard_items = list(zip(shard_paths, shard_counts))
        random.shuffle(shard_items)
        target_train = max(1, int(sum(shard_counts) * 0.95))
        train_shards: List[str] = []
        train_counts: List[int] = []
        val_shards: List[str] = []
        val_counts: List[int] = []
        acc = 0
        for sp, cnt in shard_items:
            if acc < target_train:
                train_shards.append(sp)
                train_counts.append(int(cnt))
                acc += int(cnt)
            else:
                val_shards.append(sp)
                val_counts.append(int(cnt))

        # infer dims from first shard
        first_shard = _load_dataset_obj(train_shards[0])
        if not isinstance(first_shard, list) or not first_shard:
            raise ValueError(f"Empty shard file: {train_shards[0]}")
        sample0 = next((s for s in first_shard if hasattr(s, "move_feat")), None)
        if sample0 is None:
            raise ValueError(f"No training samples with move_feat in shard: {train_shards[0]}")

        # Important: don't keep the whole first shard alive for the rest of training.
        del first_shard
        gc.collect()

        train_s = (train_shards, train_counts)
        val_s = (val_shards, val_counts)
        train_len = int(sum(train_counts))
    node_feat_dim = sample0.x.size(1)
    global_feat_dim = sample0.global_feats.size(1)
    move_feat_dim = sample0.move_feat.view(-1).size(0)

    model = PolicyValueNet(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim, move_feat_dim=move_feat_dim).to(device_t)

    # CPU tuning
    n_cpus = max(1, (os.cpu_count() or 1))
    torch.set_num_threads(n_cpus)

    loader_kwargs: dict[str, Any] = {
        "num_workers": max(0, int(num_workers)),
        "persistent_workers": bool(persistent_workers) if int(num_workers) > 0 else False,
        "pin_memory": bool(pin_memory),
    }
    if loader_kwargs["num_workers"] > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))

    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = name.lower()
        if n.endswith(".bias") or "norm" in n or "bn" in n:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.0},
        # {"params": decay_params, "weight_decay": 1e-4},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(optim_groups, lr=lr)
    value_crit = nn.MSELoss()

    # LR scheduler: linear warmup (3% of steps) then cosine decay.
    # Important: respect the optional user-provided cap `steps_per_epoch`.
    dataset_steps_per_epoch = max(1, int(math.ceil(float(train_len) / float(batch_size))))
    cap_steps = int(steps_per_epoch) if steps_per_epoch is not None else 0
    effective_steps_per_epoch = (
        min(dataset_steps_per_epoch, cap_steps) if cap_steps > 0 else dataset_steps_per_epoch
    )
    total_steps = max(1, int(effective_steps_per_epoch * epochs))
    warmup_steps = int(0.03 * total_steps)
    def lr_lambda(step: int) -> float:
        if step < warmup_steps and warmup_steps > 0:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val = None
    try:
        for epoch in range(epochs):
            if manifest is None:
                train_loader = DataLoader(train_s, batch_size=batch_size, shuffle=True, **loader_kwargs)
                val_loader = DataLoader(val_s, batch_size=batch_size, shuffle=False, **loader_kwargs) if val_s else None
            else:
                train_shards, train_counts = train_s
                val_shards, val_counts = val_s
                train_ds = _ShardIterableDataset(train_shards, train_counts, shuffle=True, seed=1234 + epoch)
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)  # type: ignore
                if val_shards:
                    val_ds = _ShardIterableDataset(val_shards, val_counts, shuffle=False, seed=4242)
                    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)  # type: ignore
                else:
                    val_loader = None

            model.train()
            tot_loss = 0.0
            n_batches = 0
            for batch in train_loader:
                batch = batch.to(device_t)
                with amp_ctx:
                    p_logit, v = model(batch)
                    state_id = batch.state_id.view(-1)
                    p_targets = batch.y.view(-1)
                    _, inv = torch.unique(state_id, return_inverse=True)
                    num_groups = int(inv.max().item()) + 1 if inv.numel() else 0

                    ones = torch.ones_like(p_targets)
                    denom = torch.zeros(num_groups, device=device_t).scatter_add_(0, inv, p_targets)
                    counts = torch.zeros(num_groups, device=device_t).scatter_add_(0, inv, ones)

                    denom_safe = denom.clamp(min=1e-12)
                    tgt_norm = p_targets / denom_safe[inv]
                    zero_mask = denom[inv] <= 0
                    tgt = torch.where(zero_mask, 1.0 / counts[inv].clamp(min=1.0), tgt_norm)

                    p = pyg_softmax(p_logit, inv)
                    logp = torch.log(p.clamp(min=1e-12))
                    loss_p = -(tgt * logp).sum() / max(1, num_groups)

                    v_target = batch.value.view(-1)
                    loss_v = value_crit(torch.sigmoid(v), v_target)
                    loss = loss_p + loss_v

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                scheduler.step()
                tot_loss += float(loss.item())
                n_batches += 1

                if effective_steps_per_epoch > 0 and n_batches >= effective_steps_per_epoch:
                    break

                if device_t.type == "mps" and hasattr(torch, "mps"):
                    if n_batches % 50 == 0:
                        cur = torch.mps.current_allocated_memory()
                        driver = None
                        try:
                            driver = torch.mps.driver_allocated_memory()
                        except Exception:
                            driver = None
                        if driver is not None:
                            print(f"[mps memory] batch={n_batches} current={cur/1e6:.2f}MB driver={driver/1e6:.2f}MB")
                        else:
                            print(f"[mps memory] batch={n_batches} current={cur/1e6:.2f}MB")
                    if clear_cache_interval and clear_cache_interval > 0 and n_batches % clear_cache_interval == 0:
                        _clear_device_cache(device_t)

                if rss_log_interval and rss_log_interval > 0 and n_batches % rss_log_interval == 0:
                    try:
                        ru = resource.getrusage(resource.RUSAGE_SELF)
                        rss_bytes = float(ru.ru_maxrss if os.name == "posix" and hasattr(os, "uname") and os.uname().sysname == "Darwin" else ru.ru_maxrss * 1024.0)
                        rss_mb = rss_bytes / (1024.0 * 1024.0)
                        print(f"[rss] batch={n_batches} rss={rss_mb:.2f}MB (bytes={rss_bytes:.0f})")
                    except Exception as e:
                        print(f"[rss] batch={n_batches} failed to read rss: {e}")

            avg_loss = tot_loss / max(1, n_batches)
            val_loss = None
            if val_loader:
                model.eval()
                v_tot = 0.0
                v_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device_t)
                        with amp_ctx:
                            p_logit, v = model(batch)
                            state_id = batch.state_id.view(-1)
                            p_targets = batch.y.view(-1)
                            _, inv = torch.unique(state_id, return_inverse=True)
                            num_groups = int(inv.max().item()) + 1 if inv.numel() else 0

                            ones = torch.ones_like(p_targets)
                            denom = torch.zeros(num_groups, device=device_t).scatter_add_(0, inv, p_targets)
                            counts = torch.zeros(num_groups, device=device_t).scatter_add_(0, inv, ones)
                            denom_safe = denom.clamp(min=1e-12)
                            tgt_norm = p_targets / denom_safe[inv]
                            zero_mask = denom[inv] <= 0
                            tgt = torch.where(zero_mask, 1.0 / counts[inv].clamp(min=1.0), tgt_norm)

                            p = pyg_softmax(p_logit, inv)
                            logp = torch.log(p.clamp(min=1e-12))
                            loss_p = -(tgt * logp).sum() / max(1, num_groups)
                            loss_v = value_crit(torch.sigmoid(v), batch.value.view(-1))
                            v_tot += float((loss_p + loss_v).item())
                        v_batches += 1
                    val_loss = v_tot / max(1, v_batches)

            print(f"epoch {epoch+1}/{epochs} train_loss={avg_loss:.4f} val_loss={val_loss if val_loss is not None else 'NA'}")
            if out_path and val_loss is not None:
                if best_val is None or val_loss < best_val:
                    torch.save(model.state_dict(), out_path)
                    best_val = val_loss
                    print(f"Saved best model to {out_path} (val_loss={best_val:.6f})")

            _clear_device_cache(device_t)
    finally:
        try:
            del train_loader
            del val_loader
        except Exception:
            pass
        try:
            del train_s
            del val_s
        except Exception:
            pass
        try:
            del opt
            del model
        except Exception:
            pass
        gc.collect()
        _clear_device_cache(device_t)


def main():
    parser = argparse.ArgumentParser(description="Train AlphaZero policy+value GNN")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--amp", action="store_true", help="Enable autocast (mps/cuda) to reduce memory usage")
    parser.add_argument("--clear-cache-interval", type=int, default=200, help="Clear device cache every N batches on mps; set 0 to disable")
    parser.add_argument("--mps-cache-ratio", type=float, default=None, help="Set PYTORCH_MPS_HIGH_WATERMARK_RATIO (e.g., 0.3) to limit MPS allocator cache")
    parser.add_argument("--rss-log-interval", type=int, default=None, help="Print ru_maxrss every N batches to track OS-reported memory")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for batching/collation")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory (useful for CUDA; harmless on MPS)")
    parser.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive between epochs (requires num-workers>0)")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Batches prefetched per worker (requires num-workers>0)")
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="If >0, limit training to this many batches per epoch (helps stability on MPS)")
    args = parser.parse_args()
    train(
        args.dataset,
        args.epochs,
        args.batch_size,
        args.lr,
        args.device,
        args.out,
        args.amp,
        args.clear_cache_interval,
        args.mps_cache_ratio,
        args.rss_log_interval,
        args.num_workers,
        args.pin_memory,
        args.persistent_workers,
        args.prefetch_factor,
        int(args.steps_per_epoch) if int(args.steps_per_epoch) > 0 else None,
    )


if __name__ == "__main__":
    main()
