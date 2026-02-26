from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import resource
import time
from contextlib import nullcontext
from typing import Any, Dict, Iterator, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.data as pyg_data
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
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
        state_id_mod: int | None = None,
        state_id_rem: int = 0,
        state_id_keep_equal: bool = True,
    ) -> None:
        super().__init__()
        self.shard_paths = list(shard_paths)
        self._length = int(sum(shard_counts)) if shard_counts is not None else None
        self.shuffle = shuffle
        self.seed = seed
        self.state_id_mod = int(state_id_mod) if state_id_mod is not None else None
        self.state_id_rem = int(state_id_rem)
        self.state_id_keep_equal = bool(state_id_keep_equal)

    def __len__(self) -> int:
        return int(self._length or 0)

    def __iter__(self) -> Iterator[Any]:
        # Be robust to older pickled instances in DataLoader workers (macOS uses
        # spawn by default). If the main process constructed an instance before
        # these attributes existed, the worker may unpickle it without them.
        if not hasattr(self, "state_id_mod"):
            self.state_id_mod = None  # type: ignore[attr-defined]
        if not hasattr(self, "state_id_rem"):
            self.state_id_rem = 0  # type: ignore[attr-defined]
        if not hasattr(self, "state_id_keep_equal"):
            self.state_id_keep_equal = True  # type: ignore[attr-defined]

        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1

        seed = int(getattr(self, "seed", 0))
        shuffle = bool(getattr(self, "shuffle", False))
        shard_paths = list(getattr(self, "shard_paths", []))

        rng = random.Random(seed + worker_id)
        if shuffle:
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
            if shuffle:
                rng.shuffle(shard)
            state_id_mod = getattr(self, "state_id_mod", None)

            def _is_grouped_sample(s: Any) -> bool:
                return (
                    isinstance(s, dict)
                    and ("graph" in s)
                    and ("move_feat" in s)
                    and ("policy" in s)
                )

            if state_id_mod is None:
                for s in shard:
                    if _is_grouped_sample(s) or hasattr(s, "move_feat"):
                        yield s
            else:
                mod = int(state_id_mod)
                rem = int(getattr(self, "state_id_rem", 0))
                keep_equal = bool(getattr(self, "state_id_keep_equal", True))
                idx = 0
                for s in shard:
                    if not (_is_grouped_sample(s) or hasattr(s, "move_feat")):
                        idx += 1
                        continue
                    sid = None
                    if _is_grouped_sample(s):
                        try:
                            sid = int(s.get("state_id", idx))
                        except Exception:
                            sid = None
                    elif hasattr(s, "state_id"):
                        try:
                            sid = int(s.state_id.view(-1)[0].item())
                        except Exception:
                            sid = None
                    if sid is None:
                        sid = idx
                    idx += 1
                    ok = (sid % mod) == rem
                    if ok == keep_equal:
                        yield s
            del shard
            gc.collect()


def _is_grouped_sample(s: Any) -> bool:
    return isinstance(s, dict) and ("graph" in s) and ("move_feat" in s) and ("policy" in s)


def _collate_grouped(samples: list[dict]) -> dict[str, Any]:
    # One graph per sample, but variable number of moves per graph.
    graphs = [s["graph"] for s in samples]
    graph_batch = pyg_data.Batch.from_data_list(graphs)

    move_feats: list[torch.Tensor] = []
    policy: list[torch.Tensor] = []
    owners: list[torch.Tensor] = []
    values: list[float] = []

    for i, s in enumerate(samples):
        mf = torch.as_tensor(s["move_feat"]).to(torch.float32)
        pi = torch.as_tensor(s["policy"]).to(torch.float32).view(-1)
        if mf.dim() != 2:
            raise ValueError(f"grouped sample move_feat must be 2D, got shape={tuple(mf.shape)}")
        if pi.numel() != int(mf.size(0)):
            raise ValueError(f"grouped sample policy len != K (policy={pi.numel()} K={int(mf.size(0))})")
        move_feats.append(mf)
        policy.append(pi)
        owners.append(torch.full((int(mf.size(0)),), int(i), dtype=torch.long))
        values.append(float(s.get("value", 0.5)))

    move_feat_all = torch.cat(move_feats, dim=0) if move_feats else torch.empty((0, 0), dtype=torch.float32)
    policy_all = torch.cat(policy, dim=0) if policy else torch.empty((0,), dtype=torch.float32)
    owner_all = torch.cat(owners, dim=0) if owners else torch.empty((0,), dtype=torch.long)
    value_targets = torch.tensor(values, dtype=torch.float32)

    return {
        "graph": graph_batch,
        "move_feat": move_feat_all,
        "move_owner": owner_all,
        "policy": policy_all,
        "value": value_targets,
    }


def _grouped_log_softmax(logits: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
    # logits: [M], group: [M] with values in 0..B-1
    if logits.numel() == 0:
        return logits
    device = logits.device
    num_groups = int(group.max().item()) + 1 if group.numel() else 0
    out = torch.empty_like(logits)
    for g in range(num_groups):
        mask = group == g
        if not mask.any():
            continue
        lg = logits[mask]
        out[mask] = torch.nn.functional.log_softmax(lg, dim=0)
    return out


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
    def __init__(self, node_feat_dim: int, global_feat_dim: int, move_feat_dim: int = 5, hidden: int = 384, num_layers: int = 5, dropout: float = 0.1):
        super().__init__()

        base = GNNEval(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim, hidden=hidden, num_hidden_layers=num_layers, dropout=dropout)
        self.convs = base.convs
        self.norms = base.norms
        self.dropout_p = base.dropout_p

        self.policy_mlp = nn.Sequential(nn.Linear(hidden + move_feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.value_mlp = nn.Sequential(nn.Linear(hidden + global_feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, data):
        pooled, g_flat = self.encode_graph(data)
        v_in = torch.cat([pooled, g_flat], dim=-1)
        value = self.value_mlp(v_in).squeeze(-1)

        move_feat = data.move_feat
        bsize = g_flat.size(0)
        if move_feat.dim() == 1:
            move_feat = move_feat.view(bsize, -1)
        elif move_feat.dim() == 2 and move_feat.size(0) != bsize:
            move_feat = move_feat.view(bsize, -1)
        p_in = torch.cat([pooled, move_feat], dim=-1)
        policy_logit = self.policy_mlp(p_in).squeeze(-1)
        return policy_logit, value

    def encode_graph(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, edge_attr, batch, g = data.x, data.edge_index, data.edge_attr, data.batch, data.global_feats
        h = x
        for i, conv in enumerate(self.convs):
            h_in = h
            h = conv(h, edge_index, edge_attr)
            h = self.norms[i](h, batch)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)
            if h.shape == h_in.shape:
                h = h + h_in
        pooled = global_mean_pool(h, batch)
        g_flat = g.view(g.size(0), -1)
        return pooled, g_flat

    def policy_logits_grouped(self, pooled: torch.Tensor, move_feat: torch.Tensor, move_owner: torch.Tensor) -> torch.Tensor:
        # pooled: [B, H], move_feat: [M, D], move_owner: [M] in [0..B-1]
        p_in = torch.cat([pooled[move_owner], move_feat], dim=-1)
        return self.policy_mlp(p_in).squeeze(-1)


def train(
    dataset_path: str | list[str],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    out_path: str | None,
    init_from: str | None = None,
    amp: bool = False,
    clear_cache_interval: int | None = 200,
    mps_cache_ratio: float | None = None,
    rss_log_interval: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    steps_per_epoch: int | None = None,
    seed: int | None = None,
    diagnostics_out: str | None = None,
    value_weight: float = 1.0,
    value_lr_mult: float = 1.0,
):
    device_t = torch.device(device)
    train_loader = None
    val_loader = None

    if seed is not None:
        seed_i = int(seed)
        random.seed(seed_i)
        torch.manual_seed(seed_i)
        if device_t.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_i)
    
    dataset_paths = [dataset_path] if isinstance(dataset_path, str) else list(dataset_path)
    if not dataset_paths:
        raise ValueError("No dataset paths provided")

    # Support training from multiple datasets (replay buffer) by combining sharded manifests.
    dataset_objs = [_load_dataset_obj(p) for p in dataset_paths]
    manifests = [_maybe_sharded_manifest(obj) for obj in dataset_objs]
    all_sharded = all(m is not None for m in manifests)
    any_sharded = any(m is not None for m in manifests)

    if any_sharded and not all_sharded:
        raise ValueError("Mixed sharded and non-sharded datasets are not supported")

    # If not sharded, just concatenate all samples in memory.
    if not any_sharded:
        samples_all: List = []
        for obj in dataset_objs:
            if not isinstance(obj, list):
                raise ValueError("Expected non-sharded dataset to be a list")
            samples_all.extend(obj)
        dataset_obj = samples_all
        manifest = None
    else:
        # Combine all shards into a single manifest-like stream (absolute shard paths).
        combined_shard_items: list[tuple[str, int]] = []
        for dp, man in zip(dataset_paths, manifests):
            assert man is not None
            base_dir = os.path.dirname(dp) or "."
            shard_dir = os.path.join(base_dir, str(man.get("shard_dir")))
            shard_names = list(man.get("shards", []))
            shard_counts = list(man.get("counts", []))
            if len(shard_names) != len(shard_counts):
                raise ValueError(f"Sharded dataset manifest mismatch in {dp}")
            for name, cnt in zip(shard_names, shard_counts):
                combined_shard_items.append((os.path.join(shard_dir, str(name)), int(cnt)))
        dataset_obj = {"format": "sharded-v1", "combined": True}
        manifest = {"format": "sharded-v1", "combined": True, "items": combined_shard_items}

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

    grouped_policy = False

    if manifest is None:
        if not isinstance(dataset_obj, list):
            raise ValueError("Expected dataset_obj to be a list when manifest is None")
        samples = dataset_obj
        if not samples:
            raise ValueError("Empty dataset")
        grouped_policy = _is_grouped_sample(samples[0])
        if grouped_policy:
            samples = [s for s in samples if _is_grouped_sample(s)]
        else:
            samples = [s for s in samples if hasattr(s, "move_feat")]

        random.shuffle(samples)
        split = max(1, int(len(samples) * 0.95))
        train_s = samples[:split]
        val_s = samples[split:]
        train_len = len(train_s)
        del samples
        sample0 = train_s[0]
    else:
        shard_items = list(manifest.get("items", []))  # type: ignore[arg-type]
        if not shard_items:
            raise ValueError("Sharded dataset manifest has no shards")
        random.shuffle(shard_items)
        target_train = max(1, int(sum(cnt for _, cnt in shard_items) * 0.95))
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

        # With coarse sharding, a count-based 95/5 split can easily end up placing
        # *all* shards in train (e.g. if all but one shard are large). Ensure there
        # is at least one validation shard when possible.
        if not val_shards and len(train_shards) > 1:
            sp = train_shards.pop()
            cnt = train_counts.pop()
            val_shards.append(sp)
            val_counts.append(int(cnt))

        # infer dims from first shard
        first_shard = _load_dataset_obj(train_shards[0])
        if not isinstance(first_shard, list) or not first_shard:
            raise ValueError(f"Empty shard file: {train_shards[0]}")
        grouped_policy = _is_grouped_sample(first_shard[0])
        if grouped_policy:
            sample0 = next((s for s in first_shard if _is_grouped_sample(s)), None)
        else:
            sample0 = next((s for s in first_shard if hasattr(s, "move_feat")), None)
        if sample0 is None:
            raise ValueError(f"No training samples with move_feat in shard: {train_shards[0]}")

        # Important: don't keep the whole first shard alive for the rest of training.
        del first_shard
        gc.collect()

        train_s = (train_shards, train_counts)
        val_s = (val_shards, val_counts)
        train_len = int(sum(train_counts))
    if grouped_policy:
        g0 = sample0["graph"]
        node_feat_dim = g0.x.size(1)
        global_feat_dim = g0.global_feats.size(1)
        move_feat_dim = int(torch.as_tensor(sample0["move_feat"]).size(1))
    else:
        node_feat_dim = sample0.x.size(1)
        global_feat_dim = sample0.global_feats.size(1)
        move_feat_dim = sample0.move_feat.view(-1).size(0)

    model = PolicyValueNet(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim, move_feat_dim=move_feat_dim).to(device_t)

    if init_from:
        try:
            ck = torch.load(str(init_from), map_location=device_t)
            state = ck if isinstance(ck, dict) and 'state_dict' not in ck else ck.get('state_dict', ck)
            if isinstance(state, dict):
                # If this is a GNNEval checkpoint (has 'head.*' but no 'value_mlp.*'),
                # remap the value head keys so they load into PolicyValueNet.
                has_head = any(str(k).startswith('head.') for k in state)
                has_value_mlp = any(str(k).startswith('value_mlp.') for k in state)
                if has_head and not has_value_mlp:
                    # GNNEval head: [Linear(0), ReLU(1), Dropout(2), Linear(3)]
                    # PolicyValueNet value_mlp: [Linear(0), ReLU(1), Linear(2)]
                    remap = {
                        'head.0.weight': 'value_mlp.0.weight',
                        'head.0.bias': 'value_mlp.0.bias',
                        'head.3.weight': 'value_mlp.2.weight',
                        'head.3.bias': 'value_mlp.2.bias',
                    }
                    for old_key, new_key in remap.items():
                        if old_key in state:
                            state[new_key] = state.pop(old_key)
                    print(f"Remapped GNNEval head → value_mlp ({len(remap)} keys)")

                missing, unexpected = model.load_state_dict(state, strict=False)
                print(
                    f"Warm-started from {init_from} "
                    f"(missing={len(missing)} unexpected={len(unexpected)})"
                )
            else:
                print(f"Warning: --init-from {init_from} did not contain a state dict; ignoring")
        except Exception as e:
            print(f"Warning: failed to warm-start from {init_from}: {e}")

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
    # collect value head params separately so we can apply an LR multiplier
    value_params = list(model.value_mlp.parameters())
    value_param_ids = {id(p) for p in value_params}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # skip value params; they'll be added to their own group below
        if id(param) in value_param_ids:
            continue
        n = name.lower()
        if n.endswith(".bias") or "norm" in n or "bn" in n:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.0},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    # add value head param group with scaled LR
    if value_params:
        optim_groups.append({"params": value_params, "weight_decay": 0.0, "lr": float(lr) * float(value_lr_mult)})
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

    try:
        for epoch in range(epochs):
            if manifest is None:
                if grouped_policy:
                    train_loader = TorchDataLoader(train_s, batch_size=batch_size, shuffle=True, collate_fn=_collate_grouped, **loader_kwargs)
                    val_loader = TorchDataLoader(val_s, batch_size=batch_size, shuffle=False, collate_fn=_collate_grouped, **loader_kwargs) if val_s else None
                else:
                    train_loader = PyGDataLoader(train_s, batch_size=batch_size, shuffle=True, **loader_kwargs)
                    val_loader = PyGDataLoader(val_s, batch_size=batch_size, shuffle=False, **loader_kwargs) if val_s else None
            else:
                train_shards, train_counts = train_s
                val_shards, val_counts = val_s
                if val_shards:
                    train_ds = _ShardIterableDataset(train_shards, train_counts, shuffle=True, seed=1234 + epoch)
                    if grouped_policy:
                        train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_grouped, **loader_kwargs)  # type: ignore
                    else:
                        train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)  # type: ignore
                    val_ds = _ShardIterableDataset(val_shards, val_counts, shuffle=False, seed=4242)
                    if grouped_policy:
                        val_loader = TorchDataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_grouped, **loader_kwargs)  # type: ignore
                    else:
                        val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)  # type: ignore
                else:
                    # If the shard-level split produced no validation shards (common when there's only one shard),
                    # fall back to a deterministic per-state split based on state_id.
                    split_mod = 20  # 5% validation (state_id % 20 == 0)
                    train_ds = _ShardIterableDataset(
                        train_shards,
                        train_counts,
                        shuffle=True,
                        seed=1234 + epoch,
                        state_id_mod=split_mod,
                        state_id_rem=0,
                        state_id_keep_equal=False,
                    )
                    if grouped_policy:
                        train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_grouped, **loader_kwargs)  # type: ignore
                    else:
                        train_loader = PyGDataLoader(train_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)  # type: ignore
                    val_ds = _ShardIterableDataset(
                        train_shards,
                        train_counts,
                        shuffle=False,
                        seed=4242,
                        state_id_mod=split_mod,
                        state_id_rem=0,
                        state_id_keep_equal=True,
                    )
                    if grouped_policy:
                        val_loader = TorchDataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_grouped, **loader_kwargs)  # type: ignore
                    else:
                        val_loader = PyGDataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)  # type: ignore

            model.train()
            tot_loss = 0.0
            n_batches = 0
            # diagnostics writer (jsonl)
            diag_fh = None
            if diagnostics_out:
                try:
                    PathDir = os.path.dirname(diagnostics_out) or "."
                    os.makedirs(PathDir, exist_ok=True)
                except Exception:
                    pass
                try:
                    diag_fh = open(diagnostics_out, "a", encoding="utf-8")
                except Exception:
                    diag_fh = None
            for batch in train_loader:
                if grouped_policy:
                    graph = batch["graph"].to(device_t)
                    move_feat = batch["move_feat"].to(device_t)
                    move_owner = batch["move_owner"].to(device_t)
                    p_targets = batch["policy"].to(device_t)
                    v_target = batch["value"].to(device_t)
                    with amp_ctx:
                        pooled, g_flat = model.encode_graph(graph)
                        v_in = torch.cat([pooled, g_flat], dim=-1)
                        v = model.value_mlp(v_in).squeeze(-1)

                        p_logit = model.policy_logits_grouped(pooled, move_feat, move_owner)
                        logp = _grouped_log_softmax(p_logit, move_owner)
                        bsz = int(pooled.size(0))
                        loss_p = -(p_targets * logp).sum() / max(1, bsz)
                        loss_v = value_crit(torch.sigmoid(v), v_target.view(-1))
                        loss = loss_p + float(value_weight) * loss_v
                else:
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
                        loss = loss_p + float(value_weight) * loss_v

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                scheduler.step()
                tot_loss += float(loss.item())
                n_batches += 1

                # write per-batch diagnostics if requested
                if diag_fh is not None:
                    try:
                        lr_now = float(opt.param_groups[0]["lr"]) if opt.param_groups else float(lr)
                        v_pred = torch.sigmoid(v.detach())
                        v_tgt = v_target.detach().view(-1)
                        v_diff = v_pred - v_tgt
                        v_mae = float(v_diff.abs().mean().item()) if v_diff.numel() else 0.0
                        v_mse = float((v_diff * v_diff).mean().item()) if v_diff.numel() else 0.0
                        v_rmse = float(math.sqrt(v_mse))
                        v_pred_mean = float(v_pred.mean().item()) if v_pred.numel() else 0.0
                        v_tgt_mean = float(v_tgt.mean().item()) if v_tgt.numel() else 0.0
                        v_pred_min = float(v_pred.min().item()) if v_pred.numel() else 0.0
                        v_pred_max = float(v_pred.max().item()) if v_pred.numel() else 0.0
                        v_tgt_min = float(v_tgt.min().item()) if v_tgt.numel() else 0.0
                        v_tgt_max = float(v_tgt.max().item()) if v_tgt.numel() else 0.0
                        # compute grad norm
                        gn = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                try:
                                    gn += float(p.grad.data.float().norm().item() ** 2)
                                except Exception:
                                    pass
                        grad_norm = float(math.sqrt(max(0.0, gn)))
                        rec = {
                            "ts": time.time(),
                            "epoch": int(epoch),
                            "batch": int(n_batches),
                            "loss": float(loss.item()),
                            "loss_p": float(loss_p) if 'loss_p' in locals() else None,
                            "loss_v": float(loss_v) if 'loss_v' in locals() else None,
                            "lr": lr_now,
                            "grad_norm": grad_norm,
                            "value_pred_mean": v_pred_mean,
                            "value_target_mean": v_tgt_mean,
                            "value_pred_min": v_pred_min,
                            "value_pred_max": v_pred_max,
                            "value_target_min": v_tgt_min,
                            "value_target_max": v_tgt_max,
                            "value_mae": v_mae,
                            "value_rmse": v_rmse,
                            "value_mse": v_mse,
                        }
                        diag_fh.write(json.dumps(rec) + "\n")
                        if n_batches % 50 == 0:
                            diag_fh.flush()
                    except Exception:
                        pass

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
                        if grouped_policy:
                            graph = batch["graph"].to(device_t)
                            move_feat = batch["move_feat"].to(device_t)
                            move_owner = batch["move_owner"].to(device_t)
                            p_targets = batch["policy"].to(device_t)
                            v_target = batch["value"].to(device_t)
                            with amp_ctx:
                                pooled, g_flat = model.encode_graph(graph)
                                v_in = torch.cat([pooled, g_flat], dim=-1)
                                v = model.value_mlp(v_in).squeeze(-1)
                                p_logit = model.policy_logits_grouped(pooled, move_feat, move_owner)
                                logp = _grouped_log_softmax(p_logit, move_owner)
                                bsz = int(pooled.size(0))
                                loss_p = -(p_targets * logp).sum() / max(1, bsz)
                                loss_v = value_crit(torch.sigmoid(v), v_target.view(-1))
                                v_tot += float((loss_p + loss_v).item())
                        else:
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
        if out_path:
            torch.save(model.state_dict(), out_path)
            print(f"Saved final model to {out_path}")

        _clear_device_cache(device_t)
    finally:
        try:
            if diag_fh is not None:
                diag_fh.close()
        except Exception:
            pass
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
    parser.add_argument("--dataset", type=str, required=True, nargs='+', help="One or more dataset .pt files (use multiple for replay buffer)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--init-from", type=str, default=None, help="Warm-start from a previous PolicyValueNet checkpoint")
    parser.add_argument("--amp", action="store_true", help="Enable autocast (mps/cuda) to reduce memory usage")
    parser.add_argument("--clear-cache-interval", type=int, default=200, help="Clear device cache every N batches on mps; set 0 to disable")
    parser.add_argument("--mps-cache-ratio", type=float, default=None, help="Set PYTORCH_MPS_HIGH_WATERMARK_RATIO (e.g., 0.3) to limit MPS allocator cache")
    parser.add_argument("--rss-log-interval", type=int, default=None, help="Print ru_maxrss every N batches to track OS-reported memory")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for batching/collation")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory (useful for CUDA; harmless on MPS)")
    parser.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive between epochs (requires num-workers>0)")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Batches prefetched per worker (requires num-workers>0)")
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="If >0, limit training to this many batches per epoch (helps stability on MPS)")
    parser.add_argument("--seed", type=int, default=None, help="If set, seed Python+Torch RNGs for deterministic shuffles/splits")
    parser.add_argument("--diagnostics-out", type=str, default=None, help="Optional JSONL file to append per-batch training diagnostics")
    parser.add_argument("--value-weight", type=float, default=1.0, help="Weight applied to value loss when forming total loss")
    parser.add_argument("--value-lr-mult", type=float, default=1.0, help="Multiplier applied to base LR for the value head")
    args = parser.parse_args()
    train(
        args.dataset,
        args.epochs,
        args.batch_size,
        args.lr,
        args.device,
        args.out,
        args.init_from,
        args.amp,
        args.clear_cache_interval,
        args.mps_cache_ratio,
        args.rss_log_interval,
        args.num_workers,
        args.pin_memory,
        args.persistent_workers,
        args.prefetch_factor,
        int(args.steps_per_epoch) if int(args.steps_per_epoch) > 0 else None,
        args.seed,
        diagnostics_out=args.diagnostics_out,
        value_weight=args.value_weight,
        value_lr_mult=args.value_lr_mult,
    )


if __name__ == "__main__":
    main()
