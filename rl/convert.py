from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import torch
from torch_geometric.data import Data

from game import Game
from gnn.encode import encode_game_to_graph
from models import PASS, D, Move


def move_to_feat(m: dict) -> torch.Tensor:
    # Features: x, y, is_pass, is_rock, dir_idx (0..7 or -1)
    x = float(m["x"]) if "x" in m else 0.0
    y = float(m["y"]) if "y" in m else 0.0
    t = m.get("t", "P")
    is_pass = 1.0 if t == "P" else 0.0
    is_rock = 1.0 if t == "R" else 0.0
    # direction index for stick moves (use -1 for non-stick)
    return torch.tensor([x, y, is_pass, is_rock, float(convert_dir(t))], dtype=torch.float32)

def convert_dir(input_dir: str) -> int:
    try:
        return D[input_dir].as_int
    except Exception:
        return -1

def convert_dir_from_entry(entry: dict) -> int:
    return convert_dir(entry.get("t", "P"))


def convert_games_to_dataset(input_dir: str, out_path: str, augment: bool = False, shard_size: int | None = None) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "game_*.json")))

    shard_dir: str | None = None
    shard_paths: List[str] = []
    shard_counts: List[int] = []
    shard_idx = 0

    def flush_shard(out_samples: List[Data]) -> None:
        nonlocal shard_idx
        if shard_dir is None or not out_samples:
            return
        shard_name = f"shard_{shard_idx:05d}.pt"
        shard_path = os.path.join(shard_dir, shard_name)
        tmp_path = shard_path + ".tmp"
        torch.save(out_samples, tmp_path)
        os.replace(tmp_path, shard_path)
        shard_paths.append(shard_name)
        shard_counts.append(len(out_samples))
        out_samples.clear()
        shard_idx += 1

    if shard_size is not None and shard_size > 0:
        shard_dir = out_path + ".shards"
        os.makedirs(shard_dir, exist_ok=True)
        # Avoid mixing old shards with new ones.
        for old in glob.glob(os.path.join(shard_dir, "shard_*.pt")):
            try:
                os.remove(old)
            except OSError:
                pass
        for old_tmp in glob.glob(os.path.join(shard_dir, "shard_*.pt.tmp")):
            try:
                os.remove(old_tmp)
            except OSError:
                pass

    out_samples: List[Data] = []
    next_state_id = 0
    interrupted = False

    transforms = [
        lambda x, y: (x, y),
        lambda x, y: (-x, y),
        lambda x, y: (x, -y),
        lambda x, y: (-x, -y),
        lambda x, y: (y, x),
        lambda x, y: (-y, x),
        lambda x, y: (y, -x),
        lambda x, y: (-y, -x),
    ]

    try:
        for p in paths:
            with open(p, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            moves = payload.get("moves", [])
            policy_targets = payload.get("policy_targets", [])

            game = Game()
            n_policy_steps = len(policy_targets)
            n_total_moves = len(moves)
            n_initial = n_total_moves - n_policy_steps

            for j in range(n_initial):
                mv = moves[j]
                if mv.get("t") == "P":
                    mobj = PASS
                else:
                    mobj = Move(mv["x"], mv["y"], mv["t"])
                if mobj is not PASS:
                    game.add_node_and_neighbours(mobj.c)
                game.do_move(game.current_player, mobj)

            for k in range(n_policy_steps):
                enc = encode_game_to_graph(game)
                pt = policy_targets[k]

                # collect raw visit counts or probs and normalize per decision
                raw = [float(e.get("prob", e.get("visits", 0))) for e in pt]
                s = sum(raw)
                if s <= 0:
                    norm = [1.0 / max(1, len(raw)) for _ in raw]
                else:
                    norm = [r / s for r in raw]

                winner_val = payload.get("winner", None)
                if winner_val is None:
                    v = 0.5
                else:
                    v = 1.0 if winner_val == enc.perspective else 0.0

                base_data = enc.data
                aug_n = 8 if augment else 0
                sel_trans = transforms[:aug_n] if aug_n > 0 else [transforms[0]]

                for tf in sel_trans:
                    state_id = next_state_id
                    next_state_id += 1

                    # Pre-transform the base graph once per symmetry transform;
                    # then clone per-move to attach move-specific labels.
                    base_tf = base_data.clone()
                    x_all = base_tf.x
                    if x_all is not None and x_all.shape[1] >= 2:
                        feats = x_all[:, :-2]
                        coords = x_all[:, -2:]
                        a = coords[:, 0]
                        b = coords[:, 1]
                        tx, ty = tf(a, b)
                        coords_tf = torch.stack((tx, ty), dim=1)
                        base_tf.x = torch.cat((feats, coords_tf), dim=1)

                    if hasattr(base_tf, "node_coords"):
                        nc = base_tf.node_coords
                        txi, tyi = tf(nc[:, 0], nc[:, 1])
                        base_tf.node_coords = torch.stack((txi.to(torch.long), tyi.to(torch.long)), dim=1)

                    for entry, prob in zip(pt, norm):
                        data = base_tf.clone()

                        mf = move_to_feat(entry)
                        is_pass = int(mf[2].item())
                        is_rock = int(mf[3].item())
                        if is_pass:
                            new_x, new_y = mf[0].item(), mf[1].item()
                            new_dir_idx = int(mf[4].item())
                        else:
                            orig_x = int(mf[0].item())
                            orig_y = int(mf[1].item())
                            new_x_f, new_y_f = tf(
                                torch.tensor([orig_x], dtype=torch.float32),
                                torch.tensor([orig_y], dtype=torch.float32),
                            )
                            new_x = int(new_x_f[0].item())
                            new_y = int(new_y_f[0].item())
                            if is_rock:
                                new_dir_idx = int(mf[4].item())
                            else:
                                from models import D, delta_to_direction

                                dir_idx = int(mf[4].item())
                                dir_enum = next((d for d in D if d.as_int == dir_idx), None)
                                if dir_enum is None:
                                    new_dir_idx = dir_idx
                                else:
                                    dx, dy = dir_enum.delta
                                    ndx_t, ndy_t = tf(
                                        torch.tensor([dx], dtype=torch.float32),
                                        torch.tensor([dy], dtype=torch.float32),
                                    )
                                    ndx = int(ndx_t[0].item())
                                    ndy = int(ndy_t[0].item())
                                    nd = delta_to_direction((ndx, ndy))
                                    new_dir_idx = nd.as_int if nd is not None else dir_idx

                        data.move_feat = torch.tensor(
                            [new_x, new_y, float(is_pass), float(is_rock), float(new_dir_idx)],
                            dtype=torch.float32,
                        ).unsqueeze(0)
                        data.y = torch.tensor([prob], dtype=torch.float32)
                        data.value = torch.tensor([v], dtype=torch.float32)
                        data.state_id = torch.tensor([state_id], dtype=torch.long)
                        out_samples.append(data)

                        if shard_dir is not None and shard_size is not None and shard_size > 0 and len(out_samples) >= shard_size:
                            flush_shard(out_samples)

                mv = moves[n_initial + k]
                if mv.get("t") == "P":
                    mobj = PASS
                else:
                    mobj = Move(mv["x"], mv["y"], mv["t"])
                if mobj is not PASS:
                    game.add_node_and_neighbours(mobj.c)
                game.do_move(game.current_player, mobj)
    except KeyboardInterrupt:
        interrupted = True

    if shard_dir is not None and shard_size is not None and shard_size > 0:
        flush_shard(out_samples)
        manifest = {
            "format": "sharded-v1",
            "shard_dir": os.path.basename(shard_dir),
            "shards": shard_paths,
            "counts": shard_counts,
            "total": int(sum(shard_counts)),
            "interrupted": bool(interrupted),
        }
        print(f"Converted {manifest['total']} samples from {len(paths)} games")
        tmp_out = out_path + ".tmp"
        torch.save(manifest, tmp_out)
        os.replace(tmp_out, out_path)
        if interrupted:
            print(f"Interrupted: saved partial sharded dataset manifest to {out_path} (shards in {shard_dir})")
        else:
            print(f"Saved sharded dataset manifest to {out_path} (shards in {shard_dir})")
    else:
        print(f"Converted {len(out_samples)} samples from {len(paths)} games")
        tmp_out = out_path + ".tmp"
        torch.save(out_samples, tmp_out)
        os.replace(tmp_out, out_path)
        print(f"Saved dataset to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MCTS JSON games into training dataset")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/alpha_dataset.pt")
    parser.add_argument("--augment", action="store_true", help="Apply all symmetric augmentations (8 transforms)")
    parser.add_argument("--shard-size", type=int, default=0, help="Write dataset in shards of N samples (reduces peak memory). 0 disables.")
    args = parser.parse_args()
    shard_size = args.shard_size if args.shard_size and args.shard_size > 0 else None
    convert_games_to_dataset(args.input_dir, args.out, args.augment, shard_size=shard_size)


if __name__ == "__main__":
    main()
