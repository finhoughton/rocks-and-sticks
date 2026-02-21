from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List

import torch

from game import Game
from gnn.encode import encode_game_to_graph
from models import PASS, D, Move, calculate_end


def move_to_feat(m: dict) -> torch.Tensor:
    """Create move features for the policy head.

        Layout (16 dims):
            [x, y, is_pass, is_rock, end_x, end_y, dx, dy, dir_onehot(8)]

    Notes:
      - For rocks/pass: (dx,dy) = (0,0), end=(x,y), onehot=0s.
      - For sticks: end is computed from (x,y) and direction.
    """

    x = float(m.get("x", 0.0))
    y = float(m.get("y", 0.0))
    t = str(m.get("t", "P"))
    is_pass = 1.0 if t == "P" else 0.0
    is_rock = 1.0 if t == "R" else 0.0

    dir_idx = convert_dir(t)
    if (is_pass > 0.0) or (is_rock > 0.0) or (dir_idx < 0):
        dx = 0.0
        dy = 0.0
        end_x = x
        end_y = y
        onehot = [0.0] * 8
    else:
        d = next((d for d in D if d.as_int == dir_idx), None)
        if d is None:
            dx = 0.0
            dy = 0.0
            end_x = x
            end_y = y
            onehot = [0.0] * 8
        else:
            dx = float(d.delta[0])
            dy = float(d.delta[1])
            ex, ey = calculate_end((int(x), int(y)), d)
            end_x = float(ex)
            end_y = float(ey)
            onehot = [0.0] * 8
            if 0 <= int(dir_idx) < 8:
                onehot[int(dir_idx)] = 1.0

    return torch.tensor(
        [x, y, is_pass, is_rock, end_x, end_y, dx, dy, *onehot],
        dtype=torch.float32,
    )

def convert_dir(input_dir: str) -> int:
    try:
        return D[input_dir].as_int
    except Exception:
        return -1

def convert_dir_from_entry(entry: dict) -> int:
    return convert_dir(entry.get("t", "P"))


def convert_games_to_dataset(
    input_dir: str,
    out_path: str,
    augment: bool = False,
    shard_size: int | None = None,
    policy_topk: int | None = None,
    policy_min_prob: float = 0.0,
    grouped_policy: bool = False,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "game_*.json")))

    shard_dir: str | None = None
    shard_paths: List[str] = []
    shard_counts: List[int] = []
    shard_idx = 0

    def flush_shard(out_samples: List[object]) -> None:
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

    out_samples: List[object] = []
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

    def _emit_samples_for_policy_target(game: Game, pt: list[dict], payload: dict) -> None:
        nonlocal next_state_id

        enc = encode_game_to_graph(game)

        # collect raw visit counts or probs and normalize per decision
        raw = [float(e.get("prob", e.get("visits", 0))) for e in pt]
        s = sum(raw)
        if s <= 0:
            probs = [1.0 / max(1, len(raw)) for _ in raw]
        else:
            probs = [r / s for r in raw]

        # Optional dataset size reduction: keep only top-K moves (and/or those above
        # a minimum probability). This can drastically reduce the number of samples
        # because each move duplicates the same graph encoding.
        items = list(zip(pt, probs))
        topk = int(policy_topk) if policy_topk is not None else 0
        if topk > 0 and len(items) > topk:
            items.sort(key=lambda x: float(x[1]), reverse=True)
            items = items[:topk]
        min_p = float(policy_min_prob) if policy_min_prob is not None else 0.0
        if min_p > 0.0 and items:
            kept = [(e, p) for (e, p) in items if float(p) >= min_p]
            if kept:
                items = kept

        # Renormalize after filtering.
        psum = float(sum(float(p) for _, p in items))
        if psum <= 0.0:
            norm_items = [(e, 1.0 / max(1, len(items))) for e, _ in items]
        else:
            norm_items = [(e, float(p) / psum) for e, p in items]

        winner_val = payload.get("winner", None)
        if winner_val is None:
            v = 0.5
        else:
            v = 1.0 if winner_val == enc.perspective else 0.0

        base_data = enc.data
        aug_n = 8 if augment else 0
        sel_trans = transforms[:aug_n] if aug_n > 0 else [transforms[0]]

        def _transform_entry(entry: dict, prob: float, tf) -> tuple[torch.Tensor, float]:
            """Apply the same symmetry transform used for the board to a move feature row."""

            from models import delta_to_direction

            mf = move_to_feat(entry)

            # Layout: [x, y, is_pass, is_rock, end_x, end_y, dx, dy, onehot(8)]
            is_pass = int(mf[2].item())
            is_rock = int(mf[3].item())

            orig_x = int(mf[0].item())
            orig_y = int(mf[1].item())
            orig_ex = int(mf[4].item())
            orig_ey = int(mf[5].item())

            if is_pass:
                # Pass is invariant under symmetries.
                new_x, new_y = orig_x, orig_y
                new_ex, new_ey = orig_ex, orig_ey
                new_dx, new_dy = 0, 0
                onehot = [0.0] * 8
            else:
                # Transform start coordinate.
                new_x_f, new_y_f = tf(
                    torch.tensor([orig_x], dtype=torch.float32),
                    torch.tensor([orig_y], dtype=torch.float32),
                )
                new_x = int(new_x_f[0].item())
                new_y = int(new_y_f[0].item())

                # Rock moves have no direction/endpoint.
                if is_rock:
                    new_ex, new_ey = new_x, new_y
                    new_dx, new_dy = 0, 0
                    onehot = [0.0] * 8
                else:
                    # Transform end coordinate too, then infer new direction from the transformed delta.
                    new_ex_f, new_ey_f = tf(
                        torch.tensor([orig_ex], dtype=torch.float32),
                        torch.tensor([orig_ey], dtype=torch.float32),
                    )
                    new_ex = int(new_ex_f[0].item())
                    new_ey = int(new_ey_f[0].item())
                    new_dx = int(new_ex - new_x)
                    new_dy = int(new_ey - new_y)
                    nd = delta_to_direction((new_dx, new_dy))
                    onehot = [0.0] * 8
                    if nd is not None and 0 <= int(nd.as_int) < 8:
                        onehot[int(nd.as_int)] = 1.0

            feat = torch.tensor(
                [
                    float(new_x),
                    float(new_y),
                    float(is_pass),
                    float(is_rock),
                    float(new_ex),
                    float(new_ey),
                    float(new_dx),
                    float(new_dy),
                    *onehot,
                ],
                dtype=torch.float32,
            )
            return feat, float(prob)

        for tf in sel_trans:
            state_id = next_state_id
            next_state_id += 1

            # Pre-transform the base graph once per symmetry transform.
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

            if grouped_policy:
                mf_list: list[torch.Tensor] = []
                prob_list: list[float] = []
                for entry, prob in norm_items:
                    mf, pr = _transform_entry(entry, float(prob), tf)
                    mf_list.append(mf)
                    prob_list.append(pr)

                if not mf_list:
                    return

                sample = {
                    "graph": base_tf,
                    "move_feat": torch.stack(mf_list, dim=0),
                    "policy": torch.tensor(prob_list, dtype=torch.float32),
                    "value": float(v),
                    "state_id": int(state_id),
                }
                out_samples.append(sample)

                if shard_dir is not None and shard_size is not None and shard_size > 0 and len(out_samples) >= shard_size:
                    flush_shard(out_samples)  # type: ignore[arg-type]
            else:
                # Legacy dense format: one sample per move, grouped by state_id.
                for entry, prob in norm_items:
                    data = base_tf.clone()
                    mf, _ = _transform_entry(entry, float(prob), tf)
                    data.move_feat = mf.unsqueeze(0)
                    data.y = torch.tensor([float(prob)], dtype=torch.float32)
                    data.value = torch.tensor([float(v)], dtype=torch.float32)
                    data.state_id = torch.tensor([int(state_id)], dtype=torch.long)
                    out_samples.append(data)

                    if shard_dir is not None and shard_size is not None and shard_size > 0 and len(out_samples) >= shard_size:
                        flush_shard(out_samples)  # type: ignore[arg-type]

    try:
        for p in paths:
            with open(p, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            moves = payload.get("moves", [])
            policy_targets = payload.get("policy_targets", [])

            game = Game()
            n_total_moves = len(moves)
            aligned_policy = (
                isinstance(policy_targets, list)
                and len(policy_targets) == n_total_moves
                and (n_total_moves == 0 or any((pt is None) or isinstance(pt, list) for pt in policy_targets))
            )

            if aligned_policy:
                for j in range(n_total_moves):
                    pt = policy_targets[j]
                    if isinstance(pt, list) and pt:
                        _emit_samples_for_policy_target(game, pt, payload)

                    mv = moves[j]
                    mover = mv.get("p", None)
                    if mv.get("t") == "P":
                        mobj = PASS
                    else:
                        mobj = Move(mv["x"], mv["y"], mv["t"])
                    if mobj is not PASS:
                        game.add_node_and_neighbours(mobj.c)
                    game.do_move(int(mover) if mover is not None else game.current_player, mobj)
            else:
                # Legacy format: policy_targets is dense for the final N moves.
                n_policy_steps = len(policy_targets)
                n_initial = n_total_moves - n_policy_steps

                for j in range(n_initial):
                    mv = moves[j]
                    mover = mv.get("p", None)
                    if mv.get("t") == "P":
                        mobj = PASS
                    else:
                        mobj = Move(mv["x"], mv["y"], mv["t"])
                    if mobj is not PASS:
                        game.add_node_and_neighbours(mobj.c)
                    game.do_move(int(mover) if mover is not None else game.current_player, mobj)

                for k in range(n_policy_steps):
                    pt = policy_targets[k]
                    if isinstance(pt, list) and pt:
                        _emit_samples_for_policy_target(game, pt, payload)

                    mv = moves[n_initial + k]
                    mover = mv.get("p", None)
                    if mv.get("t") == "P":
                        mobj = PASS
                    else:
                        mobj = Move(mv["x"], mv["y"], mv["t"])
                    if mobj is not PASS:
                        game.add_node_and_neighbours(mobj.c)
                    game.do_move(int(mover) if mover is not None else game.current_player, mobj)
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
    parser.add_argument("--policy-topk", type=int, default=0, help="If >0, keep only the top-K policy targets per decision (reduces dataset size).")
    parser.add_argument("--policy-min-prob", type=float, default=0.0, help="If >0, drop policy targets below this probability (after normalization).")
    parser.add_argument("--grouped-policy", action="store_true", help="If set, store one graph per decision with K moves (faster training; avoids duplicating graphs).")
    args = parser.parse_args()
    shard_size = args.shard_size if args.shard_size and args.shard_size > 0 else None
    topk = int(args.policy_topk) if int(args.policy_topk) > 0 else None
    convert_games_to_dataset(
        args.input_dir,
        args.out,
        args.augment,
        shard_size=shard_size,
        policy_topk=topk,
        policy_min_prob=float(args.policy_min_prob),
        grouped_policy=bool(args.grouped_policy),
    )


if __name__ == "__main__":
    main()
