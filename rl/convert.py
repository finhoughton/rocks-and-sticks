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


def convert_games_to_dataset(input_dir: str, out_path: str, augment: bool = False) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "game_*.json")))
    out_samples: List[Data] = []
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
            game.do_move(game.players[game.current_player], mobj)

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

            aug_n = 8 if augment else 0
            sel_trans = transforms[:aug_n] if aug_n > 0 else [transforms[0]]

            for tf in sel_trans:
                # use a stable id for this transformed state
                state_id = len(out_samples)
                # emit one Data per legal move with the same state_id
                for entry, prob in zip(pt, norm):
                    data = base_data.clone()
                    # transform node coords embedded in data.x (last two cols)
                    x_all = data.x
                    if x_all is not None and x_all.shape[1] >= 2:
                        feats = x_all[:, :-2]
                        coords = x_all[:, -2:]
                        a = coords[:, 0]
                        b = coords[:, 1]
                        tx, ty = tf(a, b)
                        coords_tf = torch.stack((tx, ty), dim=1)
                        data.x = torch.cat((feats, coords_tf), dim=1)
                    # transform node_coords field if present
                    if hasattr(data, "node_coords"):
                        nc = data.node_coords
                        txi, tyi = tf(nc[:, 0], nc[:, 1])
                        data.node_coords = torch.stack((txi.to(torch.long), tyi.to(torch.long)), dim=1)

                    mf = move_to_feat(entry)
                    is_pass = int(mf[2].item())
                    is_rock = int(mf[3].item())
                    if is_pass:
                        new_x, new_y = mf[0].item(), mf[1].item()
                        new_dir_idx = int(mf[4].item())
                    else:
                        orig_x = int(mf[0].item())
                        orig_y = int(mf[1].item())
                        new_x_f, new_y_f = tf(torch.tensor([orig_x], dtype=torch.float32), torch.tensor([orig_y], dtype=torch.float32))
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
                                ndx_t, ndy_t = tf(torch.tensor([dx], dtype=torch.float32), torch.tensor([dy], dtype=torch.float32))
                                ndx = int(ndx_t[0].item())
                                ndy = int(ndy_t[0].item())
                                nd = delta_to_direction((ndx, ndy))
                                new_dir_idx = nd.as_int if nd is not None else dir_idx

                    data.move_feat = torch.tensor([new_x, new_y, float(is_pass), float(is_rock), float(new_dir_idx)], dtype=torch.float32).unsqueeze(0)
                    data.y = torch.tensor([prob], dtype=torch.float32)
                    data.value = torch.tensor([v], dtype=torch.float32)
                    data.state_id = torch.tensor([state_id], dtype=torch.long)
                    out_samples.append(data)

            mv = moves[n_initial + k]
            if mv.get("t") == "P":
                mobj = PASS
            else:
                mobj = Move(mv["x"], mv["y"], mv["t"])
            # ensure the involved node exists (replay safety)
            if mobj is not PASS:
                game.add_node_and_neighbours(mobj.c)
            game.do_move(game.players[game.current_player], mobj)

    print(f"Converted {len(out_samples)} samples from {len(paths)} games")
    torch.save(out_samples, out_path)
    print(f"Saved dataset to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MCTS JSON games into training dataset")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="data/alpha_dataset.pt")
    parser.add_argument("--augment", action="store_true", help="Apply all symmetric augmentations (8 transforms)")
    args = parser.parse_args()
    convert_games_to_dataset(args.input_dir, args.out, args.augment)


if __name__ == "__main__":
    main()
