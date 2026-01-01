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


def convert_games_to_dataset(input_dir: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    paths = sorted(glob.glob(os.path.join(input_dir, "game_*.json")))
    out_samples: List[Data] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        moves = payload.get("moves", [])
        policy_targets = payload.get("policy_targets", [])

        # Replay game: apply moves sequentially; initial moves include randomize_start
        game = Game()
        # Apply all moves; for each move within the policy_targets list, record state BEFORE applying it
        # policy_targets length should match number of decision moves recorded (excluding initial randomize_start)
        # We assume policy_targets correspond to suffix of moves (after initial randomize_start moves)
        # Determine how many initial moves were recorded by checking lengths
        n_policy_steps = len(policy_targets)
        n_total_moves = len(moves)
        n_initial = n_total_moves - n_policy_steps

        # Apply initial moves first
        for j in range(n_initial):
            mv = moves[j]
            # construct Move-like object via Game.do_move expects Player; simplest is to use Game.players[game.current_player]
            # We create a Move dict and call game.do_move using Move object from models.Move
            if mv.get("t") == "P":
                mobj = PASS
            else:
                mobj = Move(mv["x"], mv["y"], mv["t"])
            # ensure the involved node exists (replay safety)
            if mobj is not PASS:
                game.add_node_and_neighbours(mobj.c)
            game.do_move(game.players[game.current_player], mobj)

        for k in range(n_policy_steps):
            enc = encode_game_to_graph(game)
            pt = policy_targets[k]
            for entry in pt:
                prob = float(entry.get("prob", entry.get("visits", 0)))
                data = enc.data.clone()
                # make move_feat 2D (1, move_feat_dim) so DataLoader batches into (batch, move_feat_dim)
                data.move_feat = move_to_feat(entry).unsqueeze(0)
                data.y = torch.tensor([prob], dtype=torch.float32)
                winner_val = payload.get("winner", None)
                if winner_val is None:
                    v = 0.5
                else:
                    v = 1.0 if winner_val == enc.perspective else 0.0
                data.value = torch.tensor([v], dtype=torch.float32)
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
    args = parser.parse_args()
    convert_games_to_dataset(args.input_dir, args.out)


if __name__ == "__main__":
    main()
