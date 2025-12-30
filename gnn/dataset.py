import json
import os
import random
from glob import glob

from game import Game
from gnn.encode import EncodedGraph, encode_game_to_graph
from models import PASS, Move
from players import Player, RandomPlayer

Sample = tuple[EncodedGraph, float, float]

def load_balanced_saved_game_samples(
    ab_dir: str,
    mcts_dir: str,
    human_dir: str,
    gamma: float = 0.9,
    balance_classes: bool = False,
    balance_strategy: str = "upsample",
    balance_seed: int | None = None,
) -> list[Sample]:
    def load_samples_from_dir(d: str) -> list[Sample]:
        samples: list[Sample] = []
        paths = sorted(glob(os.path.join(d, "game_*.json")))
        for path in paths:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            moves_raw = payload.get("moves", [])
            winner = payload.get("winner", None)
            players: list[Player] = [RandomPlayer(0), RandomPlayer(1)]
            game = Game(players)
            trajectory: list[EncodedGraph] = []
            for mv_dict in moves_raw:
                trajectory.append(encode_game_to_graph(game))
                mover = game.players[game.current_player]
                mv = Move(int(mv_dict["x"]), int(mv_dict["y"]), str(mv_dict["t"])) if mv_dict["t"] != "P" else PASS
                game.do_move(mover, mv)
                if game.winner is not None:
                    break
            n = len(trajectory)
            for i, enc in enumerate(trajectory):
                label = 0.5 if winner is None else float(winner == enc.perspective)
                weight = gamma ** (n - i - 1)
                samples.append((enc, label, weight))
        return samples

    ab_samples = load_samples_from_dir(ab_dir)
    mcts_samples = load_samples_from_dir(mcts_dir)
    human_samples = load_samples_from_dir(human_dir)

    n = min(len(ab_samples), len(mcts_samples))
    human_weight = 3
    combined = ab_samples[:n] + mcts_samples[:n] + human_samples * human_weight

    # Optional class balancing: positive (label==1.0) vs negative (label==0.0)
    if balance_classes:
        if balance_seed is not None:
            random.seed(balance_seed)

        pos = [s for s in combined if s[1] == 1.0]
        neg = [s for s in combined if s[1] == 0.0]
        draws = [s for s in combined if s[1] == 0.5]

        # If no pos or no neg, nothing to balance
        if pos and neg:
            if balance_strategy == "upsample":
                target = max(len(pos), len(neg))
                if len(pos) < target:
                    pos = pos + [random.choice(pos) for _ in range(target - len(pos))]
                if len(neg) < target:
                    neg = neg + [random.choice(neg) for _ in range(target - len(neg))]
            elif balance_strategy == "downsample":
                target = min(len(pos), len(neg))
                pos = random.sample(pos, target)
                neg = random.sample(neg, target)
            else:
                raise ValueError(f"Unknown balance_strategy: {balance_strategy}")

        combined = pos + neg + draws

    random.shuffle(combined)
    return combined
