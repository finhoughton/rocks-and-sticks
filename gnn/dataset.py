import json
import math
import os
import random
from glob import glob

from game import Game
from gnn.encode import EncodedGraph, encode_game_to_graph
from models import PASS, Move
from players import Player, RandomPlayer

# Try fast C++ heuristic eval; fall back to slow Python version
try:
    import players_ext
    _HAS_CPP_HEVAL = True
except ImportError:
    _HAS_CPP_HEVAL = False

Sample = tuple[EncodedGraph, float, float]


def _compute_soft_label_cpp(cpp_state: "players_ext.GameState", perspective: int,
                            temperature: float = 3.0) -> float:
    """Fast C++ heuristic eval → soft label probability."""
    score = players_ext.heval_evaluate(cpp_state, perspective)
    return players_ext.heval_score_to_prob(score, temperature)


def _compute_soft_label(game: Game, perspective: int, temperature: float = 3.0) -> float:
    """Compute a soft label from the AB handcrafted evaluation (slow Python fallback).

    Returns a probability in [0, 1] representing how good the position is
    for the given perspective player, using sigmoid(score / temperature).
    """
    if game.winner is not None:
        return 1.0 if game.winner == perspective else 0.0
    from players.ai import AIPlayer
    evaluator = AIPlayer(perspective, use_gnn_eval=False)
    player = game.players[perspective]
    score = evaluator._evaluate_position_handcrafted(game, player)
    logit = score / temperature
    logit = max(-20.0, min(20.0, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def load_balanced_saved_game_samples(
    ab_dir: str,
    mcts_dir: str,
    human_dir: str,
    gamma: float = 0.9,
    balance_classes: bool = False,
    balance_strategy: str = "upsample",
    balance_seed: int | None = None,
    soft_labels: bool = False,
    soft_label_blend: float = 0.7,
    soft_label_temperature: float = 3.0,
    extra_dirs: list[str] | None = None,
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

            # Keep a C++ GameState in sync for fast soft-label computation
            cpp_state = players_ext.GameState() if (soft_labels and _HAS_CPP_HEVAL) else None
            from players.move_utils import to_cpp_move

            trajectory: list[tuple[EncodedGraph, float | None]] = []
            for mv_dict in moves_raw:
                enc = encode_game_to_graph(game)
                # Compute soft label now while game state is at position i
                soft_val = None
                if soft_labels and game.winner is None:
                    if cpp_state is not None:
                        soft_val = _compute_soft_label_cpp(cpp_state, enc.perspective, soft_label_temperature)
                    else:
                        soft_val = _compute_soft_label(game, enc.perspective, soft_label_temperature)
                elif soft_labels and game.winner is not None:
                    soft_val = 1.0 if game.winner == enc.perspective else 0.0
                trajectory.append((enc, soft_val))
                mv = Move(int(mv_dict["x"]), int(mv_dict["y"]), str(mv_dict["t"])) if mv_dict["t"] != "P" else PASS
                try:
                    player = game.current_player
                    game.do_move(player, mv)
                    if cpp_state is not None:
                        cpp_mv = to_cpp_move(mv)
                        cpp_state.do_move(cpp_mv, player)
                except Exception:
                    os.remove(path)
                    print(f"Removed corrupted game file: {path}")
                    break
                if game.winner is not None:
                    break
            n = len(trajectory)
            for i, (enc, soft_val) in enumerate(trajectory):
                outcome_label = 0.5 if winner is None else float(winner == enc.perspective)
                if soft_labels and soft_val is not None:
                    label = (1 - soft_label_blend) * outcome_label + soft_label_blend * soft_val
                else:
                    label = outcome_label
                weight = gamma ** (n - i - 1)
                samples.append((enc, label, weight))
        return samples

    ab_samples = load_samples_from_dir(ab_dir)
    mcts_samples = load_samples_from_dir(mcts_dir)
    human_samples = load_samples_from_dir(human_dir)

    # Load extra directories (e.g. ab-vs-ab)
    extra_samples: list[Sample] = []
    for d in (extra_dirs or []):
        if os.path.isdir(d):
            es = load_samples_from_dir(d)
            print(f"  Extra dir '{d}': {len(es)} samples")
            extra_samples.extend(es)

    # Use ALL samples from every source (no truncation).
    # Weight human games higher since they're scarce but high-quality.
    human_weight = 3
    combined = ab_samples + mcts_samples + human_samples * human_weight + extra_samples
    print(f"  Dataset: {len(ab_samples)} AB + {len(mcts_samples)} MCTS + {len(human_samples)}×{human_weight} human + {len(extra_samples)} extra = {len(combined)} total samples")

    # Optional class balancing: positive (label>0.5) vs negative (label<0.5)
    if balance_classes:
        if balance_seed is not None:
            random.seed(balance_seed)

        pos = [s for s in combined if s[1] > 0.5]
        neg = [s for s in combined if s[1] < 0.5]
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
