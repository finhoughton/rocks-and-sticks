from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from game import Game
from gnn.encode import SAMPLE_ENC
from players import AlphaBetaPlayer, MCTSPlayer, Player, RandomPlayer
from rl.PPO import PPOGNNPolicy, PPOPlayer


@dataclass(frozen=True)
class EvalConfig:
    games: int = 50
    max_turns: int = 200
    mcts_rollouts: int = 400
    mcts_sim_depth: int = 60
    mcts_time_limit: float | None = None
    mcts0_seed: int | None = 0
    mcts1_seed: int | None = 1
    random_seed: int | None = 1
    show: bool = False
    pause_s: float = 0.0
    mode: str = "mcts-vs-random"  # left-vs-right token, swapping alternates sides between games
    profile: bool = False
    profile_top: int = 40
    profile_out: str | None = None
    gnn_model: str | None = None
    device: str = "cpu"
    policy_checkpoint: str | None = None
    policy_device: str = "cpu"


def _load_gnn(model_path: str, device: str) -> None:
    """Load GNN weights once and enable GNN eval for AI players."""
    from gnn.model import load_model

    node_dim = SAMPLE_ENC.data.x.size(1) # type: ignore
    global_dim = SAMPLE_ENC.data.global_feats.size(1)
    load_model(model_path, node_dim, global_dim, device=device)


def _load_policy(checkpoint: str, device: str) -> PPOGNNPolicy:
    node_dim = SAMPLE_ENC.data.x.size(1)  # type: ignore
    global_dim = SAMPLE_ENC.data.global_feats.size(1)
    policy = PPOGNNPolicy(node_feat_dim=node_dim, global_feat_dim=global_dim)
    policy.to(device)
    sd = torch.load(checkpoint, map_location=device)
    try:
        policy.load_state_dict(sd)
    except Exception:
        model_keys = set(policy.state_dict().keys())
        filtered = {k: v for k, v in sd.items() if k in model_keys}
        policy.load_state_dict(filtered, strict=False)
    policy.eval()
    return policy


def play_one_game(cfg: EvalConfig, game_index: int, policy_model: Optional[nn.Module] = None, swap: bool = False) -> Optional[int]:
    # Keep seeds deterministic but varied per game.

    # Parse left/right tokens from mode
    try:
        left_tok, right_tok = cfg.mode.split("-vs-")
    except Exception:
        raise ValueError(f"Unrecognized mode format: {cfg.mode}")

    def make_player(token: str, slot_index: int) -> Player:
        # slot_index is the numeric player id in the game (0 or 1)
        if token == "mcts":
            base_seed: Optional[int] = cfg.mcts0_seed if slot_index == 0 else cfg.mcts1_seed
            seed = None if base_seed is None else base_seed + game_index
            return MCTSPlayer(
                slot_index,
                seed=seed,
                n_rollouts=cfg.mcts_rollouts,
                max_sim_depth=cfg.mcts_sim_depth,
                time_limit=cfg.mcts_time_limit,
            )
        if token == "random":
            return RandomPlayer(slot_index, seed=None if cfg.random_seed is None else cfg.random_seed + game_index)
        if token == "alphabeta":
            return AlphaBetaPlayer(slot_index)
        if token == "policy":
            if policy_model is None:
                raise ValueError("Policy player requested but no policy checkpoint was provided.")
            return PPOPlayer(slot_index, policy_model, cfg.device)
        raise ValueError(f"Unknown player token: {token}")

    left_player: Player = make_player(left_tok, 0)
    right_player: Player = make_player(right_tok, 1)

    # If swap requested, assign players to slots reversed (but use slot indices 0 and 1)
    if swap:
        p0 = make_player(right_tok, 0)
        p1 = make_player(left_tok, 1)
    else:
        p0 = left_player
        p1 = right_player

    g = Game(players=[p0, p1])

    plt = None
    show = cfg.show
    if show:
        import matplotlib.pyplot as plt  # type: ignore

        plt.ion()  # type: ignore[reportUnknownMemberType]
        try:
            g.render(block=False, game_index=game_index, total_games=cfg.games)
        except SystemError:
            # Matplotlib interactive backends can occasionally error on macOS.
            show = False

    while g.winner is None and g.turn_number < cfg.max_turns:
        for player in g.players:
            move = player.get_move(g)
            g.do_move(player, move)
            if show:
                try:
                    g.render(block=False, game_index=game_index, total_games=cfg.games)
                except SystemError:
                    show = False
                if cfg.pause_s > 0:
                    time.sleep(cfg.pause_s)
            if g.winner is not None:
                break

    # Close the figure so multiple games don't spawn a pile of windows.
    if show and plt is not None:
        try:
            fig = getattr(g, "_render_fig", None)
            if fig is not None:
                plt.close(fig)  # type: ignore[reportPrivateUsage]
            else:
                plt.close("all")
        except Exception:
            pass

    return g.winner


def build_config_from_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate MCTS vs Random")
    parser.add_argument(
        "--mode",
        choices=[
            "mcts-vs-random",
            "mcts-vs-mcts",
            "mcts-vs-alphabeta",
            "alphabeta-vs-alphabeta",
            "policy-vs-alphabeta",
            "policy-vs-mcts",
        ],
        default="mcts-vs-random",
        help="Which matchup to run (left-vs-right). The script alternates sides between games.",
    )
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--rollouts", type=int, default=400)
    parser.add_argument("--sim-depth", type=int, default=60)
    parser.add_argument("--time-limit", type=float, default=None, help="Per-move MCTS time limit in seconds (overrides rollout cap)")
    parser.add_argument("--mcts0-seed", type=int, default=0)
    parser.add_argument("--mcts1-seed", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--show", action="store_true", help="Render games as they play")
    parser.add_argument("--pause", type=float, default=0.0, help="Extra pause between moves (seconds)")
    parser.add_argument("--profile", action="store_true", help="Run under cProfile and print hotspots")
    parser.add_argument("--profile-top", type=int, default=40, help="How many lines of profile output to print")
    parser.add_argument("--profile-out", type=str, default=None, help="Optional path to write a .prof file")
    parser.add_argument("--gnn-model", type=str, default=None, help="Path to GNN weights to enable NN eval")
    parser.add_argument("--device", type=str, default="cpu", help="Device for GNN (cpu/cuda)")
    parser.add_argument("--policy-checkpoint", type=str, default=None, help="Path to PPO policy checkpoint to evaluate")
    parser.add_argument("--policy-device", type=str, default="cpu", help="Device for loaded policy")
    args = parser.parse_args()

    cfg = EvalConfig(
        mode=args.mode,
        games=args.games,
        max_turns=args.max_turns,
        mcts_rollouts=args.rollouts,
        mcts_sim_depth=args.sim_depth,
        mcts_time_limit=args.time_limit,
        mcts0_seed=None if args.mcts0_seed < 0 else args.mcts0_seed,
        mcts1_seed=None if args.mcts1_seed < 0 else args.mcts1_seed,
        random_seed=None if args.random_seed < 0 else args.random_seed,
        show=args.show,
        pause_s=args.pause,
        profile=args.profile,
        profile_top=args.profile_top,
        profile_out=args.profile_out,
        gnn_model=args.gnn_model,
        policy_checkpoint=args.policy_checkpoint,
        policy_device=args.policy_device,
        device=args.device,
    )
    return cfg


def load_models(cfg: EvalConfig) -> Optional[nn.Module]:
    if cfg.gnn_model:
        _load_gnn(cfg.gnn_model, cfg.device)
        print(f"Using GNN evaluator from {cfg.gnn_model} on device {cfg.device}.")
    policy_model: Optional[nn.Module] = None
    if cfg.policy_checkpoint:
        policy_model = _load_policy(cfg.policy_checkpoint, cfg.policy_device)
        print(f"Loaded policy from {cfg.policy_checkpoint} onto device {cfg.policy_device}.")
    return policy_model


def evaluate(cfg: EvalConfig, policy_model: Optional[nn.Module]) -> Dict[str | None, int]:
    wins_lr: Dict[str | None, int] = {"left": 0, "right": 0, None: 0}

    for i in range(cfg.games):
        swap = (i % 2) == 1
        if not cfg.show:
            side_note = "(swapped)" if swap else ""
            print(f"Game {i + 1}/{cfg.games} {side_note} ({cfg.mode})...", flush=True)
        try:
            w = play_one_game(cfg, i, policy_model, swap=swap)
        except KeyboardInterrupt:
            if not cfg.show:
                print(f"Interrupted during game {i + 1}/{cfg.games}.")
            raise
        if w is None:
            wins_lr[None] += 1
        else:
            if not swap:
                wins_lr["left" if w == 0 else "right"] += 1
            else:
                wins_lr["right" if w == 0 else "left"] += 1
        if not cfg.show:
            if w is None:
                w_s = "no winner"
            else:
                w_s = f"player {w + 1}"
            print(f"  Result: {w_s}", flush=True)
    return wins_lr


def main() -> None:
    cfg = build_config_from_args()
    wins_lr: Dict[str | None, int]
    policy_model = load_models(cfg)

    pr: cProfile.Profile | None = None
    try:
        if cfg.profile:
            pr = cProfile.Profile()
            pr.enable()
            try:
                wins_lr = evaluate(cfg, policy_model)
            finally:
                pr.disable()

            if cfg.profile_out:
                pr.dump_stats(cfg.profile_out)
                print(f"Wrote profile to: {cfg.profile_out}")

            stats = pstats.Stats(pr)
            stats.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
            print("\n=== cProfile (top cumulative) ===")
            stats.print_stats(max(1, cfg.profile_top))
        else:
            wins_lr = evaluate(cfg, policy_model)
    except KeyboardInterrupt:
        print("\nInterrupted by user; printing partial results...")
        if cfg.profile and pr is not None:
            try:
                if cfg.profile_out:
                    pr.dump_stats(cfg.profile_out)
                    print(f"Wrote profile to: {cfg.profile_out}")
                stats = pstats.Stats(pr)
                stats.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
                print("\n=== cProfile (top cumulative, partial run) ===")
                stats.print_stats(max(1, cfg.profile_top))
            except Exception:
                pass
        return

    total_played = int(sum(v for k, v in wins_lr.items() if k is not None))
    print(f"Games (decisive): {total_played}")
    if total_played <= 0:
        print(f"No decisive games played. Draws: {wins_lr[None]}")
        return
    # Generic X-vs-Y summary (maps short tokens to display names)
    try:
        left_tok, right_tok = cfg.mode.split("-vs-")
    except Exception:
        left_tok, right_tok = "player1", "player2"
    name_map = {
        "mcts": "MCTS",
        "random": "Random",
        "alphabeta": "AlphaBeta",
        "policy": "PPO",
    }
    left_name = name_map.get(left_tok, left_tok.capitalize())
    right_name = name_map.get(right_tok, right_tok.capitalize())
    left_wins = wins_lr["left"]
    right_wins = wins_lr["right"]
    print(f"{left_name} (left) wins: {left_wins} ({left_wins / total_played:.1%})")
    print(f"{right_name} (right) wins: {right_wins} ({right_wins / total_played:.1%})")
    print(f"No winner (max_turns reached / draws): {wins_lr[None]} ({wins_lr[None] / (total_played + wins_lr[None]):.1%})")


if __name__ == "__main__":
    main()
