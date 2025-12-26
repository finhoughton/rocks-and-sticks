from __future__ import annotations

import argparse
import random
import time
from typing import Callable

import torch

from gnn.augment import samples_to_data
from gnn.dataset import load_balanced_saved_game_samples
from gnn.encode import EncodedGraph
from gnn.game_generation import generate_self_play_games
from gnn.plotting import plot_losses
from gnn.train import train
from players import (
    AlphaBetaPlayer,
    MCTSPlayer,
    OnePlyGreedyPlayer,
    Player,
    RandomPlayer,
    RockBiasedRandomPlayer,
)

Sample = tuple[EncodedGraph, float, float]

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate games and/or train GNN eval")
    parser.add_argument("--games", type=int, default=None, help="number of self-play games (omit to skip generation)")
    parser.add_argument("--epochs", type=int, default=0, help="training epochs (0 = generate games only, no training)")
    parser.add_argument("--batch-size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--out", type=str, default="gnn_eval.pt", help="output path for weights")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--max-moves", type=int, default=256, help="move cap per game")
    parser.add_argument(
        "--use-mcts",
        action="store_true",
        help="use a slow MCTS+random mix (default: both players random for speed)",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="base seed for random self-play (defaults to a fresh random value each run)",
    )
    parser.add_argument(
        "--augment-sym",
        default=True,
        action="store_true",
        help="augment each sample with 8 board symmetries (rotations/reflections)",
    )
    parser.add_argument(
        "--greedy-frac",
        type=float,
        default=0.15,
        help="probability a player is greedy-heuristic instead of random (0..1)",
    )
    parser.add_argument(
        "--game-type",
        type=str,
        default="ab-vs-random",
        choices=["ab-vs-random", "mcts-vs-random", "mcts-vs-mcts", "greedy-vs-greedy", "random-vs-random"],
        help="game type to generate (ab-vs-random [default], mcts-vs-random, mcts-vs-mcts, greedy-vs-greedy, random-vs-random)",
    )
    parser.add_argument(
        "--ab-depth",
        type=int,
        default=2,
        help="search depth for AlphaBeta when game-type includes AB",
    )
    mcts_group = parser.add_mutually_exclusive_group()
    mcts_group.add_argument(
        "--mcts-time-limit",
        type=float,
        default=None,
        help="time limit (seconds) for MCTS simulations (mutually exclusive with --mcts-rollouts)",
    )
    mcts_group.add_argument(
        "--mcts-rollouts",
        type=int,
        default=None,
        help="number of rollouts for MCTS simulations (mutually exclusive with --mcts-time-limit)",
    )
    parser.add_argument(
        "--swap-roles",
        default=True,
        action="store_true",
        help="when using asymmetric players, alternate which side uses the stronger player each game",
    )
    parser.add_argument(
        "--save-games-dir",
        type=str,
        default=None,
        help="optional directory to dump generated games as JSON",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="fraction of samples to hold out for validation loss reporting",
    )
    parser.add_argument(
        "--plot-loss",
        action="store_true",
        help="save a PNG of train/val loss curves",
    )
    parser.add_argument(
        "--loss-plot-out",
        type=str,
        default="loss_curve.png",
        help="path to save the loss plot when --plot-loss is set",
    )

    args = parser.parse_args()
    seed_base = args.seed_base
    if seed_base is None:
        seed_base = random.Random().randrange(1_000_000_000) ^ int(time.time() * 1e6)
    seeds = [seed_base + i for i in range(2)]

    def randomish_factory(idx: int) -> Player:
        seed = seeds[idx % len(seeds)]
        if random.random() < 0.2:
            return OnePlyGreedyPlayer(idx)
        if random.random() < 0.4:
            return RockBiasedRandomPlayer(idx, seed=seed)
        return RandomPlayer(idx, seed=seed)

    def mcts_factory(idx: int) -> Player:
        seed = seeds[idx % len(seeds)]
        time_limit = args.mcts_time_limit
        n_rollouts = args.mcts_rollouts
        if (time_limit is not None) and (n_rollouts is not None):
            raise ValueError("Specify only one of --mcts-time-limit or --mcts-rollouts")
        return MCTSPlayer(idx, time_limit=time_limit, n_rollouts=n_rollouts if n_rollouts is not None else 1000, max_sim_depth=30, seed=seed, use_gnn=True)

    def greedy_factory(idx: int) -> Player:
        return OnePlyGreedyPlayer(idx)

    def ab_factory(idx: int) -> Player:
        seed = seeds[idx % len(seeds)]
        x = random.random()
        if x < 0.6:
            return AlphaBetaPlayer(idx, depth=args.ab_depth)
        elif x < 0.8:
            return AlphaBetaPlayer(idx, depth=max(1, args.ab_depth - 1))
        return RockBiasedRandomPlayer(idx, seed=seed)

    if args.game_type == "ab-vs-random":
        player_factories: list[Callable[[int], Player]] = [ab_factory, randomish_factory]
    elif args.game_type == "mcts-vs-random":
        player_factories = [mcts_factory, randomish_factory]
    elif args.game_type == "mcts-vs-mcts":
        player_factories = [mcts_factory, mcts_factory]
    elif args.game_type == "greedy-vs-greedy":
        player_factories = [greedy_factory, greedy_factory]
    else:
        player_factories = [randomish_factory, randomish_factory]

    if args.games is not None:
        generate_self_play_games(
            args.games,
            player_factories,
            max_moves=args.max_moves,
            swap_roles=args.swap_roles,
            save_games_dir=args.save_games_dir,
        )
        print(f"Generated {args.games} games")
        train_dataset = []
        val_dataset = []
    else:
        # Only training, not generating new games
        print("Loading balanced dataset from saved_games_ab2, saved_games_mcts, saved_games_human...")
        samples = load_balanced_saved_game_samples("saved_games_ab2", "saved_games_mcts", "saved_games_human")
        idx = list(range(len(samples)))
        random.shuffle(idx)
        split = max(1, int(len(idx) * args.val_frac))
        train_samples = [samples[i] for i in idx[split:]]
        val_samples = [samples[i] for i in idx[:split]]
        train_dataset = samples_to_data(train_samples, augment_sym=args.augment_sym)
        val_dataset = samples_to_data(val_samples, augment_sym=args.augment_sym) if val_samples else []

    if args.epochs > 0:
        if not train_dataset:
            raise ValueError("No training data: either specify --games or provide a dataset to train on.")
        print(f"Training: {len(train_dataset)}; validation: {len(val_dataset)}")
        model, train_losses, val_losses = train(
            train_dataset,
            val_dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        torch.save(model.state_dict(), args.out)
        print(f"saved weights to {args.out}")
        if args.plot_loss:
            plot_losses(train_losses, val_losses, args.loss_plot_out)

if __name__ == "__main__":
    main()

"""
examples usage:

rocks and sicks> python3 -m gnn.gnn_main --games 50 --game-type mcts-vs-random --mcts-rollouts 100 --save-games-dir saved_games_mcts
rocks and sicks> python3 -m gnn.gnn_main --epochs 10 --lr 1e-3 --batch-size 8 --device cpu

"""
