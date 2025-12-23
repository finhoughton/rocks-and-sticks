from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Callable, Iterable

import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import DataLoader  # type: ignore

from game import Game
from gnn_eval import EncodedGraph, GNNEval, encode_game_to_graph
from models import D, Move, move_sort_key
from players import (
    AIPlayer,
    AlphaBetaPlayer,
    MCTSPlayer,
    Player,
    RandomPlayer,
    applied_move,
)

"""Self-play training loop for the GNN evaluator.

This script:
- runs self-play games to generate labeled graph samples
- trains the GNNEval model on win/loss labels for the side to move
- saves the trained weights for use via gnn_eval.load_model

Usage example:
python gnn_train_selfplay.py --games 100 --epochs 5 --batch-size 64 --lr 1e-3 --max-moves 180 --use-biased-random --augment-sym
"""

def play_self_play_game(
    player_factories: Iterable[Callable[[int], Player]],
    max_moves: int = 256,
) -> tuple[list[EncodedGraph], list[Move], int | None]:
    players = [factory(i) for i, factory in enumerate(player_factories)]
    game = Game(players)
    trajectory: list[EncodedGraph] = []
    move_log: list[Move] = []

    _randomize_start(game, move_log=move_log)

    while game.winner is None and len(game.moves) < max_moves:
        player = game.players[game.current_player]
        trajectory.append(encode_game_to_graph(game))
        mv = player.get_move(game)
        move_log.append(mv)
        game.do_move(player, mv)

    return trajectory, move_log, game.winner


def _randomize_start(
    game: Game,
    max_sticks: int = 6,
    max_rocks: int = 1,
    rollout_moves: int = 6,
    move_log: list[Move] | None = None,
) -> None:
    """Create a shallow, non-winning random start to diversify training."""
    rng = random.Random()
    player = game.players[game.current_player]

    # Place a few non-scoring sticks.
    target_sticks = rng.randint(1, max_sticks)
    attempts = 0
    while target_sticks > 0 and attempts < 60:
        attempts += 1
        moves = [m for m in game.get_possible_moves(player) if m.t in D.__members__]
        if not moves:
            break
        rng.shuffle(moves)
        mv = moves[0]
        before = game.players_scores[player.number]
        game.do_move(player, mv)
        gained = game.players_scores[player.number] - before
        if gained == 0 and game.winner is None:
            target_sticks -= 1
            if move_log is not None:
                move_log.append(mv)
            continue
        game.undo_move()

    # Optionally place a rock.
    if max_rocks > 0:
        rock_moves = [m for m in game.get_possible_moves(player) if m.t == "R"]
        if rock_moves and rng.random() < 0.4:
            mv = rng.choice(rock_moves)
            game.do_move(player, mv)
            if game.winner is not None:
                game.undo_move()
            else:
                if move_log is not None:
                    move_log.append(mv)

    # Short random rollout to reach mid-game-ish states.
    for _ in range(rollout_moves):
        mover = game.players[game.current_player]
        moves = list(game.get_possible_moves(mover))
        if not moves:
            break
        rng.shuffle(moves)
        mv = moves[0]
        game.do_move(mover, mv)
        undo_needed = False
        if game.winner is not None and game.winner != mover.number:
            undo_needed = True
        elif game.winner is not None:
            undo_needed = True

        if undo_needed:
            game.undo_move()
        else:
            if move_log is not None:
                move_log.append(mv)
        if undo_needed:
            break

    # ensure we start from the human (player 0) perspective each game
    game.current_player = 0

def _augment_symmetries(enc: EncodedGraph) -> list[Data]:
    data = enc.data
    x = data.x
    assert x is not None, "data.x must not be None"
    feats = x[:, :-2]
    coords = x[:, -2:]

    transforms: list[Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = [
        lambda a, b: (a, b),   # identity
        lambda a, b: (-a, b),  # reflect x
        lambda a, b: (a, -b),  # reflect y
        lambda a, b: (-a, -b), # rotate 180
        lambda a, b: (b, a),   # transpose
        lambda a, b: (-b, a),  # rot 90
        lambda a, b: (b, -a),  # rot -90
        lambda a, b: (-b, -a), # transpose+reflect
    ]

    out: list[Data] = []
    for tf in transforms:
        a, b = coords[:, 0], coords[:, 1] # type: ignore
        tx, ty = tf(a, b) # type: ignore
        coords_tf = torch.stack((tx, ty), dim=1)
        x_tf = torch.cat((feats, coords_tf), dim=1) # type: ignore
        out.append(
            Data(
                x=x_tf,
                edge_index=data.edge_index,
                batch=data.batch,
                global_feats=data.global_feats,
            )
        )
    return out


class BiasedRandomPlayer(RandomPlayer):
    """Random player with a slight bias toward rock moves to diversify play."""

    def get_move(self, game: Game) -> Move:
        moves = sorted(game.get_possible_moves(self), key=move_sort_key)
        rock_moves = [m for m in moves if m.t == "R"]
        if rock_moves and self._rng.random() < 0.7:
            return self._rng.choice(rock_moves)
        return self._rng.choice(moves)


class GreedyHeuristicPlayer(AIPlayer):
    """1-ply greedy using the handcrafted heuristic for speed."""

    def get_move(self, game: Game) -> Move:
        self._clear_search_caches()
        moves = sorted(game.get_possible_moves(self), key=move_sort_key)
        best = moves[0]
        best_v = float("-inf")
        for mv in moves:
            with applied_move(game, self, mv):
                v = self._evaluate_position_handcrafted(game, self)
            if v > best_v:
                best_v = v
                best = mv
        return best


class DepthLimitedAlphaBetaPlayer(AlphaBetaPlayer):
    """Alpha-beta player with a configurable depth (for data generation)."""

    def __init__(self, player_number: int, depth: int = 2):
        super().__init__(player_number)
        self.depth = depth

    def get_move(self, game: Game) -> Move:
        if game.num_players != 2:
            raise ValueError("Alpha-beta AI is only implemented for 2 players")
        self._clear_search_caches()
        move, _ = self.alpha_beta(game, self.depth, float("-inf"), float("inf"), True)
        return move


def build_dataset(
    num_games: int,
    player_factories: Iterable[Callable[[int], Player]],
    max_moves: int = 256,
    augment_sym: bool = True,
    swap_roles: bool = False,
    save_games_dir: str | None = None,
) -> list[Data]:
    samples: list[Data] = []
    pf = list(player_factories)
    if save_games_dir:
        os.makedirs(save_games_dir, exist_ok=True)
    for i in range(num_games):
        print(f"Generating game {i+1}/{num_games}...")
        this_pf = pf if (not swap_roles or (i % 2 == 0)) else list(reversed(pf))
        traj, moves, winner = play_self_play_game(this_pf, max_moves=max_moves)

        if save_games_dir:
            game_path = os.path.join(save_games_dir, f"game_{i:05d}.json")
            payload: dict[str, object] = {
                "winner": winner,
                "moves": [
                    {"x": m.c[0], "y": m.c[1], "t": m.t}
                    for m in moves
                ],
                "max_moves_reached": (len(moves) >= max_moves and winner is None),
            }
            with open(game_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)

        for enc in traj:
            label = 0.5 if winner is None else float(winner == enc.perspective)
            datas = _augment_symmetries(enc) if augment_sym else [enc.data]
            for data in datas:
                data.y = torch.tensor([label], dtype=torch.float32)
                samples.append(data)
    return samples


def train(dataset: list[Data], epochs: int = 5, batch_size: int = 16, lr: float = 1e-3, device: str = "cpu") -> GNNEval:
    if not dataset:
        raise ValueError("Dataset is empty; generate samples first.")
    device_t = torch.device(device)
    sample = dataset[0]
    node_feat_dim = sample.x.size(1) # type: ignore
    global_feat_dim = sample.global_feats.size(1)
    model = GNNEval(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim).to(device_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device_t)
            logits = model(batch)
            loss = criterion(logits, batch.y.view_as(logits))
            opt.zero_grad()
            loss.backward()
            opt.step() # type: ignore
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(loader))
        print(f"epoch {epoch+1}/{epochs} loss={avg_loss:.4f}")

    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GNN eval via self-play")
    parser.add_argument("--games", type=int, default=20, help="number of self-play games")
    parser.add_argument("--epochs", type=int, default=5, help="training epochs")
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
        action="store_true",
        help="augment each sample with 8 board symmetries (rotations/reflections)",
    )
    parser.add_argument(
        "--use-biased-random",
        action="store_true",
        help="use a stick-biased random player mix to diversify data",
    )
    parser.add_argument(
        "--greedy-frac",
        type=float,
        default=0.15,
        help="probability a player is greedy-heuristic instead of random (0..1)",
    )
    parser.add_argument(
        "--ab-vs-random",
        action="store_true",
        help="generate games with depth-limited AlphaBeta vs random/biased random opponents",
    )
    parser.add_argument(
        "--ab-depth",
        type=int,
        default=2,
        help="search depth for the AlphaBeta player when --ab-vs-random is used",
    )
    parser.add_argument(
        "--swap-roles",
        action="store_true",
        help="when using asymmetric players, alternate which side uses AlphaBeta each game",
    )
    parser.add_argument(
        "--save-games-dir",
        type=str,
        default=None,
        help="optional directory to dump generated games as JSON",
    )
    args = parser.parse_args()

    seed_base = args.seed_base
    if seed_base is None:
        seed_base = random.Random().randrange(1_000_000_000) ^ int(time.time() * 1e6)
    seeds = [seed_base + i for i in range(2)]

    def randomish_factory(idx: int) -> Player:
        # Default to all-random for fast data generation; optionally use MCTS, biased random, or greedy heuristic.
        seed = seeds[idx % len(seeds)]
        r = random.random()
        if args.use_mcts and idx % 2 == 0:
            return MCTSPlayer(idx, time_limit=0.10, n_rollouts=120, max_sim_depth=30, seed=seed)
        if args.greedy_frac > 0.0 and r < args.greedy_frac:
            return GreedyHeuristicPlayer(idx)
        if args.use_biased_random and (idx % 2 == 1 or random.random() < 0.5):
            return BiasedRandomPlayer(idx, seed=seed)
        return RandomPlayer(idx, seed=seed)

    if args.ab_vs_random:
        def ab_factory(idx: int):
            return DepthLimitedAlphaBetaPlayer(idx, depth=args.ab_depth)
        player_factories: list[Callable[[int], Player]] = [ab_factory, randomish_factory]
    else:
        player_factories = [randomish_factory, randomish_factory]

    dataset = build_dataset(
        args.games,
        player_factories,
        max_moves=args.max_moves,
        augment_sym=args.augment_sym,
        swap_roles=args.swap_roles,
        save_games_dir=args.save_games_dir,
    )
    model = train(dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
    torch.save(model.state_dict(), args.out)
    print(f"saved weights to {args.out}")


if __name__ == "__main__":
    main()
