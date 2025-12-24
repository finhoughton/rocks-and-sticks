from __future__ import annotations

import argparse
import json
import os
import random
import time
from glob import glob
from typing import Any, Callable, Iterable, cast

import torch
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import DataLoader

from game import Game
from gnn.gnn_encode import EncodedGraph, encode_game_to_graph
from gnn.gnn_model import GNNEval
from models import PASS, D, Move
from players import (
    AlphaBetaPlayer,
    MCTSPlayer,
    OnePlyGreedyPlayer,
    Player,
    RandomPlayer,
    RockBiasedRandomPlayer,
)

Sample = tuple[EncodedGraph, float]

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
    max_rocks: int = 2,
    rollout_moves: int = 6,
    move_log: list[Move] | None = None,
) -> None:
    player = game.players[game.current_player]
    target_sticks = random.randint(1, max_sticks)
    attempts = 0
    while target_sticks > 0 and attempts < 60:
        attempts += 1
        moves = [m for m in game.get_possible_moves(player) if m.t in D.__members__]
        if not moves:
            break
        mv = random.choice(moves)
        before = game.players_scores[player.number]
        game.do_move(player, mv)
        gained = game.players_scores[player.number] - before
        if gained == 0 and game.winner is None:
            target_sticks -= 1
            if move_log is not None:
                move_log.append(mv)
            continue
        game.undo_move()
    for _ in range(random.randint(0, max_rocks)):
        rock_moves = [m for m in game.get_possible_moves(player) if m.t == "R"]
        if rock_moves and random.random() < 0.7:
            mv = random.choice(rock_moves)
            game.do_move(player, mv)
            if game.winner is not None:
                game.undo_move()
            else:
                if move_log is not None:
                    move_log.append(mv)
    for _ in range(rollout_moves):
        mover = game.players[game.current_player]
        moves = list(game.get_possible_moves(mover))
        if not moves:
            break
        random.shuffle(moves)
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
    game.current_player = 0

def _augment_symmetries(enc: EncodedGraph) -> list[Data]:
    data = enc.data
    x = data.x
    assert x is not None, "data.x must not be None"
    feats = x[:, :-2]
    coords = x[:, -2:]
    transforms: list[Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = [
        lambda a, b: (a, b),
        lambda a, b: (-a, b),
        lambda a, b: (a, -b),
        lambda a, b: (-a, -b),
        lambda a, b: (b, a),
        lambda a, b: (-b, a),
        lambda a, b: (b, -a),
        lambda a, b: (-b, -a),
    ]
    out: list[Data] = []
    for tf in transforms:
        a, b = coords[:, 0], coords[:, 1]
        tx, ty = tf(a, b)
        coords_tf = torch.stack((tx, ty), dim=1)
        x_tf = torch.cat((feats, coords_tf), dim=1)
        out.append(
            Data(
                x=x_tf,
                edge_index=data.edge_index,
                edge_attr=data.edge_attr,
                batch=data.batch,
                global_feats=data.global_feats,
            )
        )
    return out

def _load_saved_game_samples(save_games_dir: str) -> list[Sample]:
    samples: list[Sample] = []
    paths = sorted(glob(os.path.join(save_games_dir, "game_*.json")))
    if not paths:
        return samples
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
        for enc in trajectory:
            label = 0.5 if winner is None else float(winner == enc.perspective)
            samples.append((enc, label))
    return samples

def get_make_dataset(
    num_games: int,
    player_factories: Iterable[Callable[[int], Player]],
    max_moves: int = 256,
    swap_roles: bool = False,
    save_games_dir: str | None = None,
) -> list[Sample]:
    samples: list[Sample] = []
    pf = list(player_factories)
    start_index = 0
    if save_games_dir:
        os.makedirs(save_games_dir, exist_ok=True)
        samples.extend(_load_saved_game_samples(save_games_dir))
        existing = [p for p in glob(os.path.join(save_games_dir, "game_*.json"))]
        if existing:
            def _idx(p: str) -> int:
                stem = os.path.basename(p)
                try:
                    return int(stem.split("_")[1].split(".")[0])
                except Exception:
                    return -1
            start_index = max(map(_idx, existing)) + 1
    for i in range(num_games):
        print(f"Generating game {i+1}/{num_games}...")
        this_pf = pf if (not swap_roles or (i % 2 == 0)) else list(reversed(pf))
        traj, moves, winner = play_self_play_game(this_pf, max_moves=max_moves)
        if save_games_dir:
            game_path = os.path.join(save_games_dir, f"game_{start_index + i:05d}.json")
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
            samples.append((enc, label))
    return samples

def _samples_to_data(samples: list[Sample], augment_sym: bool) -> list[Data]:
    out: list[Data] = []
    for enc, label in samples:
        datas = _augment_symmetries(enc) if augment_sym else [enc.data]
        for data in datas:
            data.y = torch.tensor([label], dtype=torch.float32)
            out.append(data)
    return out

def _plot_losses(train_losses: list[float], val_losses: list[float], out_path: str) -> None:
    import matplotlib.pyplot as plt
    plt = cast(Any, plt)
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="train")
    if any(torch.isfinite(torch.tensor(val_losses))):
        plt.plot(epochs, val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("BCE loss")
    plt.title("GNN eval loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"saved loss plot to {out_path}")

def train(
    train_dataset: list[Data],
    val_dataset: list[Data],
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
) -> tuple[GNNEval, list[float], list[float]]:
    if not train_dataset:
        raise ValueError("Dataset is empty; generate samples first.")
    device_t = torch.device(device)
    sample = train_dataset[0]
    node_feat_dim = sample.x.size(1) # type: ignore
    global_feat_dim = sample.global_feats.size(1)
    model = GNNEval(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim).to(device_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    train_losses: list[float] = []
    val_losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device_t)
            logits = model(batch)
            loss = criterion(logits, batch.y.view_as(logits))
            opt.zero_grad()
            loss.backward()
            opt.step() # type: ignore
            total_loss += float(loss.item())
        avg_loss = total_loss / max(1, len(train_loader))
        val_loss = None
        if val_loader is not None:
            model.eval()
            v_total = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device_t)
                    logits = model(batch)
                    loss = criterion(logits, batch.y.view_as(logits))
                    v_total += float(loss.item())
            val_loss = v_total / max(1, len(val_loader))
        train_losses.append(avg_loss)
        val_losses.append(val_loss if val_loss is not None else float("nan"))
        if val_loss is None:
            print(f"epoch {epoch+1}/{epochs} train loss={avg_loss:.4f}")
        else:
            print(f"epoch {epoch+1}/{epochs} train loss={avg_loss:.4f} test loss={val_loss:.4f}")
    model.eval()
    return model, train_losses, val_losses

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
        return MCTSPlayer(idx, time_limit=time_limit, n_rollouts=n_rollouts if n_rollouts is not None else 1000, max_sim_depth=30, seed=seed)
    def greedy_factory(idx: int) -> Player:
        return OnePlyGreedyPlayer(idx)
    def ab_factory(idx: int) -> Player:
        x = random.random()
        if x < 0.6:
            return AlphaBetaPlayer(idx, depth=args.ab_depth)
        elif x < 0.8:
            return AlphaBetaPlayer(idx, depth=max(1, args.ab_depth - 1))
        return RockBiasedRandomPlayer(idx)
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
    samples = []
    if args.games is not None:
        samples = get_make_dataset(
            args.games,
            player_factories,
            max_moves=args.max_moves,
            swap_roles=args.swap_roles,
            save_games_dir=args.save_games_dir,
        )
        if args.val_frac > 0.0:
            idx = list(range(len(samples)))
            random.shuffle(idx)
            split = max(1, int(len(idx) * args.val_frac))
            train_samples = [samples[i] for i in idx[split:]]
            val_samples = [samples[i] for i in idx[:split]]
        else:
            train_samples = samples
            val_samples = []
        train_dataset = _samples_to_data(train_samples, augment_sym=args.augment_sym)
        val_dataset = _samples_to_data(val_samples, augment_sym=args.augment_sym) if val_samples else []
    else:
        train_dataset = []
        val_dataset = []
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
            _plot_losses(train_losses, val_losses, args.loss_plot_out)
    elif args.games is not None:
        print(f"Generated {len(samples)} base samples (no training; --epochs is 0)")
    else:
        print("Nothing to do: specify --games to generate or --epochs to train.")

if __name__ == "__main__":
    main()
