import json
import math
import os
import random
from glob import glob
from typing import Callable, Iterable

from game import Game, GameProtocol
from gnn.encode import EncodedGraph, encode_game_to_graph
from models import D, Move
from players import Player


def randomize_start(
    game: GameProtocol,
    max_sticks: int = 5,
    max_rocks: int = 3, # per player
    move_log: list[Move] | None = None,
) -> None:
    # not true randomization; biased towards interesting / good for training positions

    weights = [math.exp(-0.33 * ((k - (max_sticks + 0.5) / 2)) ** 4) for k in range(1, max_sticks + 1)]
    target_sticks = random.choices(range(1, max_sticks + 1), weights=weights, k=1)[0]

    attempts = 0
    while target_sticks > 0 and attempts < 20:
        attempts += 1
        moves = [m for m in game.get_possible_moves(game.current_player) if m.t in D.__members__]
        if not moves:
            break
        mv = random.choice(moves)
        game.do_move(game.current_player, mv)
        failed = False
        for move in game.get_possible_moves(game.current_player):
            if move.t in D.__members__:
                game.do_move(game.current_player, move)
                if max(game.players_scores) > 0:
                    failed = True
                    game.undo_move()
                    break
                game.undo_move()
        if failed:
            game.undo_move()
            continue
        target_sticks -= 1
        if move_log is not None:
            move_log.append(mv)

    alpha = 0.7
    r_weights = [math.exp(alpha * k) for k in range(0, max_rocks)]
    rocks_each = random.choices(range(0, max_rocks), weights=r_weights, k=1)[0]
    for p in game.players:
        for _ in range(rocks_each + (random.random() > 0.3)):
            rock_moves = [m for m in game.get_possible_moves(p.number) if m.t == "R"]
            if not rock_moves:
                break
            weights = [12.0 if ((node := game.points.get(m.c)) is not None and node.connected) else 1.0 for m in rock_moves]
            mv = random.choices(rock_moves, weights=weights, k=1)[0]
            game.do_move(p.number, mv)
            if move_log is not None:
                move_log.append(mv)

    game.set_current_player0()

def play_self_play_game(
    player_factories: Iterable[Callable[[int], Player]],
    max_moves: int = 256
) -> tuple[list[EncodedGraph], list[Move], int | None]:

    players = [factory(i) for i, factory in enumerate(player_factories)]
    game = Game(players)
    trajectory: list[EncodedGraph] = []
    move_log: list[Move] = []
    randomize_start(game, move_log=move_log)

    while game.winner is None and len(game.moves) < max_moves:
        player = game.players[game.current_player]
        trajectory.append(encode_game_to_graph(game))
        mv = player.get_move(game)
        move_log.append(mv)
        game.do_move(player.number, mv)
    return trajectory, move_log, game.winner


def generate_self_play_games(
    num_games: int,
    player_factories: Iterable[Callable[[int], Player]],
    save_games_dir: str,
    max_moves: int = 256,
    swap_roles: bool = True
) -> None:

    pf = list(player_factories)
    start_index = 0
    os.makedirs(save_games_dir, exist_ok=True)
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
        _, moves, winner = play_self_play_game(this_pf, max_moves=max_moves)
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
