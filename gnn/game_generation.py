import json
import os
import random
from glob import glob
from typing import Callable, Iterable

from game import Game
from gnn.encode import EncodedGraph, encode_game_to_graph
from models import D
from players import Move, Player


def randomize_start(
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
        game.do_move(player, mv)
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
