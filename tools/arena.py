"""Play an unrated match between two bots, optionally from random openings or move by move.

Examples:
    python3 tools/arena.py --a rs:200ms --b ab:3 --games 20
    python3 tools/arena.py --a rs:500ms --b mcts:2000 --games 10 --random-start --verbose

Bots are named by spec (players/specs.py): random, ab:<depth>, mcts:<rollouts>,
rs:<budget>[:<threads>t][:nobook] with a budget such as 5s, 500ms, 100k or 1M.
For rated games use tools/rate.py instead.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Game  # noqa: E402
from models import PASS  # noqa: E402
from players.base import Player, RandomPlayer  # noqa: E402


def make_player(spec: str, number: int, seed: int) -> Player:
    from players.specs import make_bot

    return make_bot(spec, number, seed=seed)


def play_game(spec_a: str, spec_b: str, a_is_p0: bool, seed: int, max_moves: int, random_start: bool, verbose: bool):
    import players_ext

    from players.game_total import GameTotal

    a_num = 0 if a_is_p0 else 1
    pa = make_player(spec_a, a_num, seed)
    pb = make_player(spec_b, 1 - a_num, seed + 1)
    players = [pa, pb] if a_is_p0 else [pb, pa]
    game = GameTotal(Game(players=players), players_ext.GameState())
    if random_start:
        from gnn.game_generation import randomize_start

        random.seed(seed)
        randomize_start(game, max_sticks=5)
    n = 0
    log = []
    while game.winner is None and n < max_moves:
        cp = game.current_player
        mv = game.players[cp].get_move(game)
        log.append("P" if mv is PASS else f"{mv.c[0]},{mv.c[1]}{mv.t}")
        game.do_move(cp, mv)
        n += 1
    if verbose:
        print("   ", " ".join(log))
    if game.winner is None:
        return 0.5, n
    return (1.0 if game.winner == a_num else 0.0), n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--max-moves", type=int, default=80)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--random-start", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    wins = draws = losses = 0
    t0 = time.time()
    for g in range(args.games):
        res, n = play_game(args.a, args.b, g % 2 == 0, args.seed + 7 * g, args.max_moves, args.random_start, args.verbose)
        wins += res == 1.0
        draws += res == 0.5
        losses += res == 0.0
        side = "P0" if g % 2 == 0 else "P1"
        print(f"game {g + 1}: A as {side} -> {'win' if res == 1.0 else 'draw' if res == 0.5 else 'loss'} in {n} moves", flush=True)
    print(f"A={args.a} vs B={args.b}: +{wins} ={draws} -{losses}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
