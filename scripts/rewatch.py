"""Replay a saved Rocks and Sticks game move by move.

    python3 scripts/rewatch.py saved_games_human/game_00040.json
    python3 scripts/rewatch.py saved_games_human/game_00040.json --analyse 2

If the game was saved with the rs engine's evaluations they are shown as a chess-style bar and
printed next to each move. With --analyse SECONDS the rs engine evaluates every position
afresh (thinking that long on each), which also works for games saved without evaluations.

Reading the evaluation: positive favours Player 1, negative Player 2, with the estimated win
chance beside it; "+M3" means Player 1 has a forced win in 3 of its own moves.
"""

import argparse
import json
import os
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Game  # noqa: E402
from models import PASS, Move  # noqa: E402
from players import Player, RandomPlayer  # noqa: E402


def load_moves_from_json(path: str) -> Tuple[List[Tuple[int | None, Move]], Optional[str]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    moves: List[Tuple[int | None, Move]] = []
    for mv in payload["moves"]:
        p = mv.get("p", None)
        if mv["t"] == "P":
            moves.append((p, PASS))
        else:
            moves.append((p, Move(int(mv["x"]), int(mv["y"]), str(mv["t"]))))
    winner: Optional[str] = payload.get("winner", None)
    return moves, winner


def rewatch_game(path: str, delay: float = 1.0, analyse: float = 0.0, threads: int = 4) -> None:
    moves, winner = load_moves_from_json(path)
    with open(path, "r", encoding="utf-8") as fh:
        saved_evals = json.load(fh).get("evals") or []
    players: List[Player] = [RandomPlayer(0), RandomPlayer(1)]
    game = Game(players)

    import matplotlib.pyplot as plt

    show_eval = analyse > 0 or any(e for e in saved_evals)
    analyst = None
    if show_eval:
        from players.rs_display import describe, draw_eval_bar, final
    if analyse > 0:
        from players.rs_player import RSPlayer

        analyst = RSPlayer(0, time_ms=int(analyse * 1000), threads=threads, tt_bits=22)

    game.render(block=False)
    for i, (p, mv) in enumerate(moves):
        mover = int(p) if p is not None else game.current_player
        game.do_move(mover, mv)
        ev = None
        if show_eval:
            if game.winner is not None:
                ev = final(game.winner)
            elif analyst is not None:
                info = analyst.analyse(game)
                ev = describe(info["score"], info["mover"])
            elif i < len(saved_evals) and saved_evals[i]:
                # recorded by the engine just before it played this move
                ev = describe(int(saved_evals[i]["score"]), int(saved_evals[i]["mover"]))
        note = f"   {ev.text}" if ev is not None else ""
        print(f"Move {i+1}: Player {mover+1} plays {mv}{note}")
        game.render(block=False)
        if show_eval:
            draw_eval_bar(plt.gcf(), ev)
        plt.pause(delay)
    print(f"Game over. Winner: {winner}")

    plt.show() # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("game_file", type=str, help="Path to saved game JSON")
    parser.add_argument("--delay", type=float, default=0.8, help="Delay between moves (seconds)")
    parser.add_argument("--analyse", type=float, default=0.0, help="have the rs engine evaluate each position for this many seconds")
    parser.add_argument("--threads", type=int, default=4, help="search threads for --analyse")
    args = parser.parse_args()
    rewatch_game(args.game_file, delay=args.delay, analyse=args.analyse, threads=args.threads)
