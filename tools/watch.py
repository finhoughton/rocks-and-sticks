"""Watch the rs engine play against itself.

    python3 tools/watch.py                 # 3 games, 2 seconds per move
    python3 tools/watch.py --games 5 --time 5 --delay 1.5
    python3 tools/watch.py --time0 5 --time1 0.5    # give one side more thinking time

Each game is drawn move by move with an evaluation bar, and the engine's assessment is
printed in the terminal. The evaluation reads like a chess one: positive favours Player 1,
negative Player 2, with the estimated win chance beside it; "+M3" means Player 1 has a forced
win in 3 of its own moves ("-M3" for Player 2).
Games are saved as JSON with the evaluations, so they can be replayed with scripts/rewatch.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Game  # noqa: E402
from models import PASS  # noqa: E402
from players.rs_display import describe, draw_eval_bar, final  # noqa: E402
from players.rs_player import RSPlayer  # noqa: E402


def _fmt(mv) -> str:
    return "pass" if mv is PASS else f"{mv.c[0]} {mv.c[1]} {mv.t}"


def _save(game: Game, save_dir: str, info: dict, evals: list) -> str:
    os.makedirs(save_dir, exist_ok=True)
    idx = len(glob.glob(os.path.join(save_dir, "game_*.json")))
    path = os.path.join(save_dir, f"game_{idx:05d}.json")
    payload = {
        "winner": game.winner,
        "moves": [{"x": m.c[0], "y": m.c[1], "t": m.t, "p": i % 2} for i, m in enumerate(game.moves)],
        "max_moves_reached": game.winner is None,
        "evals": evals,
        "meta": info,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--time", type=float, default=2.0, help="seconds per move for both sides")
    ap.add_argument("--time0", type=float, default=None, help="seconds per move for player 1 (overrides --time)")
    ap.add_argument("--time1", type=float, default=None, help="seconds per move for player 2 (overrides --time)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--delay", type=float, default=1.0, help="extra pause after each move, in seconds")
    ap.add_argument("--variety", type=int, default=15, help="how far from its best move the engine may stray, so games differ (0 = always best)")
    ap.add_argument("--max-moves", type=int, default=120)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-dir", default="saved_games_rs_selfplay")
    ap.add_argument("--no-display", action="store_true", help="print moves only, no window")
    args = ap.parse_args()

    times = [args.time0 if args.time0 is not None else args.time, args.time1 if args.time1 is not None else args.time]
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")

    if not args.no_display:
        import matplotlib.pyplot as plt

    tally = {0: 0, 1: 0, None: 0}
    for g in range(args.games):
        players = [
            RSPlayer(p, time_ms=int(times[p] * 1000), threads=args.threads, tt_bits=22, root_margin=args.variety, seed=seed + 2 * g + p)
            for p in (0, 1)
        ]
        game = Game(players)
        print(f"\n=== Game {g + 1}/{args.games} ===")
        evals: list = []

        def show(ev) -> None:
            if args.no_display:
                return
            game.render(block=False, game_index=g, total_games=args.games)
            draw_eval_bar(plt.gcf(), ev)
            plt.pause(args.delay)

        show(None)
        while game.winner is None and len(game.moves) < args.max_moves:
            cp = game.current_player
            mv = players[cp].get_move(game)
            info = players[cp].last_info
            ev = describe(info["score"], cp)
            evals.append({"score": int(info["score"]), "depth": int(info["depth"]), "mover": cp})
            print(f"{len(game.moves) + 1:3d}. player {cp + 1}: {_fmt(mv):10s}  {ev.text:24s} depth {info['depth']}")
            game.do_move(cp, mv)
            show(final(game.winner) if game.winner is not None else ev)
        tally[game.winner] += 1
        if game.winner is None:
            print(f"No winner after {len(game.moves)} moves")
        else:
            w = game.winner
            print(f"Player {w + 1} wins with area {game.players_scores[w] / 2:g} after {len(game.moves)} moves")
        path = _save(game, args.save_dir, {"mode": "rs-selfplay", "time": times, "threads": args.threads, "variety": args.variety}, evals)
        print(f"saved {path}")
        if not args.no_display:
            # leave the final position up for a moment, then start the next game in a new window
            plt.pause(max(3.0, 3 * args.delay))
            if g + 1 < args.games:
                plt.close("all")

    print(f"\nPlayer 1 wins: {tally[0]}   Player 2 wins: {tally[1]}   unfinished: {tally[None]}")
    if not args.no_display:
        print("Close the window to exit.")
        plt.show()


if __name__ == "__main__":
    main()
