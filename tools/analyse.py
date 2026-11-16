"""Search one position for a long time, reporting each depth as it completes.

    python3 tools/analyse.py --minutes 60                       # the starting position
    python3 tools/analyse.py --moves "0,1NE 0,0E" --minutes 10  # after some moves

Moves are written x,yT with T one of N NE E SE S SW W NW R, or P for a pass. Scores are shown
the chess way (positive favours Player 1); see players/rs_display.py.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rs_engine  # noqa: E402

from players.rs_display import describe  # noqa: E402

from players.rs_player import DEFAULT_WEIGHTS as WEIGHTS  # noqa: E402


def parse_moves(text: str) -> list[tuple[int, int, str]]:
    out = []
    for tok in text.split():
        if tok == "P":
            out.append((0, 0, "P"))
            continue
        m = re.fullmatch(r"(-?\d+),(-?\d+)([A-Z]+)", tok)
        if not m:
            raise ValueError(f"bad move: {tok}")
        out.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    return out


def fmt(moves) -> str:
    return " ".join("P" if t == "P" else f"{x},{y}{t}" for x, y, t in moves)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moves", default="", help="moves played from the starting position")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tt-bits", type=int, default=25, help="log2 of transposition table entries (25 = 512 MB)")
    ap.add_argument("--start-depth", type=int, default=6)
    ap.add_argument("--max-depth", type=int, default=60)
    args = ap.parse_args()

    eng = rs_engine.Engine(args.tt_bits, args.threads)
    if not eng.load_weights(WEIGHTS):
        raise SystemExit(f"could not load {WEIGHTS}")
    for x, y, t in parse_moves(args.moves):
        eng.do_move(x, y, t)
    mover = eng.current_player
    print(f"position after: {args.moves or '(start)'}; player {mover + 1} to move; {args.threads} threads", flush=True)

    budget = args.minutes * 60.0
    t0 = time.time()
    total_nodes = 0
    for depth in range(args.start_depth, args.max_depth + 1):
        remaining_ms = int((budget - (time.time() - t0)) * 1000)
        if remaining_ms <= 0:
            break
        r = eng.search(remaining_ms, depth, 0)
        total_nodes += r["nodes"]
        el = time.time() - t0
        done = r["depth"] >= depth
        ev = describe(r["score"], mover)
        tag = "" if done else f"  (stopped at depth {r['depth']})"
        print(f"depth {depth:2d}  {ev.text:24s} raw {r['score']:6d}  {el / 60:6.1f} min  {total_nodes / 1e6:9.1f}M nodes  {fmt(r['pv'][:14])}{tag}", flush=True)
        if abs(r["score"]) >= 29000:
            print("forced result found", flush=True)
            break
        if not done:
            break
    print(f"finished after {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
