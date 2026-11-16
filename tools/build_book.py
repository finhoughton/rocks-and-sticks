"""Build the rs engine's opening book with long searches.

    caffeinate -i python3 tools/build_book.py --hours 10

The book covers the engine playing either side. It grows from the starting position, always
expanding the position the engine is most likely to meet next:

  * where the engine is to move, a long search (--node-minutes) picks its move;
  * where the opponent is to move, every reply gets a short search (--screen-seconds), and
    the replies that are hardest for the engine are followed up, weighted by how good they are
    for the opponent (the first move of the game is the exception: all of player 1's first
    moves are covered, since a human may open with anything).

It stops after --hours, or when nothing within --max-ply is left. It can be stopped at any
time (Ctrl-C) and run again: finished work is kept in the book and the work file.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rs_engine  # noqa: E402

from game import Game  # noqa: E402
from players.base import RandomPlayer  # noqa: E402
from players.rs_book import DEFAULT_BOOK, Book, position_key, to_move, transform_move  # noqa: E402
from players.rs_display import describe  # noqa: E402
from players.rs_player import sync_engine  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from players.rs_player import DEFAULT_WEIGHTS as WEIGHTS  # noqa: E402
MATE_BOUND = 29000


def fmt(moves) -> str:
    return " ".join("P" if t == "P" else f"{x},{y}{t}" for x, y, t in moves) or "(start)"


def replay(moves) -> Game:
    g = Game([RandomPlayer(0), RandomPlayer(1)])
    for m in moves:
        g.do_move(g.current_player, to_move(tuple(m)))
    return g


def near_structure(game: Game, x: int, y: int) -> bool:
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            p = game.points.get((x + dx, y + dy))
            if p is not None and p.connected:
                return True
    return False


class Builder:
    def __init__(self, args) -> None:
        self.args = args
        self.eng = rs_engine.Engine(args.tt_bits, args.threads)
        if not self.eng.load_weights(WEIGHTS):
            raise SystemExit(f"could not load {WEIGHTS}")
        self.book = Book(args.book)
        self.work_path = args.work
        self.queue: list = []       # (-priority, ply, counter, moves)
        self.screens: dict = {}     # key of an opponent-to-move position -> [[move, score], ...]
        self.counter = 0
        self.deep_done = 0
        if os.path.exists(self.work_path) and not args.fresh:
            with open(self.work_path, "r", encoding="utf-8") as fh:
                w = json.load(fh)
            self.screens = w.get("screens", {})
            for pr, moves in w.get("queue", []):
                self.push(pr, [tuple(m) for m in moves])
            print(f"resumed: {len(self.book)} book positions, {len(self.queue)} queued", flush=True)
        if not self.queue:
            # the engine as player 1: the starting position; as player 2: after any first move
            self.push(0.5, [])
            g = replay([])
            # (an opening pass is left to the normal search)
            firsts = [f for f in self.screen_replies(g, [], all_equal=True) if f[0][2] != "P"]
            for m, _ in firsts:
                self.push(0.5 / len(firsts), [m])

    def push(self, priority: float, moves) -> None:
        self.counter += 1
        heapq.heappush(self.queue, (-priority, len(moves), self.counter, list(moves)))

    def save(self) -> None:
        self.book.save()
        tmp = self.work_path + ".tmp"
        os.makedirs(os.path.dirname(self.work_path) or ".", exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"queue": [[-q[0], q[3]] for q in self.queue], "screens": self.screens}, fh)
        os.replace(tmp, self.work_path)

    def screen_replies(self, game: Game, moves, all_equal: bool = False):
        """Score every reply of the side to move with a short search, from the other side's
        point of view. Replies that lead to symmetric images of each other are merged. Returns
        [(move, score)] worst-first for the side that is not moving."""
        # screens are cached by the position's canonical image, with moves stored in that
        # image's orientation; `sym` turns them to and from this game's orientation
        key, sym = position_key(game)
        if key in self.screens and not all_equal:
            return [(transform_move(tuple(m), sym), s) for m, s in self.screens[key]]
        sync_engine(self.eng, game)
        seen: set = set()
        out = []
        for m in self.eng.legal_moves(True):
            x, y, t = m
            if t == "R" and not near_structure(game, x, y):
                continue
            g2 = replay(list(moves) + [m])
            k2, _ = position_key(g2)
            if k2 in seen:
                continue
            seen.add(k2)
            if g2.winner is not None:
                score = MATE_BOUND + 900 if g2.winner != game.current_player else -(MATE_BOUND + 900)
            elif all_equal:
                score = 0
            else:
                self.eng.do_move(x, y, t)
                score = int(self.eng.search(int(self.args.screen_seconds * 1000), 0, 0)["score"])
                self.eng.undo_move()
            out.append((m, score))
        out.sort(key=lambda ms: ms[1])
        if not all_equal:
            self.screens[key] = [[list(transform_move(tuple(m), sym)), s] for m, s in out]
        return out

    def expand(self, priority: float, moves) -> None:
        """`moves` leads to a position where the engine is to move."""
        game = replay(moves)
        if game.winner is not None:
            return
        mover = game.current_player
        entry = self.book.lookup(game)
        if entry is None:
            sync_engine(self.eng, game)
            t0 = time.time()
            r = self.eng.search(int(self.args.node_minutes * 60000), 0, 0)
            self.book.store(game, tuple(r["move"]), int(r["score"]), int(r["depth"]))
            entry = self.book.lookup(game)
            self.deep_done += 1
            print(
                f"[{self.deep_done}] p={priority:.4f} ply {len(moves)}: {fmt(moves)}  ->  {fmt([entry['move']])}   "
                f"{describe(entry['score'], mover).text}  depth {entry['depth']}  ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
        best = tuple(entry["move"])
        score = entry["score"]
        if score <= -MATE_BOUND:
            return  # lost against best play: nothing to prepare
        if score >= MATE_BOUND and 30000 - score <= self.args.live_mate_plies:
            return  # the normal search finds a win this short by itself
        line = list(moves) + [best]
        if len(line) + 1 > self.args.max_ply:
            return
        g2 = replay(line)
        if g2.winner is not None:
            return
        replies = self.screen_replies(g2, line)
        if not replies:
            return
        # the opponent prefers replies that score badly for the engine
        lo = replies[0][1]
        weights = [math.exp(-(s - lo) / self.args.temperature) for _, s in replies]
        total = sum(weights)
        kept = 0
        for (m, s), w in zip(replies, weights):
            p = w / total
            if kept >= self.args.replies or p < self.args.min_share:
                break
            if s >= MATE_BOUND and 30000 - s <= self.args.live_mate_plies:
                continue  # this reply loses quickly
            self.push(priority * p, line + [m])
            kept += 1

    def run(self) -> None:
        t_end = time.time() + self.args.hours * 3600
        try:
            while self.queue and time.time() < t_end:
                neg, _ply, _c, moves = heapq.heappop(self.queue)
                try:
                    self.expand(-neg, moves)
                except BaseException:
                    self.push(-neg, moves)  # not finished: keep it for the next run
                    raise
                self.save()
        except KeyboardInterrupt:
            print("interrupted", flush=True)
        self.save()
        print(f"book: {len(self.book)} positions; {len(self.queue)} still queued; saved to {self.book.path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=8.0)
    ap.add_argument("--node-minutes", type=float, default=4.0, help="long search per position where the engine is to move")
    ap.add_argument("--screen-seconds", type=float, default=3.0, help="short search per opponent reply")
    ap.add_argument("--replies", type=int, default=5, help="at most this many opponent replies followed per position")
    ap.add_argument("--min-share", type=float, default=0.04, help="ignore replies given less than this share of the weight")
    ap.add_argument("--temperature", type=float, default=150.0, help="how sharply replies are weighted by their score")
    ap.add_argument("--max-ply", type=int, default=12)
    ap.add_argument("--live-mate-plies", type=int, default=11, help="forced wins this short are left to the normal search")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tt-bits", type=int, default=25)
    ap.add_argument("--book", default=DEFAULT_BOOK)
    ap.add_argument("--work", default=os.path.join(ROOT, "data", "rs", "book_work.json"))
    ap.add_argument("--fresh", action="store_true", help="ignore the saved work queue")
    args = ap.parse_args()
    Builder(args).run()


if __name__ == "__main__":
    main()
