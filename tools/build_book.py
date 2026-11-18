"""Build the rs engine's opening book with long searches.

    caffeinate -i python3 tools/build_book.py --hours 10

The book covers the engine playing either side. It grows from the starting position, always
expanding the position the engine is most likely to meet next:

  * where the engine is to move, a long search (--node-minutes) picks its move;
  * where the opponent is to move, every reply gets a short search (--screen-seconds), and
    the replies that are hardest for the engine are followed up, weighted by how good they are
    for the opponent (the first move of the game is the exception: all of player 1's first
    moves are covered, since a human may open with anything).

The book corrects itself. Each entry is one search, which cannot see past its own horizon
(and a selective search does not always see the same things twice), so a book move sometimes
leads somewhere much worse than its search believed. Two checks catch that:

  * every newly searched position is played on for --verify-plies moves, the way two engines
    would play it (book moves where the book has them, otherwise --verify-seconds searches).
    If the evaluation falls more than --backup-margin below what the entry promised, the
    move is marked as bad, the line is walked back so the engine knows what lies ahead, and
    the position is searched again for the best move not yet shown to be bad;
  * whenever a position turns out to be worth more than --backup-margin less than the book
    move leading to it promised, the same is done for the position before it, and so on back
    up the line until the book agrees with itself. The opponent's entries along the line
    are raised to match.

Moves are told apart by the position they lead to, so a bad move and its mirror image in a
symmetric position count as the same move.

Variety. Up to --variety-ply the builder also follows the mover's other moves that a short
search rates within --variety-margin of the best (--variety-moves of them), so the book can
judge several moves in the same position and the engine has openings to choose between.
Nothing is prepared after a move shown to lose (worth less than minus --lost to its player).

--first LINE checks a particular line straight away, whether or not it is already in the book.

It stops after --hours, or when nothing within --max-ply is left. It can be stopped at any
time (Ctrl-C) and run again: finished work is kept in the book and the work file.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rs_engine  # noqa: E402

from game import Game  # noqa: E402
from players.base import RandomPlayer  # noqa: E402
from players.rs_book import DEFAULT_BOOK, Book, inverse, position_key, to_move, transform_move  # noqa: E402
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
        # key of a position -> {key of the position a move leads to: the most that move is worth},
        # for moves that turned out worse than they looked
        self.refuted: dict = {}
        self.counter = 0
        self.deep_done = 0
        self.recheck: list | None = None  # a line to verify even though it is already in the book (--first)
        if os.path.exists(self.work_path) and not args.fresh:
            with open(self.work_path, "r", encoding="utf-8") as fh:
                w = json.load(fh)
            self.screens = w.get("screens", {})
            # (entries that are not plain numbers are from an earlier format: drop them)
            self.refuted = {k: {c: v for c, v in bad.items() if isinstance(v, int)} for k, bad in w.get("refuted", {}).items()}
            self.refuted = {k: bad for k, bad in self.refuted.items() if bad}
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
            json.dump({"queue": [[-q[0], q[3]] for q in self.queue], "screens": self.screens, "refuted": self.refuted}, fh)
        os.replace(tmp, self.work_path)

    def screen_replies(self, game: Game, moves, all_equal: bool = False):
        """Score every reply of the side to move with a short search, from the other side's
        point of view. Replies that lead to symmetric images of each other are merged. Returns
        [(move, score)] worst-first for the side that is not moving."""
        # screens are cached by the position's canonical image, with moves stored in that
        # image's orientation; `sym` turns this game's moves into it
        key, sym = position_key(game)
        if key in self.screens and not all_equal:
            return [(transform_move(tuple(m), inverse(sym)), s) for m, s in self.screens[key]]
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
        if game.winner is not None or (self.recheck != [tuple(m) for m in moves] and self.lost_line(moves)):
            return
        mover = game.current_player
        entry = self.book.lookup(game)
        fresh = entry is None
        if fresh:
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
        if fresh or self.recheck == [tuple(m) for m in moves]:
            entry = self.settle(game, moves, entry)
            self.backup(priority, moves, entry["score"])
        best = tuple(entry["move"])
        score = entry["score"]
        if score <= -MATE_BOUND:
            return  # lost against best play: nothing to prepare
        if score >= MATE_BOUND and 30000 - score <= self.args.live_mate_plies:
            return  # the normal search finds a win this short by itself
        if len(moves) < self.args.variety_ply:
            # the mover's other good moves, so that the book can judge more than one move here
            bad = self.refuted.get(position_key(game)[0], {})
            own = [(m, s) for m, s in self.screen_replies(game, moves) if self.after(game, m) not in bad]
            for m, s in own[: self.args.variety_moves]:
                if s - own[0][1] <= self.args.variety_margin:
                    self.push(priority / 2, list(moves) + [m])
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

    def lost_line(self, moves) -> bool:
        """Does the line pass through a move shown to lose? Nothing needs preparing after one."""
        g = replay([])
        for m in moves:
            bad = self.refuted.get(position_key(g)[0])
            if bad and bad.get(self.after(g, m), 0) <= -self.args.lost:
                return True
            g.do_move(g.current_player, to_move(tuple(m)))
        return False

    def play_on(self, game: Game, moves, entry: dict) -> tuple[int, list[tuple[list, bool]]]:
        """Play on from a book position the way two engines would: a book move where the book
        has one, otherwise a short search. Returns what the position looks worth at the end,
        for the side to move in it, and the lines passed through (with whether each was a
        book position)."""
        mover = game.current_player
        ms = int(self.args.verify_seconds * 1000)
        line = list(moves) + [tuple(entry["move"])]
        g = replay(line)
        visited: list[tuple[list, bool]] = []
        last = entry["score"]
        for _ in range(self.args.verify_plies):
            if g.winner is not None:
                last = 30000 if g.winner == mover else -30000
                break
            known = self.book.lookup(g)
            if known is None:
                sync_engine(self.eng, g)
                known = self.eng.search(ms, 0, 0)
            last = int(known["score"]) if g.current_player == mover else -int(known["score"])
            visited.append((list(line), "nodes" not in known))
            if abs(last) >= MATE_BOUND:
                break
            mv = tuple(known["move"])
            g.do_move(g.current_player, to_move(mv))
            line.append(mv)
        return last, visited

    def walk_back(self, base: int, visited: list[tuple[list, bool]], last: int, mover: int) -> None:
        """A played-on line came to `last` (for `mover`). Carry that back along the line, last
        position first. Positions the line had to search are searched again, so that each
        search knows what the next one found. A book position is never simply searched again
        (a fresh search may not see what the book has already learned): its entry is raised
        if its move led to more than it promised, and if it led to clearly less the move is
        marked as bad there and the best of the others takes its place."""
        worth = None  # what the next position along the line is worth to the side to move in it
        for pos, in_book in reversed(visited):
            g = replay(pos)
            led_to = -worth if worth is not None else (last if g.current_player == mover else -last)
            if not in_book:
                sync_engine(self.eng, g)
                worth = int(self.eng.search(int(self.args.verify_seconds * 1000), 0, 0)["score"])
                continue
            old = self.book.lookup(g)
            if old["score"] - led_to > self.args.backup_margin:
                self.refute(g, old["move"], led_to)
                self.research(g)
            elif led_to - old["score"] > self.args.backup_margin:
                self.book.store(g, old["move"], led_to, old["depth"])
            now = self.book.lookup(g)
            worth = now["score"]
            if now != old:
                print(
                    f"    on the way back, ply {len(pos)}: ...{fmt(pos[base:])}  ->  {fmt([now['move']])}   {describe(now['score'], g.current_player).text}"
                    f"   (was {fmt([old['move']])}, {describe(old['score'], g.current_player).text})",
                    flush=True,
                )

    @staticmethod
    def after(game: Game, move) -> str:
        """The key of the position a move leads to. The book tells moves apart by this, so the
        mirror image of a move in a symmetric position is the same move."""
        game.do_move(game.current_player, to_move(tuple(move)))
        key = position_key(game)[0]
        game.undo_move()
        return key

    def refute(self, game: Game, move, value: int) -> None:
        """Record that `move` here is worth no more than `value`."""
        bad = self.refuted.setdefault(position_key(game)[0], {})
        child = self.after(game, move)
        bad[child] = min(int(value), bad.get(child, int(value)))

    def fall_back(self, game: Game, depth: int) -> None:
        """Make the book entry here the move that fell least of those shown to be bad, at what
        it was shown to be worth."""
        bad = self.refuted[position_key(game)[0]]
        least = max(bad, key=lambda k: bad[k])
        move = next(t for t in self.eng_moves(game) if self.after(game, t) == least)
        self.book.store(game, move, bad[least], depth)

    @staticmethod
    def eng_moves(game: Game) -> list:
        return [(0, 0, "P") if m.t == "P" else (m.c[0], m.c[1], m.t) for m in game.get_possible_moves(game.current_player)]

    def research(self, game: Game) -> dict:
        """Search the position for the best move not shown to be bad and make it the book
        entry, unless a move shown to be bad is still worth more (then that one is, at what
        it was shown to be worth). Returns the search result."""
        bad = self.refuted.get(position_key(game)[0], {})
        excluded = [t for t in self.eng_moves(game) if self.after(game, t) in bad] if bad else []
        sync_engine(self.eng, game)
        r = self.eng.search(int(self.args.node_minutes * 60000), 0, 0, False, excluded)
        if r["depth"] > 0 and (not bad or int(r["score"]) > max(bad.values())):
            self.book.store(game, tuple(r["move"]), int(r["score"]), int(r["depth"]))
        else:
            old = self.book.lookup(game)
            self.fall_back(game, int(old["depth"] if old else r["depth"]))
        return r

    def settle(self, game: Game, moves, entry: dict) -> dict:
        """Make the book entry for this position one that survives being played on. While the
        entry's move ends up clearly below what it promises, it is marked as bad, the line is
        walked back, and the position is searched again for the best move not yet shown to be
        bad. If nothing holds up, the least bad move is kept at what it was shown to be worth.
        Returns the entry to rely on."""
        if self.args.verify_plies <= 0:
            return entry
        mover = game.current_player
        bad = self.refuted.setdefault(position_key(game)[0], {})
        for attempt in range(self.args.max_tries + 1):
            last, visited = self.play_on(game, moves, entry)
            if entry["score"] - last <= self.args.backup_margin:
                break
            t0 = time.time()
            self.refute(game, entry["move"], last)
            came_to = f"    played on, ply {len(moves)} {fmt([tuple(entry['move'])])} came to {describe(last, mover).text} (it promised {describe(entry['score'], mover).text})"
            if attempt < self.args.max_tries:
                self.walk_back(len(moves), visited, last, mover)
                r = self.research(game)
                print(f"{came_to}; best other move: {fmt([tuple(r['move'])])}   {describe(int(r['score']), mover).text}  depth {r['depth']}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            else:
                self.fall_back(game, int(entry["depth"]))
                print(came_to, flush=True)
            entry = self.book.lookup(game)
            if self.after(game, entry["move"]) in bad:
                print(f"    ply {len(moves)}: no move holds up; keeping {fmt([entry['move']])} at {describe(entry['score'], mover).text}", flush=True)
                break
        if not bad:
            del self.refuted[position_key(game)[0]]
        return entry

    def backup(self, priority: float, moves, score: int) -> None:
        """The position after `moves` is worth `score` to the side to move in it. Make the
        book entries leading to it agree, one move back at a time:

          * if the opponent's book move leads here and the book rated their position lower
            than this makes it, raise that entry (their move is better than the book knew);
          * if our own book move two moves back promised clearly more than this, that move
            is bad: find the best move not yet shown to be bad, check it by playing on
            (settle), replace the entry, and carry on up the line with the new value."""
        moves = list(moves)
        while moves:
            before = replay(moves[:-1])
            theirs = self.book.lookup(before)
            if theirs is not None and self.after(before, theirs["move"]) == self.after(before, moves[-1]) and -score - theirs["score"] > self.args.backup_margin:
                self.book.store(before, tuple(theirs["move"]), -score, int(theirs["depth"]))
                print(f"    ply {len(moves) - 1}: {fmt([tuple(theirs['move'])])} is better than the book knew: now {describe(-score, before.current_player).text}", flush=True)
            if len(moves) < 2:
                break
            parent = moves[:-2]
            game = replay(parent)
            entry = self.book.lookup(game)
            played = tuple(moves[-2])
            if entry is None or self.after(game, entry["move"]) != self.after(game, played) or entry["score"] - score <= self.args.backup_margin:
                break
            mover = game.current_player
            t0 = time.time()
            self.refute(game, played, score)
            r = self.research(game)
            print(
                f"    corrected ply {len(parent)}: {fmt([played])} led to {describe(score, mover).text} (it promised {describe(entry['score'], mover).text});"
                f" best other move: {fmt([tuple(r['move'])])}   {describe(int(r['score']), mover).text}  depth {r['depth']}  ({(time.time() - t0) / 60:.1f} min)",
                flush=True,
            )
            checked = self.settle(game, parent, self.book.lookup(game))
            if self.after(game, checked["move"]) != self.after(game, played):
                self.push(priority, parent)  # follow the new line from here
            self.save()
            moves, score = parent, int(checked["score"])
        self.save()

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
    ap.add_argument("--backup-margin", type=int, default=100,
                    help="re-search the position before one that scores this much below what its book move promised (200 = from 50%% to about 27%%)")
    ap.add_argument("--verify-plies", type=int, default=10, help="play each newly searched position on for this many moves to check it (0 = off)")
    ap.add_argument("--max-tries", type=int, default=3, help="how many different moves to try at a position whose book move keeps falling apart")
    ap.add_argument("--verify-seconds", type=float, default=10.0, help="search time per move while playing a position on")
    ap.add_argument("--variety-ply", type=int, default=4, help="look at the mover's other good moves in positions before this ply (0 = off)")
    ap.add_argument("--variety-moves", type=int, default=3, help="at most this many moves followed per position for variety")
    ap.add_argument("--variety-margin", type=int, default=60, help="... that a short search rates within this much of the best")
    ap.add_argument("--lost", type=int, default=400, help="a move shown to be worth less than minus this loses: nothing after it is prepared")
    ap.add_argument("--first", default=None, metavar="LINE", help='a line to look at before anything else, e.g. "0,0S 0,-1S 0,-1R"')
    ap.add_argument("--live-mate-plies", type=int, default=11, help="forced wins this short are left to the normal search")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tt-bits", type=int, default=25)
    ap.add_argument("--book", default=DEFAULT_BOOK)
    ap.add_argument("--work", default=os.path.join(ROOT, "data", "rs", "book_work.json"))
    ap.add_argument("--fresh", action="store_true", help="ignore the saved work queue")
    args = ap.parse_args()
    builder = Builder(args)
    if args.first is not None:
        line = [(int(x), int(y), t) for x, y, t in re.findall(r"(-?\d+),(-?\d+)([A-Z]+)", args.first)]
        builder.queue = [q for q in builder.queue if [tuple(m) for m in q[3]] != line]
        heapq.heapify(builder.queue)
        builder.push(1.0, line)
        builder.recheck = line
    builder.run()


if __name__ == "__main__":
    main()
