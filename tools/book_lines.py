"""Write every line through the opening book to the position where it leaves the book.

    python3 tools/book_lines.py --out data/rs/book_lines.txt
    engine_src/rs_selfplay match --openings data/rs/book_lines.txt ...

From each position the walk follows the book's move and also every other move that leads to
a position the book covers (the book was built for both sides, so it covers the replies the
opponent is most likely to choose). A line ends at the first position with no book move and
no move into the book. Lines ending in the same position (or a mirror image of it) are
written once. Games started from these lines test how engines play the positions that
book-using engines actually reach.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game import Game  # noqa: E402
from models import PASS  # noqa: E402
from players.base import RandomPlayer  # noqa: E402
from players.rs_book import DEFAULT_BOOK, Book, position_key, to_move  # noqa: E402


def fmt(m) -> str:
    return "P" if m[2] == "P" else f"{m[0]},{m[1]}{m[2]}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default=DEFAULT_BOOK)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    book = Book(args.book)
    game = Game([RandomPlayer(0), RandomPlayer(1)])
    exits: dict[str, list] = {}
    visited: set[str] = set()

    def in_book(g) -> bool:
        return book.lookup(g) is not None

    def dfs(line: list) -> None:
        key = position_key(game)[0]
        if key in visited:
            return
        visited.add(key)
        kids = []
        entry = book.lookup(game)
        if game.winner is None:
            if entry is not None:
                kids.append(tuple(entry["move"]))
            for m in list(game.get_possible_moves(game.current_player)):
                t = (0, 0, "P") if m is PASS else (m.c[0], m.c[1], m.t)
                if t in kids:
                    continue
                game.do_move(game.current_player, m)
                if in_book(game):
                    kids.append(t)
                game.undo_move()
        if not kids:
            exits.setdefault(key, list(line))
            return
        for t in kids:
            game.do_move(game.current_player, to_move(t))
            dfs(line + [t])
            game.undo_move()

    dfs([])
    with open(args.out, "w", encoding="utf-8") as fh:
        for line in exits.values():
            fh.write(" ".join(fmt(m) for m in line) + "\n")
    lengths = [len(v) for v in exits.values()]
    print(f"{len(exits)} lines out of the book (lengths {min(lengths)}-{max(lengths)}, mean {sum(lengths) / len(lengths):.1f} moves); written to {args.out}")


if __name__ == "__main__":
    main()
