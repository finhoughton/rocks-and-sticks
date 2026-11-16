"""Opening book for the rs engine: positions mapped to a move found by a long search.

The starting stick runs from (0,0) to (0,1), and the rules treat every direction alike, so the
game has four symmetries: the left-right mirror (x -> -x), the top-bottom mirror (y -> 1 - y)
and both together (a half turn). Each position is stored under whichever of its images sorts
first, and the stored move is turned back to fit the game it is looked up for.

Book files written before November 2026 (format 1) used only the left-right mirror; they can
still be read, with lookups that match how they were written.

Built by tools/build_book.py.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING

from models import PASS, Move

if TYPE_CHECKING:
    from game import GameProtocol

# (flip x, flip y); each is its own inverse
Symmetry = tuple[bool, bool]
SYMMETRIES: tuple[Symmetry, ...] = ((False, False), (True, False), (False, True), (True, True))
FORMAT = 2

FLIP_X = {"N": "N", "S": "S", "E": "W", "W": "E", "NE": "NW", "NW": "NE", "SE": "SW", "SW": "SE", "R": "R", "P": "P"}
FLIP_Y = {"N": "S", "S": "N", "E": "E", "W": "W", "NE": "SE", "SE": "NE", "NW": "SW", "SW": "NW", "R": "R", "P": "P"}

DEFAULT_BOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine_src", "book.json")

MoveTuple = tuple[int, int, str]


def transform_point(x: int, y: int, sym: Symmetry) -> tuple[int, int]:
    return (-x if sym[0] else x, 1 - y if sym[1] else y)


def transform_move(m: MoveTuple, sym: Symmetry) -> MoveTuple:
    x, y, t = m
    if t == "P":
        return (0, 0, "P")
    if sym[0]:
        x, t = -x, FLIP_X[t]
    if sym[1]:
        y, t = 1 - y, FLIP_Y[t]
    return (x, y, t)


def mirror_move(m: MoveTuple) -> MoveTuple:
    """The left-right mirror image of a move."""
    return transform_move(m, (True, False))


def _describe(game: "GameProtocol", sym: Symmetry) -> str:
    def pt(p) -> tuple[int, int]:
        return transform_point(p.x, p.y, sym)

    sticks = sorted(tuple(sorted((pt(s.start), pt(s.end)))) for s in game.sticks)
    rocks = sorted((pt(r), r.rocked_by.number) for r in game.rocks if r.rocked_by is not None)
    # the turn number only matters while it is 0 or 1
    state = (game.current_player, min(game.turn_number, 2), tuple(game.num_rocks), tuple(game.players_scores))
    return repr((sticks, rocks, state))


def position_key(game: "GameProtocol", symmetries: tuple[Symmetry, ...] = SYMMETRIES) -> tuple[str, Symmetry]:
    """(key, sym): the key of the position's canonical image, and the symmetry that turns the
    game into that image (and back again)."""
    text, sym = min((_describe(game, s), s) for s in symmetries)
    return hashlib.sha1(text.encode()).hexdigest()[:20], sym


class Book:
    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        self.format = FORMAT
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.entries = data.get("entries", {})
            self.format = int(data.get("version", 1))
        # format 1 books were keyed with the left-right mirror only
        self.symmetries = SYMMETRIES if self.format >= 2 else SYMMETRIES[:2]

    def __len__(self) -> int:
        return len(self.entries)

    def key(self, game: "GameProtocol") -> tuple[str, Symmetry]:
        return position_key(game, self.symmetries)

    def lookup(self, game: "GameProtocol") -> dict | None:
        """The book entry for this position with its move oriented for the game, or None."""
        key, sym = self.key(game)
        e = self.entries.get(key)
        if e is None:
            return None
        m = transform_move((int(e["m"][0]), int(e["m"][1]), str(e["m"][2])), sym)
        return {"move": m, "score": int(e["s"]), "depth": int(e["d"])}

    def store(self, game: "GameProtocol", move: MoveTuple, score: int, depth: int) -> None:
        if self.format < FORMAT:
            raise ValueError("this book uses the old format: convert it before adding to it")
        key, sym = self.key(game)
        m = transform_move(move, sym)
        self.entries[key] = {"m": [m[0], m[1], m[2]], "s": int(score), "d": int(depth)}

    def save(self, path: str | None = None) -> None:
        path = path or self.path
        assert path is not None
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"version": self.format, "entries": self.entries}, fh)
        os.replace(tmp, path)


def to_move(m: MoveTuple) -> Move:
    return PASS if m[2] == "P" else Move(int(m[0]), int(m[1]), str(m[2]))
