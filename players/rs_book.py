"""Opening book for the rs engine: positions mapped to a move found by a long search.

The rules treat every direction and every place on the grid alike, so a position is the same
position wherever it sits and whichever way round it is: turned by quarter turns, mirrored,
or shifted. Each position is stored under whichever of its eight images (shifted against the
axes) sorts first, and the stored move is turned back to fit the game it is looked up for.

Older book files knew fewer symmetries and can still be read, with lookups that match how
they were written: format 1 only the left-right mirror, format 2 the four symmetries that
leave the starting stick (0,0)-(0,1) where it is.

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

# A symmetry moves a point by: swap x and y, negate x, negate y (each if set), then add (dx, dy)
Symmetry = tuple[bool, bool, bool, int, int]
IDENTITY: Symmetry = (False, False, False, 0, 0)
# the four that leave the starting stick (0,0)-(0,1) where it is
SYMMETRIES: tuple[Symmetry, ...] = (IDENTITY, (False, True, False, 0, 0), (False, False, True, 0, 1), (False, True, True, 0, 1))
FORMAT = 3

DELTA = {"N": (0, 1), "NE": (1, 1), "E": (1, 0), "SE": (1, -1), "S": (0, -1), "SW": (-1, -1), "W": (-1, 0), "NW": (-1, 1)}
DIRECTION = {d: name for name, d in DELTA.items()}

DEFAULT_BOOK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine_src", "book.json")

MoveTuple = tuple[int, int, str]


def transform_point(x: int, y: int, sym: Symmetry) -> tuple[int, int]:
    swap, flip_x, flip_y, dx, dy = sym
    if swap:
        x, y = y, x
    return ((-x if flip_x else x) + dx, (-y if flip_y else y) + dy)


def inverse(sym: Symmetry) -> Symmetry:
    """The symmetry that undoes `sym`."""
    swap, flip_x, flip_y, dx, dy = sym
    if swap:
        flip_x, flip_y = flip_y, flip_x
    back = (swap, flip_x, flip_y, 0, 0)
    x, y = transform_point(dx, dy, back)
    return (swap, flip_x, flip_y, -x, -y)


def transform_move(m: MoveTuple, sym: Symmetry) -> MoveTuple:
    x, y, t = m
    if t == "P":
        return (0, 0, "P")
    if t != "R":
        t = DIRECTION[transform_point(*DELTA[t], sym[:3] + (0, 0))]
    return (*transform_point(x, y, sym), t)


def mirror_move(m: MoveTuple) -> MoveTuple:
    """The left-right mirror image of a move."""
    return transform_move(m, SYMMETRIES[1])


def all_symmetries(game: "GameProtocol") -> list[Symmetry]:
    """The eight turns and mirrors of the grid, each shifted to put the position's sticks
    against the axes, so that the images do not depend on where the position sits."""
    points = [(p.x, p.y) for s in game.sticks for p in (s.start, s.end)]
    out = []
    for swap in (False, True):
        for flip_x in (False, True):
            for flip_y in (False, True):
                image = [transform_point(x, y, (swap, flip_x, flip_y, 0, 0)) for x, y in points]
                out.append((swap, flip_x, flip_y, -min(x for x, _ in image), -min(y for _, y in image)))
    return out


def _describe(game: "GameProtocol", sym: Symmetry) -> str:
    def pt(p) -> tuple[int, int]:
        return transform_point(p.x, p.y, sym)

    sticks = sorted(tuple(sorted((pt(s.start), pt(s.end)))) for s in game.sticks)
    rocks = sorted((pt(r), r.rocked_by.number) for r in game.rocks if r.rocked_by is not None)
    # the turn number only matters while it is 0 or 1
    state = (game.current_player, min(game.turn_number, 2), tuple(game.num_rocks), tuple(game.players_scores))
    return repr((sticks, rocks, state))


def position_key(game: "GameProtocol", symmetries: tuple[Symmetry, ...] | None = None) -> tuple[str, Symmetry]:
    """(key, sym): the key of the position's canonical image, and the symmetry that turns the
    game into that image. `symmetries` limits the images considered (for old book files)."""
    text, sym = min((_describe(game, s), s) for s in symmetries or all_symmetries(game))
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
        # older books were keyed with fewer symmetries
        self.symmetries = {1: SYMMETRIES[:2], 2: SYMMETRIES}.get(self.format)

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
        m = transform_move((int(e["m"][0]), int(e["m"][1]), str(e["m"][2])), inverse(sym))
        return {"move": m, "score": int(e["s"]), "depth": int(e["d"])}

    def _options(self, game: "GameProtocol") -> list[tuple[list[MoveTuple], int]]:
        """Every move the book can judge here, best first: ([the move and its images], its
        value to the mover). Moves that lead to images of the same position are one option;
        the book's own way of playing its move comes first among its images."""
        entry = self.lookup(game)
        if entry is None:
            return []
        mover = game.current_player
        game.do_move(mover, to_move(entry["move"]))
        own = self.key(game)[0]  # (the same stick can be named from either end: compare positions)
        game.undo_move()
        by_position: dict[str, tuple[list[MoveTuple], int]] = {}
        for m in list(game.get_possible_moves(mover)):
            t: MoveTuple = (0, 0, "P") if m is PASS else (m.c[0], m.c[1], m.t)
            game.do_move(mover, m)
            key = self.key(game)[0]
            child = None if game.winner is not None else self.entries.get(key)
            game.undo_move()
            if key in by_position:
                by_position[key][0].append(t)
            elif child is not None:
                by_position[key] = ([t], -int(child["s"]))  # the deeper look at this move
            elif key == own:
                by_position[key] = ([t], entry["score"])
        if own in by_position:
            by_position[own][0].sort(key=lambda t: t != entry["move"])
        return sorted(by_position.values(), key=lambda mv: -mv[1])

    def choices(self, game: "GameProtocol") -> list[tuple[MoveTuple, int]]:
        """Every move the book can judge here, best first, with its value to the mover: the
        book's own move, and any other move leading to a position the book has an entry for
        (searched from the opponent's side, so worth minus that entry's score). Moves that
        lead to images of the same position are listed once. Empty if the position itself is
        not in the book. Makes and unmakes moves on `game`."""
        return [(moves[0], value) for moves, value in self._options(game)]

    def pick(self, game: "GameProtocol", margin: int, rng) -> dict | None:
        """A book move for this position: with margin > 0, a random one of the moves the book
        rates within `margin` of its best here, played in a random one of its mirror images
        (so repeated games vary); otherwise the book's own move. None if the position is not
        in the book."""
        entry = self.lookup(game)
        if entry is None or margin <= 0:
            return entry
        options = self._options(game)
        viable = [(ms, v) for ms, v in options if v >= options[0][1] - margin]
        moves, value = rng.choice(viable)
        return {"move": rng.choice(moves), "score": value, "depth": entry["depth"], "choices": len(viable)}

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
