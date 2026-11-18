"""Player backed by the rs_engine C++ alpha-beta engine.

The engine keeps no game state between moves: each call copies the position out of the
Python ``Game`` (the rules reference), so it can never drift from what the human sees.
"""

from __future__ import annotations

import atexit
import importlib
import os
import random
import threading
import weakref
from typing import TYPE_CHECKING

from models import PASS, D, Move

from .base import Player

try:
    import rs_engine
except Exception as e:  # pragma: no cover - build hint
    raise ImportError("rs_engine extension not available; build it with ./build_engine.sh") from e

if TYPE_CHECKING:
    from game import GameProtocol

MATE_BOUND = 29000  # scores at or above this are forced wins (see engine_src/rs_search.hpp)

# Bump when a change to the engine, its weights or the book code changes how it plays: the
# rating system (ratings.py) rates each version separately. Growth of the book itself does not
# count. Before bumping, freeze the current version so it can still play rated games:
# copy the engine sources, the evaluation file it plays with (as weights.txt) and book.json to
# engine_src/versions/v<old>/ and run ./build_engine.sh (bots then name it with ":v<old>").
ENGINE_VERSION = "7"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Two evaluations (both read by Engine.load_weights; the file says which kind it is):
#  * LINEAR_WEIGHTS: a weighted sum of the features (tools/tune_rs_eval.py);
#  * BLEND_WEIGHTS: the average of that and a small network (tools/train_rs_mlp.py).
# Measured against the linear one on the rating ladder (Nov 2026), the blend is much stronger
# in short searches (+120 Elo at 100k positions per move, +135 at 300k) but weaker in long
# ones (-24 at 600k, -63 at 1M): the network learned what a short search sees, and that stops
# helping, then hurts, once the search itself looks further. So the engine picks by budget.
LINEAR_WEIGHTS = os.path.join(_ROOT, "engine_src", "weights.txt")
BLEND_WEIGHTS = os.path.join(_ROOT, "engine_src", "eval.txt")
# In the opening book the engine picks at random among the moves the book rates within this
# much of its best (200 = from an even game to about 27%), so games do not all open the same
# way. Frozen versions before 7 always play the book's first choice.
BOOK_VARIETY = 40
DEEP_SEARCH_NODES = 400_000   # from about here on the linear evaluation plays better
NODES_PER_MS = 600            # rough positions per millisecond of a timed search, one thread
# Long analysis and book building search deep: they use the linear evaluation.
DEFAULT_WEIGHTS = LINEAR_WEIGHTS


def default_weights(time_ms: int = 0, max_nodes: int = 0, threads: int = 1, folder: str | None = None) -> str:
    """The evaluation the engine plays best with for this search budget (from `folder` for a
    frozen version that has both kinds)."""
    linear = os.path.join(folder, "weights.txt") if folder else LINEAR_WEIGHTS
    blend = os.path.join(folder, "eval.txt") if folder else BLEND_WEIGHTS
    if max_nodes > 0:
        nodes = max_nodes
    elif time_ms > 0:
        # extra threads deepen a search less than they multiply its positions
        nodes = time_ms * NODES_PER_MS * (1 + 0.5 * (max(1, threads) - 1))
    else:
        return linear  # depth-limited or unlimited: assume a deep search
    return blend if nodes < DEEP_SEARCH_NODES else linear


def version_files(version: str | None) -> tuple[object, str | None, str | None]:
    """(engine module, folder, book path) for an engine version; None means current. A frozen
    version's folder holds weights.txt (and, from version 6, eval.txt: see default_weights)."""
    if version is None or str(version) == ENGINE_VERSION:
        return rs_engine, None, None
    folder = os.path.join(_ROOT, "engine_src", "versions", f"v{version}")
    if not os.path.isdir(folder):
        raise ValueError(f"no frozen engine version {version} (expected {folder})")
    try:
        module = importlib.import_module(f"rs_engine_v{version}")
    except ImportError as e:
        raise ImportError(f"engine version {version} is not built; run ./build_engine.sh") from e
    return module, folder, os.path.join(folder, "book.json")


def sync_engine(engine: "rs_engine.Engine", game: "GameProtocol") -> None:
    """Copy the position of a Python game into an engine."""
    sticks = [(s.start.x, s.start.y, s.d.name) for s in game.sticks]
    rocks = [(r.x, r.y, r.rocked_by.number) for r in game.rocks if r.rocked_by is not None]
    engine.set_position(
        sticks,
        rocks,
        int(game.current_player),
        int(game.turn_number),
        [int(game.players_scores[0]), int(game.players_scores[1])],
        [int(game.num_rocks[0]), int(game.num_rocks[1])],
        -1 if game.winner is None else int(game.winner),
    )


def to_py_move(mv: tuple[int, int, str]) -> Move:
    x, y, t = mv
    return PASS if t == "P" else Move(int(x), int(y), str(t))


class RSPlayer(Player):
    def __init__(
        self,
        player_number: int,
        *,
        time_ms: int = 3000,
        max_depth: int = 0,
        max_nodes: int = 0,
        weights: str | None = None,
        tt_bits: int = 22,
        threads: int = 1,
        root_margin: int = 0,
        seed: int | None = None,
        verbose: bool = False,
        book: str | None = None,
        version: str | None = None,
        ponder: bool = False,
        ponder_ms: int = 600_000,
        book_variety: int | None = None,
    ) -> None:
        super().__init__(player_number)
        module, version_folder, version_book = version_files(version)
        self.version = ENGINE_VERSION if version is None else str(version)
        self.engine = module.Engine(int(tt_bits), max(1, int(threads)))
        if weights is not None:
            path = weights
        elif version_folder is not None and not os.path.exists(os.path.join(version_folder, "eval.txt")):
            path = os.path.join(version_folder, "weights.txt")
        else:
            path = default_weights(int(time_ms), int(max_nodes), int(threads), version_folder)
        self.weights_path = path
        if path and os.path.exists(path):
            if not self.engine.load_weights(path):
                raise RuntimeError(f"could not load engine weights from {path}")
        self.time_ms = int(time_ms)
        self.max_depth = int(max_depth)
        self.max_nodes = int(max_nodes)
        self.verbose = bool(verbose)
        if root_margin:
            self.engine.set_root_margin(int(root_margin))
        if seed is not None:
            self.engine.seed(int(seed))
        # Opening book (players/rs_book.py): pass a path, or "" to play without one. A frozen
        # older version uses the book it was frozen with.
        from .rs_book import DEFAULT_BOOK, Book

        if book is None:
            book = version_book or DEFAULT_BOOK
        self.book = Book(book) if book != "" else None
        if book_variety is None:
            book_variety = BOOK_VARIETY if version_folder is None else 0
        self.book_variety = int(book_variety)
        self._book_rng = random.Random(seed)
        self.last_info: dict = {}
        self._win_announced_at = -10  # len(game.moves) when a forced win was last announced
        # Pondering: after moving, keep searching the position the opponent faces until they
        # move. The search fills the transposition table with the positions after each of
        # their replies, so the next search starts with much of its work done.
        self.ponder = bool(ponder)
        self.ponder_ms = int(ponder_ms)
        if self.ponder and not hasattr(self.engine, "stop"):
            raise ValueError(f"engine version {self.version} cannot ponder")
        self._ponder_thread: threading.Thread | None = None
        _live_players.add(self)

    def get_move(self, game: "GameProtocol") -> Move:
        self.stop_pondering()
        mv = self._choose_move(game)
        if self.ponder:
            self._start_pondering(game, mv)
        return mv

    def stop_pondering(self) -> None:
        """Stop a background search, if one is running (safe to call at any time)."""
        t = self._ponder_thread
        if t is not None:
            self.engine.stop()
            t.join()
            self._ponder_thread = None

    def _start_pondering(self, game: "GameProtocol", mv: Move) -> None:
        sync_engine(self.engine, game)
        self.engine.do_move(*_engine_move(mv))
        if self.engine.winner >= 0:
            return
        # cleared here, not in the thread: a stop sent before the search begins must not be lost
        self.engine.clear_stop()
        self._ponder_thread = threading.Thread(target=self.engine.search, args=(self.ponder_ms, 0, 0, True), daemon=True)
        self._ponder_thread.start()

    def _choose_move(self, game: "GameProtocol") -> Move:
        if self.book is not None and len(self.book):
            # (the plain Python game: choosing among book moves makes and unmakes moves)
            entry = self.book.pick(getattr(game, "py", game), self.book_variety, self._book_rng)
            if entry is not None:
                mv = to_py_move(entry["move"])
                if game.valid_move(mv, self.number):
                    self.last_info = {"move": entry["move"], "score": entry["score"], "depth": entry["depth"], "nodes": 0, "pv": [entry["move"]], "book": True}
                    self._win_announced_at = -10
                    if self.verbose:
                        from .rs_display import describe

                        print(f"[engine] book move, eval {describe(entry['score'], self.number).text}   depth {entry['depth']}")
                    return mv
        sync_engine(self.engine, game)
        info = self.engine.search(self.time_ms, self.max_depth, self.max_nodes)
        # If the previous move announced a forced win, the win is still there; a search that
        # fails to find it again has only run out of budget, so keep looking a while longer.
        n_moves = len(game.moves)
        if self._win_announced_at == n_moves - 2 and info["score"] < MATE_BOUND and self.max_depth == 0:
            for _ in range(3):
                info = self.engine.search(self.time_ms * 2, 0, self.max_nodes * 4)
                if info["score"] >= MATE_BOUND:
                    break
        self._win_announced_at = n_moves if info["score"] >= MATE_BOUND else -10
        self.last_info = info
        mv = to_py_move(info["move"])
        if not game.valid_move(mv, self.number):
            # Should not happen (the parity tests cover it); never hand back an illegal move.
            raise RuntimeError(f"engine produced an illegal move {mv}")
        if self.verbose:
            from .rs_display import describe

            pv = " ".join("P" if t == "P" else f"{x},{y}{t}" for x, y, t in info["pv"])
            print(f"[engine] eval {describe(info['score'], self.number).text}   depth {info['depth']}   line: {pv}")
        return mv

    def analyse(self, game: "GameProtocol", time_ms: int | None = None) -> dict:
        """Search the current position (whoever is to move) and return the engine's info dict,
        with the mover added. Does not need to be this player's turn."""
        self.stop_pondering()
        sync_engine(self.engine, game)
        info = dict(self.engine.search(self.time_ms if time_ms is None else int(time_ms), self.max_depth, self.max_nodes))
        info["mover"] = int(game.current_player)
        return info


def _engine_move(mv: Move) -> tuple[int, int, str]:
    return (0, 0, "P") if mv is PASS else (int(mv.c[0]), int(mv.c[1]), str(mv.t))


# Background searches must end before the interpreter does.
_live_players: "weakref.WeakSet[RSPlayer]" = weakref.WeakSet()


@atexit.register
def _stop_all_pondering() -> None:
    for p in list(_live_players):
        p.stop_pondering()


__all__ = ["RSPlayer", "sync_engine", "to_py_move", "D"]
