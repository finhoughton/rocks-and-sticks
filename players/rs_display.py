"""Turn the rs engine's raw search score into something readable.

The raw score is in arbitrary units from the mover's point of view. Here it becomes:

  * a win chance, calibrated against the results of self-play games
    (tools/calibrate_rs_eval.py prints the constants below);
  * a chess-style number from Player 1's point of view: positive favours Player 1, negative
    favours Player 2. It is scaled so that a given number means the same win chance as that
    many pawns does on a chess analysis board (+1.0 is about 59%, +3.0 about 75%);
  * for forced results, "M3"-style: the winner and how many of their own moves it takes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

MATE = 30000
MATE_BOUND = 29000

# P(mover wins) = sigmoid(SLOPE * score + OFFSET[mover]); fitted by tools/calibrate_rs_eval.py on
# 600 engine games (100k positions/move) from sound openings (data/rs/calib_sound.bin). Earlier
# data that began with random moves gave a strongly side-biased fit, so calibrate on real openings.
SLOPE = 0.005738
OFFSET = (-0.153, -0.225)

# Win chance = sigmoid(PAWN_LOGIT * pawns): the curve chess analysis boards use.
PAWN_LOGIT = 0.368
MAX_PAWNS = 9.9


@dataclass(frozen=True)
class Eval:
    p1_chance: float          # Player 1's estimated chance of winning, 0..1
    pawns: float | None       # chess-style number for Player 1; None when the result is forced
    winner: int | None        # set when a forced win has been found (0 or 1)
    moves_to_win: int | None  # the winner's own moves until the game ends
    short: str                # e.g. "+1.3", "-0.4", "+M3", "-M2"
    text: str                 # e.g. "+1.3  (Player 1 62%)", "Player 2 wins in 2"


def describe(score: int, mover: int) -> Eval:
    """Describe a search score that was computed with `mover` (0 or 1) to move."""
    if abs(score) >= MATE_BOUND:
        winner = mover if score > 0 else 1 - mover
        plies = max(1, MATE - abs(score))
        # the winner's share of the remaining plies, which alternate starting with the mover
        moves = max(1, (plies + (1 if winner == mover else 0)) // 2)
        sign = "+" if winner == 0 else "-"
        return Eval(
            p1_chance=1.0 if winner == 0 else 0.0,
            pawns=None,
            winner=winner,
            moves_to_win=moves,
            short=f"{sign}M{moves}",
            text=f"Player {winner + 1} wins in {moves}",
        )
    logit = SLOPE * score + OFFSET[mover]
    if mover == 1:
        logit = -logit
    p1 = 1.0 / (1.0 + math.exp(-logit))
    pawns = max(-MAX_PAWNS, min(MAX_PAWNS, logit / PAWN_LOGIT))
    leader = 1 if p1 >= 0.5 else 2
    pct = round(100 * max(p1, 1 - p1))
    short = f"{pawns:+.1f}"
    return Eval(p1_chance=p1, pawns=pawns, winner=None, moves_to_win=None, short=short, text=f"{short}  (Player {leader} {pct}%)")


def final(winner: int | None) -> Eval:
    """The evaluation of a finished game."""
    if winner is None:
        return Eval(0.5, 0.0, None, None, "0.0", "no winner")
    return Eval(1.0 if winner == 0 else 0.0, None, winner, 0, "1-0" if winner == 0 else "0-1", f"Player {winner + 1} won")


def draw_eval_bar(fig, ev: Eval | None) -> None:
    """Draw (or update) a chess-style evaluation bar on the right edge of a game figure.

    Player 1's colour fills from the bottom in proportion to its win chance.
    """
    ax = getattr(fig, "_rs_eval_ax", None)
    if ax is None or ax not in fig.axes:
        ax = fig.add_axes((0.925, 0.24, 0.028, 0.62))
        fig._rs_eval_ax = ax
    ax.clear()
    p1 = 0.5 if ev is None else ev.p1_chance
    ax.bar([0], [p1], width=1.0, color="C0", bottom=0.0)
    ax.bar([0], [1.0 - p1], width=1.0, color="C1", bottom=p1)
    ax.axhline(0.5, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("" if ev is None else ev.short, fontsize=10)
    ax.set_xlabel("P1", fontsize=8, color="C0")
