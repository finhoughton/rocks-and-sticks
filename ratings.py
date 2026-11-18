"""Ratings for bots and people, on the Elo scale, from every recorded game.

Each finished game is one line in ratings.jsonl. Ratings are not updated game by game as in
classic Elo; they are fitted to all recorded games at once (a Bradley-Terry model), which
gives the same scale but makes the order of games irrelevant and gives each rating an
honest uncertainty. The model:

    P(player 1 beats player 2) = 1 / (1 + 10 ** -((R1 - R2 + A) / 400))

where A is the advantage of moving first, fitted along with the ratings. A 200-point gap
means the stronger side scores about 76%, 400 points about 91%. Unfinished games count as
half a win each.

Two adjustments keep small samples sensible:
  * every pair of players that has met is credited with one extra drawn game, so a 5-0
    record means "clearly stronger" (about +420) rather than "infinitely stronger";
  * the scale is fixed by one reference player, the old depth-3 alpha-beta (ab:3) at 1000.
    Players with no chain of games linking them to it are placed relative to each other only.

Recording: main.py records every game it plays between two differently named players;
tools/rate.py shows the table and takes results entered by hand.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from players.specs import canonical, describe, is_bot_spec, version

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(ROOT, "ratings.jsonl")


def results_path(path: str | None = None) -> str:
    """The results file: an explicit path, else $RS_RATINGS_FILE, else ratings.jsonl."""
    return path or os.environ.get("RS_RATINGS_FILE") or DEFAULT_PATH
ANCHOR = "ab:3"
ANCHOR_ELO = 1000.0
VIRTUAL_DRAWS = 1.0
ADVANTAGE_PRIOR_ELO = 100.0  # prior standard deviation of the first-move advantage

ELO_PER_LOGIT = 400.0 / math.log(10.0)


def player_id(name: str) -> str:
    """Canonical id: bot specs are canonicalised, anything else is a person's name."""
    name = name.strip()
    return canonical(name) if is_bot_spec(name) else name


def record_game(
    p1: str,
    p2: str,
    winner: int | None,
    *,
    plies: int | None = None,
    source: str = "manual",
    note: str | None = None,
    path: str | None = None,
) -> dict:
    """Append one finished game. winner: 0 if p1 won, 1 if p2 won, None if unfinished."""
    if winner not in (0, 1, None):
        raise ValueError("winner must be 0, 1 or None")
    a, b = player_id(p1), player_id(p2)
    if a == b:
        raise ValueError("a player cannot play itself in a rated game")
    rec = {
        "p1": a,
        "p2": b,
        "v1": version(a) if is_bot_spec(a) else None,
        "v2": version(b) if is_bot_spec(b) else None,
        "winner": winner,
        "plies": plies,
        "source": source,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if note:
        rec["note"] = note
    with open(results_path(path), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def load_games(path: str | None = None) -> list[dict]:
    path = results_path(path)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(json.loads(line))
    return out


def rating_key(pid: str, ver: str | None) -> str:
    """Players are rated per engine version. Games recorded under an engine version that is no
    longer current belong to that frozen version, named like rs:100k:v1."""
    if ver is not None and is_bot_spec(pid) and ver != version(pid):
        return canonical(f"{pid}:v{ver}") if pid.startswith("rs:") else f"{pid} [v{ver}]"
    return pid


@dataclass
class Rating:
    name: str
    elo: float
    ci95: float          # half-width of the 95% interval; 0 for the reference player
    games: int
    wins: int
    losses: int
    unfinished: int
    linked: bool         # connected to the reference player by a chain of games

    @property
    def score(self) -> float:
        return (self.wins + 0.5 * self.unfinished) / max(1, self.games)

    @property
    def description(self) -> str:
        return describe(self.name.split(" [")[0])


@dataclass
class Fit:
    ratings: list[Rating]
    advantage: float      # first-move advantage in Elo
    advantage_ci95: float
    anchor: str | None    # None if the reference player has no games
    n_games: int


def _components(n: int, pairs: set[tuple[int, int]]) -> list[int]:
    comp = list(range(n))

    def find(i: int) -> int:
        while comp[i] != i:
            comp[i] = comp[comp[i]]
            i = comp[i]
        return i

    for i, j in pairs:
        comp[find(i)] = find(j)
    return [find(i) for i in range(n)]


def fit(games: list[dict], anchor: str = ANCHOR, anchor_elo: float = ANCHOR_ELO, virtual_draws: float = VIRTUAL_DRAWS) -> Fit:
    names: list[str] = []
    index: dict[str, int] = {}

    def idx(name: str) -> int:
        if name not in index:
            index[name] = len(names)
            names.append(name)
        return index[name]

    rows = []  # (i, j, score for i, uses first-move advantage, weight)
    tally: dict[int, list[int]] = {}
    pairs: set[tuple[int, int]] = set()
    for g in games:
        i = idx(rating_key(g["p1"], g.get("v1")))
        j = idx(rating_key(g["p2"], g.get("v2")))
        w = g.get("winner")
        s = 0.5 if w is None else (1.0 if w == 0 else 0.0)
        rows.append((i, j, s, 1.0, 1.0))
        pairs.add((min(i, j), max(i, j)))
        for k, res in ((i, s), (j, 1.0 - s)):
            t = tally.setdefault(k, [0, 0, 0])
            t[0 if res == 1.0 else 1 if res == 0.0 else 2] += 1
    n = len(names)
    if n == 0:
        return Fit([], 0.0, 0.0, None, 0)
    for i, j in pairs:
        if virtual_draws > 0:
            rows.append((i, j, 0.5, 0.0, virtual_draws))

    # Fix the scale: the reference player in its group, the first player in each other group.
    comps = _components(n, pairs)
    anchor_i = index.get(rating_key(anchor, version(anchor) if is_bot_spec(anchor) else None))
    fixed: dict[int, int] = {}  # component -> fixed player
    if anchor_i is not None:
        fixed[comps[anchor_i]] = anchor_i
    for k in range(n):
        fixed.setdefault(comps[k], k)
    fixed_players = set(fixed.values())
    free = [k for k in range(n) if k not in fixed_players]
    col = {k: c for c, k in enumerate(free)}
    m = len(free) + 1  # last column: first-move advantage
    a_col = m - 1

    I = np.array([r[0] for r in rows])
    J = np.array([r[1] for r in rows])
    S = np.array([r[2] for r in rows])
    ADV = np.array([r[3] for r in rows])
    W = np.array([r[4] for r in rows])
    prior_prec = (ELO_PER_LOGIT / ADVANTAGE_PRIOR_ELO) ** 2

    x = np.zeros(m)

    def ratings_of(x: np.ndarray) -> np.ndarray:
        r = np.zeros(n)
        for k, c in col.items():
            r[k] = x[c]
        return r

    def objective(x: np.ndarray) -> float:
        r = ratings_of(x)
        d = r[I] - r[J] + ADV * x[a_col]
        return float(np.sum(W * (S * np.logaddexp(0, -d) + (1 - S) * np.logaddexp(0, d))) + 0.5 * prior_prec * x[a_col] ** 2)

    def grad_hess(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r = ratings_of(x)
        d = r[I] - r[J] + ADV * x[a_col]
        p = 1.0 / (1.0 + np.exp(-d))
        g1 = W * (p - S)        # d objective / d d
        h1 = W * p * (1 - p)    # d2 objective / d d2
        grad = np.zeros(m)
        hess = np.zeros((m, m))
        for t in range(len(rows)):
            ci = col.get(int(I[t]))
            cj = col.get(int(J[t]))
            coef = [(ci, 1.0), (cj, -1.0), (a_col if ADV[t] else None, 1.0)]
            coef = [(c, v) for c, v in coef if c is not None]
            for c, v in coef:
                grad[c] += g1[t] * v
                for c2, v2 in coef:
                    hess[c, c2] += h1[t] * v * v2
        grad[a_col] += prior_prec * x[a_col]
        hess[a_col, a_col] += prior_prec
        return grad, hess

    f = objective(x)
    for _ in range(100):
        g, h = grad_hess(x)
        step = np.linalg.solve(h, g)
        t = 1.0
        while t > 1e-6:
            x_new = x - t * step
            f_new = objective(x_new)
            if f_new <= f + 1e-12:
                break
            t *= 0.5
        x, f_prev, f = x_new, f, f_new
        if np.max(np.abs(t * step)) < 1e-9 or abs(f_prev - f) < 1e-12:
            break
    _, h = grad_hess(x)
    cov = np.linalg.inv(h)
    r = ratings_of(x)

    # Express each group on the Elo scale: the anchor's group relative to the anchor,
    # other groups relative to their own mean.
    members: dict[int, list[int]] = {}
    for k in range(n):
        members.setdefault(comps[k], []).append(k)
    out = []
    for k in range(n):
        c = comps[k]
        linked = anchor_i is not None and c == comps[anchor_i]
        v = np.zeros(m)
        if k in col:
            v[col[k]] += 1.0
        if linked:
            centre = r[anchor_i]
        else:
            group = members[c]
            centre = float(np.mean(r[group]))
            for q in group:
                if q in col:
                    v[col[q]] -= 1.0 / len(group)
        sd = math.sqrt(max(0.0, float(v @ cov @ v)))
        t = tally.get(k, [0, 0, 0])
        out.append(
            Rating(
                name=names[k],
                elo=anchor_elo + (r[k] - centre) * ELO_PER_LOGIT,
                ci95=1.96 * sd * ELO_PER_LOGIT,
                games=sum(t),
                wins=t[0],
                losses=t[1],
                unfinished=t[2],
                linked=linked,
            )
        )
    out.sort(key=lambda q: (not q.linked, -q.elo))
    adv_sd = math.sqrt(float(cov[a_col, a_col]))
    return Fit(out, x[a_col] * ELO_PER_LOGIT, 1.96 * adv_sd * ELO_PER_LOGIT, names[anchor_i] if anchor_i is not None else None, len(games))


def expected_score(elo_a: float, elo_b: float) -> float:
    """Expected score of A against B, ignoring who moves first."""
    return 1.0 / (1.0 + 10.0 ** (-(elo_a - elo_b) / 400.0))
