"""The rating fit (ratings.py) and bot specs (players/specs.py)."""

from __future__ import annotations

import random

import pytest

import ratings
from players.specs import canonical, describe, is_bot_spec


def _games(true: dict[str, float], adv: float, n_per_pair: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    names = list(true)
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            for g in range(n_per_pair):
                p1, p2 = (a, b) if g % 2 == 0 else (b, a)
                p = ratings.expected_score(true[p1] + adv, true[p2])
                out.append({"p1": p1, "p2": p2, "winner": 0 if rng.random() < p else 1})
    return out


def _by_name(fit: ratings.Fit) -> dict[str, ratings.Rating]:
    return {r.name: r for r in fit.ratings}


def test_recovers_known_ratings_and_first_move_advantage() -> None:
    true = {"ab:3": 1000.0, "alice": 1200.0, "bob": 1450.0, "carol": 900.0}
    fit = ratings.fit(_games(true, adv=60.0, n_per_pair=400))
    got = _by_name(fit)
    assert got["ab:3"].elo == pytest.approx(1000.0) and got["ab:3"].ci95 == 0.0
    for name, elo in true.items():
        r = got[name]
        assert abs(r.elo - elo) <= max(r.ci95, 1e-9) + 15, (name, r.elo, elo, r.ci95)
        assert r.linked
    assert abs(fit.advantage - 60.0) <= fit.advantage_ci95 + 10


def test_a_clean_sweep_gives_a_finite_gap() -> None:
    games = [{"p1": "ab:3" if g % 2 else "you", "p2": "you" if g % 2 else "ab:3", "winner": 1 if g % 2 else 0} for g in range(5)]
    # "you" won all five; with one virtual draw that is 5.5 / 6
    got = _by_name(ratings.fit(games))
    expected_gap = 400 * __import__("math").log10(5.5 / 0.5)
    assert got["you"].elo - got["ab:3"].elo == pytest.approx(expected_gap, abs=25)
    assert got["you"].ci95 > 200  # five games say little


def test_unlinked_groups_are_flagged_and_centred() -> None:
    games = _games({"ab:3": 1000.0, "x": 1100.0}, 0.0, 10) + _games({"p": 0.0, "q": 300.0}, 0.0, 10, seed=1)
    got = _by_name(ratings.fit(games))
    assert got["x"].linked and not got["p"].linked and not got["q"].linked
    assert (got["p"].elo + got["q"].elo) / 2 == pytest.approx(ratings.ANCHOR_ELO, abs=1e-6)
    assert got["q"].elo > got["p"].elo


def test_old_engine_versions_are_rated_separately(tmp_path) -> None:
    path = str(tmp_path / "r.jsonl")
    ratings.record_game("rs:100k", "ab:3", 0, path=path)
    old = {"p1": "rs:100k", "p2": "ab:3", "v1": "0", "v2": None, "winner": 1}
    # a game recorded by a frozen version names it explicitly; both belong to the same player
    frozen = {"p1": "rs:100k:v0", "p2": "ab:3", "v1": "0", "v2": None, "winner": 0}
    games = ratings.load_games(path) + [old, frozen]
    got = _by_name(ratings.fit(games))
    assert set(got) == {"rs:100k", "rs:100k:v0", "ab:3"}
    assert got["rs:100k:v0"].games == 2


def test_record_canonicalises_and_validates(tmp_path) -> None:
    path = str(tmp_path / "r.jsonl")
    rec = ratings.record_game("rs:0.5s:1t", "you", None, path=path)
    assert rec["p1"] == "rs:500ms" and rec["p2"] == "you" and rec["winner"] is None
    with pytest.raises(ValueError):
        ratings.record_game("rs:500ms", "rs:0.5s", 0, path=path)
    with pytest.raises(ValueError):
        ratings.record_game("a", "b", 2, path=path)


def test_specs() -> None:
    assert canonical("rs:5.0s:4t") == "rs:5s:4t"
    assert canonical("rs:100000") == "rs:100k" and canonical("rs:1000000:1t") == "rs:1M"
    assert canonical("rs:1500ms") == "rs:1.5s"
    for bad in ("rs", "rs:5m", "ab:0", "rs:5s:fast", "you"):
        assert not is_bot_spec(bad)
    assert describe("rs:5s:4t") == "rs engine, 5 s/move, 4 threads"
    assert describe("you") == "human"
