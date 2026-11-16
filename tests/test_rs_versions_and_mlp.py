"""Frozen engine versions for rated comparisons, and the network evaluation."""

from __future__ import annotations

import importlib.util
import random

import numpy as np
import pytest

rs_engine = pytest.importorskip("rs_engine")

from players.rs_player import ENGINE_VERSION  # noqa: E402
from players.specs import canonical, make_bot, version  # noqa: E402


def test_version_tags_name_frozen_engines() -> None:
    assert canonical(f"rs:100k:v{ENGINE_VERSION}") == "rs:100k"
    assert canonical("rs:100k:v1") == "rs:100k:v1"
    assert version("rs:100k:v1") == "1" and version("rs:100k") == ENGINE_VERSION


@pytest.mark.skipif(importlib.util.find_spec("rs_engine_v1") is None, reason="frozen v1 not built")
def test_a_frozen_version_plays_with_its_own_engine_and_book() -> None:
    bot = make_bot("rs:2000:v1", 0, seed=1)
    assert type(bot.engine).__module__ == "rs_engine_v1"
    assert bot.book is not None and bot.book.format == 1
    current = make_bot("rs:2000", 0, seed=1)
    assert type(current.engine).__module__ == "rs_engine"


def _write_mlp(path, sizes, rng) -> list[tuple[np.ndarray, np.ndarray]]:
    layers = []
    with open(path, "w") as fh:
        fh.write(f"mlp {len(sizes)} {' '.join(map(str, sizes))}\n")
        for a, b in zip(sizes[:-1], sizes[1:]):
            w = rng.normal(0, 0.3, size=(a, b)).astype(np.float32)
            bias = rng.normal(0, 0.1, size=b).astype(np.float32)
            layers.append((w, bias))
            fh.write(" ".join(f"{v:.7g}" for v in w.ravel()) + "\n")
            fh.write(" ".join(f"{v:.7g}" for v in bias) + "\n")
    return layers


def test_network_evaluation_matches_a_reference_and_search_uses_it(tmp_path) -> None:
    rng = np.random.default_rng(0)
    n = rs_engine.N_FEATS
    path = tmp_path / "net.txt"
    layers = _write_mlp(path, [n, 16, 8, 1], rng)
    eng = rs_engine.Engine(14, 1)
    assert eng.load_weights(str(path)) and eng.uses_mlp()

    def reference(f):
        a = np.asarray(f, dtype=np.float64)
        for i, (w, b) in enumerate(layers):
            a = a @ w + b
            if i < len(layers) - 1:
                a = np.maximum(a, 0)
        return float(a[0]) * 200

    r = random.Random(1)
    for _ in range(30):
        eng.reset()
        for _ in range(r.randrange(0, 12)):
            if eng.winner >= 0:
                break
            eng.do_move(*r.choice(eng.legal_moves(True)))
        if eng.winner >= 0:
            continue
        f = eng.features()
        want = max(-20000, min(20000, reference(f)))
        assert abs(eng.mlp_eval(f) - want) <= 1.0
        # the search's cached evaluation (board features cached by board) is the same thing
        assert eng.eval() == eng.mlp_eval(f)
    eng.reset()
    r = eng.search(0, 4, 0)
    assert r["depth"] == 4 and eng.is_legal(*r["move"])


def test_linear_weights_files_still_load(tmp_path) -> None:
    eng = rs_engine.Engine(12, 1)
    w = eng.get_weights()
    path = tmp_path / "w.txt"
    path.write_text(" ".join(map(str, w)))
    assert eng.load_weights(str(path)) and not eng.uses_mlp()
    bad = tmp_path / "bad.txt"
    bad.write_text("mlp 3 5 4 1\n1 2 3")
    assert not eng.load_weights(str(bad))


def test_a_blend_file_averages_the_linear_and_network_evaluations(tmp_path) -> None:
    rng = np.random.default_rng(1)
    n = rs_engine.N_FEATS
    net = tmp_path / "net.txt"
    _write_mlp(net, [n, 8, 1], rng)
    lin = rs_engine.Engine(12, 1)
    weights = lin.get_weights()
    blend = tmp_path / "blend.txt"
    blend.write_text("blend 0.25\n" + " ".join(map(str, weights)) + "\n" + net.read_text())
    a, b = rs_engine.Engine(12, 1), rs_engine.Engine(12, 1)
    assert a.load_weights(str(net)) and b.load_weights(str(blend)) and b.uses_mlp()
    r = random.Random(2)
    for _ in range(20):
        for e in (lin, a, b):
            e.reset()
        for _ in range(r.randrange(1, 10)):
            if lin.winner >= 0:
                break
            m = r.choice(lin.legal_moves(True))
            for e in (lin, a, b):
                e.do_move(*m)
        if lin.winner >= 0:
            continue
        assert abs(b.eval() - (0.25 * a.eval() + 0.75 * lin.eval())) <= 1.0
