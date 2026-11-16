"""Fit the rs engine's evaluation weights to self-play results (logistic regression).

    engine_src/rs_selfplay gen --games 12000 --nodes 20000 --out data/rs/gen0.bin
    python3 tools/tune_rs_eval.py data/rs/gen0.bin --out engine_src/weights.txt

Each record is a position: the features from the side to move's point of view and the final
result of the game for that side. Positions where a scoring closure is on the board are
skipped, because the search never evaluates those statically.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rs_engine  # noqa: E402

N_FEATS = rs_engine.N_FEATS
SCALE = 200.0  # eval units per logit: 200 points ~ 73% expected score
N_SHAPE = 45
N_PAIR = 9
F_PAIR_MINE = N_SHAPE
F_PAIR_THEIRS = N_SHAPE + N_PAIR
# Never seen in quiet positions (they are immediate tactics); keep the hand-set values.
FIXED = {F_PAIR_MINE + 0: 900, F_PAIR_THEIRS + 0: -700}


def load(paths: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    width = N_FEATS + 5
    chunks = []
    for p in paths:
        raw = np.fromfile(p, dtype=np.int16)
        chunks.append(raw[: (raw.size // width) * width].reshape(-1, width))
    rec = np.concatenate(chunks)
    result = rec[:, 0].astype(np.float32) / 2.0
    score = rec[:, 1].astype(np.float32)
    tactical = rec[:, 2] != 0
    ply = rec[:, 3]
    feats = rec[:, 5:].astype(np.float32)
    return feats, result, score, tactical & True, ply  # type: ignore[return-value]


def fit(x: np.ndarray, y: np.ndarray, l2: float, iters: int, init: np.ndarray) -> np.ndarray:
    """Minimise mean cross-entropy + l2 * |theta|^2 with full-batch Adam."""
    theta = init.astype(np.float64).copy()
    free = np.array([i not in FIXED for i in range(x.shape[1])])
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    lr, b1, b2 = 0.02, 0.9, 0.999
    xd = x.astype(np.float64)
    for t in range(1, iters + 1):
        z = xd @ theta
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = xd.T @ (p - y) / len(y) + 2 * l2 * theta
        g[~free] = 0.0
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta -= lr * (m / (1 - b1**t)) / (np.sqrt(v / (1 - b2**t)) + 1e-8)
        if t % 500 == 0 or t == iters:
            loss = -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            print(f"  iter {t:5d} loss {loss:.5f}")
    return theta


def logloss(x: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    p = 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64) @ theta, -30, 30)))
    return float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    ap.add_argument("--out", default="engine_src/weights.txt")
    ap.add_argument("--l2", type=float, default=2e-4)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--min-ply", type=int, default=0)
    ap.add_argument("--lam", type=float, default=1.0,
                    help="weight of the game result in the target; the rest is the search score at the position as a win chance")
    args = ap.parse_args()

    feats, result, score, tactical, ply = load(args.data)
    # leaf records (rs_selfplay --leaves: random-walk positions, result -1) have only a search
    # score; they may have closures on the board, which the search does evaluate
    leaf = result < 0
    keep = (leaf | ~tactical) & (ply >= args.min_ply) & (np.abs(score) < 29000)
    x = feats[keep]
    soft = 1 / (1 + np.exp(-score[keep] / SCALE))
    y = np.where(leaf[keep], soft, args.lam * np.maximum(result[keep], 0) + (1 - args.lam) * soft).astype(np.float32)
    print(f"positions: {len(feats)} total, {len(x)} quiet; mean result for side to move {y.mean():.3f}")

    # hold out by position index blocks (games are contiguous in the file)
    n = len(x)
    block = (np.arange(n) // 200) % 10 == 0
    xtr, ytr, xva, yva = x[~block], y[~block], x[block], y[block]

    eng = rs_engine.Engine(10)
    w0 = np.array(eng.get_weights(), dtype=np.float64)
    theta0 = w0 / SCALE
    for i, val in FIXED.items():
        theta0[i] = val / SCALE
    base = float(-np.mean(yva * np.log(yva.mean()) + (1 - yva) * np.log(1 - yva.mean())))
    print(f"validation logloss: constant {base:.5f}, current weights {logloss(xva, yva, theta0):.5f}")

    theta = fit(xtr, ytr, args.l2, args.iters, theta0)
    print(f"validation logloss: fitted {logloss(xva, yva, theta):.5f}  (train {logloss(xtr, ytr, theta):.5f})")

    w = np.rint(theta * SCALE).astype(int)
    for i, val in FIXED.items():
        w[i] = val
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(" ".join(str(int(v)) for v in w) + "\n")
    print(f"wrote {args.out}")

    names = ["excl_score", "armed_good", "armed_good_kill", "armed_weak", "armed_weak_kill", "pair_free", "pot_good", "pot_good_kill", "pot_weak"]
    cnt = np.abs(x).sum(axis=0)
    print("pair features (weight mine / theirs, occurrences):")
    for i, nm in enumerate(names):
        print(f"  {nm:16s} {w[F_PAIR_MINE + i]:6d} {w[F_PAIR_THEIRS + i]:6d}   n={int(cnt[F_PAIR_MINE + i])}/{int(cnt[F_PAIR_THEIRS + i])}")
    tail = ["hand_mine", "hand_theirs", "rock_conn", "rock_free", "tempo", "second"]
    for i, nm in enumerate(tail):
        print(f"  {nm:16s} {w[F_PAIR_THEIRS + N_PAIR + i]:6d}")
    print("shape weights [missing][my rocks][their rocks]:")
    base_idx = [0, 5, 9, 12, 14]
    for m in range(3):
        rows = []
        for a in range(5):
            rows.append(" ".join(f"{w[m * 15 + base_idx[a] + b]:5d}" for b in range(5 - a)))
        print(f"  missing {m + 1}: " + " | ".join(rows))


if __name__ == "__main__":
    main()
