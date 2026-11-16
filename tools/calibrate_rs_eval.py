"""Calibrate the rs engine's search score against real game results.

    engine_src/rs_selfplay gen --games 600 --nodes 200000 --out data/rs/calib.bin
    python3 tools/calibrate_rs_eval.py data/rs/calib.bin

Fits P(mover wins) = sigmoid(slope * score + offset[mover]) on the positions of self-play
games where the search did not already see a forced result, and prints the constants used by
players/rs_display.py to turn a raw score into a win chance. The offset differs by mover
because the raw score includes a bonus for having the move.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rs_engine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+")
    args = ap.parse_args()

    width = rs_engine.N_FEATS + 5
    chunks = []
    for p in args.data:
        raw = np.fromfile(p, dtype=np.int16)
        chunks.append(raw[: (raw.size // width) * width].reshape(-1, width))
    rec = np.concatenate(chunks)
    rec = rec[rec[:, 0] >= 0]  # leaf records (rs_selfplay --leaves) have no game result
    result = rec[:, 0].astype(np.float64) / 2.0
    score = rec[:, 1].astype(np.float64)
    # the last feature is +1 when player 2 is to move, -1 when player 1 is (rs_eval.hpp F_SECOND);
    # ply parity is not enough because self-play games start after random opening moves
    mover = (rec[:, -1] > 0).astype(int)

    forced = np.abs(score) >= 29000
    right = np.sign(score[forced]) == np.sign(result[forced] - 0.5)
    print(f"positions: {len(rec)}; forced results announced: {forced.sum()} ({100 * right.mean():.2f}% came true)")

    s, r, m = score[~forced], result[~forced], mover[~forced]
    x = np.stack([s / 100.0, (m == 0).astype(float), (m == 1).astype(float)], axis=1)
    theta = np.array([100 / 250.0, 0.0, 0.0])
    for _ in range(50):  # Newton's method
        p = 1 / (1 + np.exp(-(x @ theta)))
        grad = x.T @ (p - r)
        hess = (x * (p * (1 - p))[:, None]).T @ x
        theta -= np.linalg.solve(hess + 1e-9 * np.eye(3), grad)
    p = 1 / (1 + np.exp(-(x @ theta)))
    ll = -np.mean(r * np.log(p + 1e-12) + (1 - r) * np.log(1 - p + 1e-12))
    slope = theta[0] / 100.0
    print(f"fit on {len(s)} positions: logloss {ll:.4f}")
    print(f"  SLOPE = {slope:.6f}   (1 / {1 / slope:.0f})")
    print(f"  OFFSET = ({theta[1]:.3f}, {theta[2]:.3f})   # mover is player 1, player 2")
    print("check (predicted vs actual win rate for the mover):")
    for lo, hi in [(0, 0.1), (0.1, 0.25), (0.25, 0.4), (0.4, 0.6), (0.6, 0.75), (0.75, 0.9), (0.9, 1.01)]:
        sel = (p >= lo) & (p < hi)
        if sel.sum():
            print(f"  predicted {lo:.2f}-{min(hi, 1):.2f}: n={sel.sum():6d}  mean predicted {p[sel].mean():.2f}  actual {r[sel].mean():.2f}")


if __name__ == "__main__":
    main()
