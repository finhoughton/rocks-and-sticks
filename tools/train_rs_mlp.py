"""Train the rs engine's neural-network evaluation from self-play records.

    engine_src/rs_selfplay gen --games 8000 --nodes 30000 --out data/rs/gen2.bin ...
    python3 tools/train_rs_mlp.py data/rs/gen*.bin --test data/rs/calib_sound.bin

Each record is a position with its evaluation features from the side to move's point of
view and a search score from there. Most come from self-play games and also carry the game's
result; "leaf" records (rs_selfplay --leaves) are positions reached by random walks from game
positions, like the odd positions deep in a search tree where the evaluation does most of its
work, and have only the score. The network predicts the side to move's chance of winning. For
game positions the target blends the result with the score as a win chance (--lam is the
weight on the result); for leaf positions it is the score alone, so the network learns to
foresee what a short search would find. Positions where the search had already seen a forced
result are left out.

Lesson from the first attempt: a network trained only on positions from games predicted
results far better than the linear evaluation, yet played about 50 Elo worse at equal nodes,
because it misjudged the positions the search actually asks about. Judge any network by a
match (rs_selfplay match), not by its log-loss.

Writes the network as text for rs_eval.hpp (Mlp::load): its output, times 200, is in the same
units as the linear evaluation, so the search's pruning margins still fit.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import rs_engine  # noqa: E402

N_FEATS = rs_engine.N_FEATS
SCALE = 200.0  # eval units per logit
MATE_BOUND = 29000


def load(paths: list[str], quiet_games_only: bool = False):
    """(features, result or NaN for leaf records, score, ply, game id, is_leaf)."""
    width = N_FEATS + 5
    out = []
    for p in paths:
        raw = np.fromfile(p, dtype=np.int16)
        rec = raw[: (raw.size // width) * width].reshape(-1, width)
        keep = np.abs(rec[:, 1].astype(np.int32)) < MATE_BOUND
        if quiet_games_only:
            keep &= (rec[:, 2] == 0) & (rec[:, 0] >= 0)
        rec = rec[keep]
        leaf = rec[:, 0] < 0
        res = np.where(leaf, np.nan, rec[:, 0].astype(np.float32) / 2)
        # game ids restart in every file: make them unique
        gid = rec[:, 4].astype(np.int64) + (len(out) << 32)
        out.append((rec[:, 5:].astype(np.float32), res.astype(np.float32), rec[:, 1].astype(np.float32), rec[:, 3].astype(np.int32), gid, leaf))
    return [np.concatenate([o[i] for o in out]) for i in range(6)]


def logloss(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+")
    ap.add_argument("--test", nargs="*", default=[], help="extra files used only for reporting")
    ap.add_argument("--hidden", default="64,32", help="hidden layer widths")
    ap.add_argument("--lam", type=float, default=0.5, help="weight of the game result in the target (rest: search score)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "rs", "candidates", "mlp.txt"),
                    help="where to write the network; blend it into engine_src/eval.txt only after it wins a match")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x, res, score, ply, gid, leaf = load(args.data)
    soft = 1 / (1 + np.exp(-score / SCALE))
    target = np.where(leaf, soft, args.lam * np.nan_to_num(res) + (1 - args.lam) * soft).astype(np.float32)
    games = np.unique(gid)
    val_games = set(rng.choice(games, size=max(1, len(games) // 10), replace=False).tolist())
    val = np.array([g in val_games for g in gid])
    print(f"{len(x)} positions ({int(leaf.sum())} leaf) from {len(games)} games; {int(val.sum())} held out for validation")

    sizes = [N_FEATS] + [int(h) for h in args.hidden.split(",") if h] + [1]
    layers: list = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
    model = torch.nn.Sequential(*layers[:-1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Xtr, Ttr = torch.tensor(x[~val]), torch.tensor(target[~val])
    Xva = torch.tensor(x[val])

    def predict(X) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(model(X).squeeze(-1)).numpy()

    best, best_state = 1e9, None
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), 1024):
            idx = perm[i : i + 1024]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xtr[idx]).squeeze(-1), Ttr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        model.eval()
        p = predict(Xva)
        v_obj = logloss(p, target[val])  # the training objective (soft targets)
        g = ~leaf[val]
        v_res = logloss(p[g], res[val][g]) if g.any() else float("nan")
        v_leaf = logloss(p[~g], target[val][~g]) if (~g).any() else float("nan")
        if v_obj < best:
            best = v_obj
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        print(f"  epoch {ep + 1:2d}: validation objective {v_obj:.4f}; game positions vs results {v_res:.4f}; leaf positions vs search {v_leaf:.4f}", flush=True)
    model.load_state_dict(best_state)

    for path in args.test:
        tx, tres, tscore, tply, _, _ = load([path], quiet_games_only=True)
        p = predict(torch.tensor(tx))
        early = tply < 10
        base = np.full(len(tres), tres.mean())
        print(f"test {os.path.basename(path)}: log-loss {logloss(p, tres):.4f} (constant {logloss(base, tres):.4f}); first 10 plies {logloss(p[early], tres[early]):.4f}")

    linears = [m for m in model if isinstance(m, torch.nn.Linear)]
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(f"mlp {len(sizes)} {' '.join(map(str, sizes))}\n")
        for lin in linears:
            w = lin.weight.detach().numpy().T.copy()  # [input][output]
            bias = lin.bias.detach().numpy().copy()
            w[np.abs(w) < 1e-12] = 0.0  # weights of unused inputs decay to nothing
            bias[np.abs(bias) < 1e-12] = 0.0
            with torch.no_grad():
                lin.weight.copy_(torch.tensor(w.T))
                lin.bias.copy_(torch.tensor(bias))
            fh.write(" ".join(f"{v:.7g}" for v in w.ravel()) + "\n")
            fh.write(" ".join(f"{v:.7g}" for v in bias) + "\n")
    print(f"wrote {args.out} (layers {sizes})")

    # the engine must compute the same thing
    eng = rs_engine.Engine(10)
    if not eng.load_weights(args.out) or not eng.uses_mlp():
        raise SystemExit("the engine could not load the network")
    worst = 0.0
    for i in rng.choice(len(x), size=200, replace=False):
        with torch.no_grad():
            want = float(model(torch.tensor(x[i : i + 1])).squeeze()) * SCALE
        worst = max(worst, abs(want - _engine_eval(eng, x[i])))
    print(f"engine check: largest difference from the trained network {worst:.2f} eval units")
    if worst > 2.0:
        raise SystemExit("engine and trained network disagree")


def _engine_eval(eng, feats: np.ndarray) -> float:
    """Evaluate a feature vector with the engine's network (via a tiny standalone engine call)."""
    return float(eng.mlp_eval([int(v) for v in feats]))


if __name__ == "__main__":
    main()
