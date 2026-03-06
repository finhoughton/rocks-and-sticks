from __future__ import annotations

import gc
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import torch

from gnn.encode import SAMPLE_ENC
from gnn.model import load_model
from rl.ab_games import generate_cpp_ab_games
from rl.cli import build_parser
from rl.convert import convert_games_to_dataset
from rl.eval import (
    evaluate_ab_vs_random,
    evaluate_cpp_model_vs_model,
    evaluate_vs_random,
)
from rl.play_games import play_self_play_games
from rl.utils import (
    append_jsonl,
    cleanup_old_datasets,
    export_gnn_eval_from_policy,
    scheduled_int,
)

def _ensure_random_init_eval(out_dir_path: Path) -> Path:
    """Create (or reuse) a deterministic random-init GNNEval checkpoint."""
    from gnn.model import GNNEval

    init_path = out_dir_path / "gnn_eval_init_random.pt"
    if init_path.exists() and init_path.is_file():
        return init_path

    node_dim = int(SAMPLE_ENC.data.x.size(1))  # type: ignore
    global_dim = int(SAMPLE_ENC.data.global_feats.size(1))

    torch.manual_seed(0)
    model = GNNEval(node_feat_dim=node_dim, global_feat_dim=global_dim)
    torch.save(model.state_dict(), init_path)
    print(f"Created random-init GNNEval evaluator at {init_path}")
    return init_path


def _resolve_evaluator(
    i: int,
    args,
    out_dir_path: Path,
    default_ckpt_dir: Path,
) -> Path | None:
    """Pick the evaluator checkpoint for iteration i"""
    if i == 1:
        if args.init_eval_model is not None:
            cand = Path(str(args.init_eval_model))
            if not cand.exists():
                raise FileNotFoundError(f"--init-eval-model not found: {cand}")
            return cand
        return _ensure_random_init_eval(out_dir_path)

    candidates: list[Path] = [
        out_dir_path / f"gnn_eval_iter_{i - 1}.pt",
        default_ckpt_dir / f"gnn_eval_iter_{i - 1}.pt",
    ]
    evaluator = next((p for p in candidates if p.exists()), None)
    if evaluator is not None:
        return evaluator

    if args.init_eval_model is not None and Path(str(args.init_eval_model)).exists():
        return Path(str(args.init_eval_model))

    evaluator = _ensure_random_init_eval(out_dir_path)
    if evaluator is not None:
        return evaluator

    for fb in (out_dir_path / "gnn_eval_balanced.pt", default_ckpt_dir / "gnn_eval_balanced.pt"):
        if fb.exists():
            return fb
    return None


def _resolve_policy_path(i: int, args, out_dir_path: Path) -> str | None:
    """Resolve the PolicyValueNet checkpoint path for C++ self-play priors."""
    policy_path = None
    if i > 1 and str(args.loop_mode) == "az":
        cand = out_dir_path / f"gnn_az_iter_{i - 1}.pt"
        if cand.exists():
            policy_path = str(cand)

    if str(args.loop_mode) == "az" and str(args.backend) == "cpp" and not policy_path:
        if bool(args.auto_init_policy):
            from rl.train import PolicyValueNet

            node_dim = int(SAMPLE_ENC.data.x.size(1))  # type: ignore
            global_dim = int(SAMPLE_ENC.data.global_feats.size(1))  # type: ignore
            move_feat_dim = 16
            torch.manual_seed(0)
            model = PolicyValueNet(node_feat_dim=node_dim, global_feat_dim=global_dim, move_feat_dim=move_feat_dim)
            cand = out_dir_path / f"gnn_az_iter_{i - 1}.pt"
            torch.save(model.state_dict(), cand)
            policy_path = str(cand)
            print(f"Auto-initialized random policy checkpoint at {cand}")
        else:
            raise RuntimeError(
                "C++ backend requires a policy AZ checkpoint for priors."
                " Ensure 'gnn_az_iter_{i-1}.pt' exists in the output directory,"
                " or pass --auto-init-policy to create a temporary random policy."
            )
    return policy_path


def _compute_exploration(i: int, args):
    """Return (temp_eff, eps_eff) for self-play at iteration *i*."""
    decay_start = int(args.selfplay_decay_start)
    age = int(i) - 1
    if decay_start > 0:
        age = max(0, int(i) - int(decay_start))

    temp_eff = float(args.selfplay_temp) * (float(args.selfplay_temp_decay) ** float(age))
    eps_eff = float(args.selfplay_dirichlet_epsilon) * (float(args.selfplay_epsilon_decay) ** float(age))
    temp_eff = max(float(args.selfplay_min_temp), temp_eff)
    eps_eff = max(float(args.selfplay_min_epsilon), eps_eff)
    return temp_eff, eps_eff


# ---------------------------------------------------------------------------
# AB-supervised iteration
# ---------------------------------------------------------------------------

def _run_ab_supervised_iteration(
    i: int,
    args,
    saved_games_path: Path,
    iter_seed: int,
    gnn_eval_ckpt: Path,
    out_dir: Path,
    replay_window: int,
) -> tuple[list[str], None]:
    """Generate AB games, retrain GNNEval.  Returns (train_dataset_paths, None)."""
    print(
        f"AB game generation: games={int(args.games)} time_limit_ms={int(args.ab_time_limit)} "
        f"depth={int(args.ab_depth)} nn_ordering_depth={int(args.ab_nn_ordering_depth)} "
        f"randomize_start={bool(args.ab_randomize_start)}"
    )
    generate_cpp_ab_games(
        num_games=int(args.games),
        save_games_dir=saved_games_path,
        seed_base=int(iter_seed),
        ab_time_limit_ms=int(args.ab_time_limit),
        ab_depth=int(args.ab_depth),
        ab_max_depth=int(args.ab_max_depth),
        ab_move_cap=int(args.ab_move_cap),
        ab_max_moves=int(args.ab_max_moves),
        ab_use_heuristic=bool(args.ab_use_heuristic),
        native_model=str(args.ab_native_model),
        nn_ordering_depth=int(args.ab_nn_ordering_depth),
        randomize_start_enabled=bool(args.ab_randomize_start),
        randomize_max_sticks=int(args.randomize_max_sticks),
    )

    # Collect game directories for training.
    train_game_dirs: list[str] = []
    if replay_window <= 0:
        for j in range(1, int(i) + 1):
            d = Path(args.saved_games_dir.format(iter=j))
            if d.exists() and d.is_dir():
                train_game_dirs.append(str(d))
    else:
        for j in range(max(1, int(i) - replay_window + 1), int(i) + 1):
            d = Path(args.saved_games_dir.format(iter=j))
            if d.exists() and d.is_dir():
                train_game_dirs.append(str(d))
    if not train_game_dirs:
        train_game_dirs = [str(saved_games_path)]

    # Warm-start from the previous iteration's GNNEval checkpoint.
    prev_eval_ckpt: Path | None = None
    if i > 1:
        cand = out_dir / f"gnn_eval_iter_{i - 1}.pt"
        if cand.exists():
            prev_eval_ckpt = cand
    elif i == 1 and args.init_eval_model is not None:
        cand = Path(str(args.init_eval_model))
        if cand.exists():
            prev_eval_ckpt = cand

    gnn_cmd = [
        sys.executable, "-m", "gnn.gnn_main",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--device", str(args.device),
        "--out", str(gnn_eval_ckpt),
        "--soft-labels",
        "--augment-sym",
        "--only-extra-dirs",
        "--extra-dirs", *train_game_dirs,
    ]
    if prev_eval_ckpt is not None:
        gnn_cmd += ["--init-from", str(prev_eval_ckpt)]
        print(f"Warm-starting from {prev_eval_ckpt}")
    print("Running supervised retraining subprocess:", " ".join(gnn_cmd))
    subprocess.run(gnn_cmd, check=True)
    print("Supervised retraining completed for iteration", i)

    return list(train_game_dirs), None


# ---------------------------------------------------------------------------
# AZ iteration
# ---------------------------------------------------------------------------

def _run_az_iteration(
    i: int,
    args,
    saved_games_path: Path,
    iter_seed: int,
    gnn_eval_ckpt: Path,
    out_dir: Path,
    data_dir: Path,
    model_path: str | None,
    policy_path: str | None,
    temp_eff: float,
    eps_eff: float,
    replay_window: int,
    start_iter: int,
    end_iter: int,
) -> tuple[list[str], Path]:
    """Self-play + convert + train.  Returns (train_dataset_paths, policy_ckpt)."""
    if str(args.backend) == "cpp":
        print(
            f"Self-play exploration: temp={temp_eff:.4f} temp_moves={int(args.selfplay_temp_moves)} "
            f"dir_alpha={float(args.selfplay_dirichlet_alpha):.4f} dir_eps={eps_eff:.4f} "
            f"vs_random_prob={float(args.selfplay_vs_random_prob):.3f}"
        )
    play_self_play_games(
        num_games=args.games,
        mcts_rollouts=args.rollouts,
        mcts_time_limit=None,
        save_games_dir=str(saved_games_path),
        model_path=model_path,
        policy_path=policy_path,
        device=args.device,
        backend=args.backend,
        cpp_verbose=int(args.cpp_verbose),
        cpp_use_nn_value=bool(args.cpp_use_nn_value),
        temp=float(temp_eff),
        temperature_moves=int(args.selfplay_temp_moves),
        dirichlet_alpha=float(args.selfplay_dirichlet_alpha),
        dirichlet_epsilon=float(eps_eff),
        vs_random_prob=float(args.selfplay_vs_random_prob),
        seed_base=int(iter_seed),
        prior_scale=float(args.prior_scale),
        prior_mix_uniform=float(args.prior_mix_uniform),
        filter_near_area=bool(args.filter_near_area),
        filter_probe_rollouts=int(args.filter_probe_rollouts),
        filter_probe_moves=int(args.filter_probe_moves),
        randomize_max_sticks=int(args.randomize_max_sticks),
    )

    # --- Convert games to dataset ---
    dataset_path = data_dir / f"alpha_dataset_iter_{i}.pt"
    shard_size = int(args.dataset_shard_size) if int(args.dataset_shard_size) > 0 else None

    fixed_topk = int(args.dataset_policy_topk)
    if fixed_topk > 0:
        topk_eff = fixed_topk
    else:
        start_k = int(args.dataset_policy_topk_start)
        end_k = int(args.dataset_policy_topk_end)
        if start_k > 0 or end_k > 0:
            sched_start = int(args.dataset_policy_topk_start_iter) if int(args.dataset_policy_topk_start_iter) > 0 else int(start_iter)
            sched_end = int(args.dataset_policy_topk_end_iter) if int(args.dataset_policy_topk_end_iter) > 0 else int(end_iter)
            topk_eff = scheduled_int(
                cur_iter=int(i),
                start_iter=int(sched_start),
                end_iter=int(sched_end),
                start_val=max(0, start_k),
                end_val=max(0, end_k),
                curve=str(args.dataset_policy_topk_curve),
            )
        else:
            topk_eff = 0

    topk = int(topk_eff) if int(topk_eff) > 0 else None
    print(f"Dataset conversion: policy_topk={topk or 'ALL'} (iter={i})")
    convert_games_to_dataset(
        str(saved_games_path),
        str(dataset_path),
        augment=not args.no_augment,
        shard_size=shard_size,
        policy_topk=topk,
        policy_min_prob=float(args.dataset_policy_min_prob),
        grouped_policy=bool(args.dataset_grouped_policy),
    )

    policy_ckpt = out_dir / f"gnn_az_iter_{i}.pt"

    # --- Collect replay datasets ---
    train_dataset_paths: list[str] = []
    if replay_window <= 0:
        for j in range(1, int(i) + 1):
            dp = data_dir / f"alpha_dataset_iter_{j}.pt"
            if dp.exists():
                train_dataset_paths.append(str(dp))
    else:
        for j in range(max(1, int(i) - replay_window + 1), int(i) + 1):
            dp = data_dir / f"alpha_dataset_iter_{j}.pt"
            if dp.exists():
                train_dataset_paths.append(str(dp))
    if not train_dataset_paths:
        train_dataset_paths = [str(dataset_path)]

    # --- Warm-start ---
    init_from: str | None = None
    if i > 1:
        prev_ckpt = out_dir / f"gnn_az_iter_{i - 1}.pt"
        if prev_ckpt.exists():
            init_from = str(prev_ckpt)
    elif i == 1 and args.init_policy_from is not None:
        cand = Path(str(args.init_policy_from))
        if not cand.exists():
            raise FileNotFoundError(f"--init-policy-from not found: {cand}")
        init_from = str(cand)
        print(f"Iteration 1: warm-starting PolicyValueNet from {init_from}")

    # --- Diagnostic training run ---
    gc.collect()
    _run_diagnostic_training(i, args, train_dataset_paths, out_dir, iter_seed, init_from)

    # --- Full training ---
    _run_full_training(i, args, train_dataset_paths, policy_ckpt, iter_seed, init_from)

    print("Training completed for iteration", i, "cleaning up old datasets...")
    _keep = max(int(args.keep_last_datasets), replay_window) if replay_window > 0 else i
    cleanup_old_datasets(data_dir, keep_last=_keep, current_iter=i)

    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass

    export_gnn_eval_from_policy(policy_ckpt, gnn_eval_ckpt)

    return list(train_dataset_paths), policy_ckpt


def _run_diagnostic_training(
    i: int, args, train_dataset_paths: list[str], out_dir: Path, iter_seed: int, init_from: str | None,
) -> None:
    """Short 1-epoch smoke run to capture training diagnostics."""
    try:
        diag_cmd = [
            sys.executable, "-m", "rl.train",
            "--dataset", *train_dataset_paths,
            "--epochs", "1",
            "--steps-per-epoch", "100",
            "--batch-size", str(max(1, min(16, int(args.batch_size)))),
            "--lr", str(args.lr),
            "--value-weight", str(args.value_weight),
            "--value-lr-mult", str(args.value_lr_mult),
            "--device", str(args.device),
            "--out", str(out_dir / f"gnn_az_diag_iter_{i}.pt"),
            "--seed", str(int(iter_seed) + 7),
            "--diagnostics-out", str(Path("logs/run_loop_diagnostics.json")),
        ]
        if init_from is not None:
            diag_cmd += ["--init-from", str(init_from)]
        dev = str(args.device)
        if dev == "mps" or dev.startswith("cuda"):
            diag_cmd.append("--amp")
        print("Running short diagnostic training subprocess:", " ".join(diag_cmd))
        subprocess.run(diag_cmd, check=True)
    except Exception as e:
        print(f"Diagnostic training run failed (continuing): {e}")


def _run_full_training(
    i: int, args, train_dataset_paths: list[str], policy_ckpt: Path, iter_seed: int, init_from: str | None,
) -> None:
    """Full training subprocess with automatic fallback on OOM/SIGKILL."""
    dev = str(args.device)
    train_cmd = [
        sys.executable, "-m", "rl.train",
        "--dataset", *train_dataset_paths,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--value-weight", str(args.value_weight),
        "--value-lr-mult", str(args.value_lr_mult),
        "--device", dev,
        "--out", str(policy_ckpt),
        "--seed", str(int(iter_seed) + 7),
    ]
    if init_from is not None:
        train_cmd += ["--init-from", str(init_from)]
    if dev == "mps" or dev.startswith("cuda"):
        train_cmd.append("--amp")
    if int(args.train_num_workers) > 0:
        train_cmd += ["--num-workers", str(int(args.train_num_workers))]
        train_cmd += ["--prefetch-factor", str(int(args.train_prefetch_factor))]
        if args.train_persistent_workers:
            train_cmd += ["--persistent-workers"]
    if int(args.train_steps_per_epoch) > 0:
        train_cmd += ["--steps-per-epoch", str(int(args.train_steps_per_epoch))]

    print("Running training subprocess:", " ".join(train_cmd))
    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        if e.returncode not in (-9, 137):
            raise
        # Retry without multiprocessing workers.
        print("Training subprocess was SIGKILLed; retrying with --num-workers 0...")
        cleaned: list[str] = []
        skip_next = False
        for c in train_cmd:
            if skip_next:
                skip_next = False
                continue
            if c in ("--num-workers", "--prefetch-factor"):
                skip_next = True
                continue
            if c == "--persistent-workers":
                continue
            cleaned.append(c)
        print("Retry command:", " ".join(cleaned))
        try:
            subprocess.run(cleaned, check=True)
        except subprocess.CalledProcessError as e2:
            if e2.returncode not in (-9, 137):
                raise
            # Last resort: smaller batch, capped steps, no AMP.
            print("Retry was SIGKILLed; retrying with smaller batch and capped steps (no AMP)...")
            safe_bs = max(1, int(args.batch_size) // 2)
            safe_steps = int(args.train_steps_per_epoch) if int(args.train_steps_per_epoch) > 0 else 2000
            fallback_cmd = [
                sys.executable, "-m", "rl.train",
                "--dataset", *train_dataset_paths,
                "--epochs", str(args.epochs),
                "--batch-size", str(safe_bs),
                "--lr", str(args.lr),
                "--device", dev,
                "--out", str(policy_ckpt),
                "--steps-per-epoch", str(safe_steps),
                "--clear-cache-interval", "50",
            ]
            if init_from is not None:
                fallback_cmd += ["--init-from", str(init_from)]
            print("Fallback command:", " ".join(fallback_cmd))
            subprocess.run(fallback_cmd, check=True)


# ---------------------------------------------------------------------------
# Evaluation dispatch
# ---------------------------------------------------------------------------

def _run_evaluation(i: int, args, gnn_eval_ckpt: Path, policy_ckpt: Path | None, out_dir: Path) -> dict | None:
    """Run post-iteration evaluation and return the record dict (or None if skipped)."""
    if not args.eval_games or args.eval_games <= 0:
        return None

    if str(args.loop_mode) == "ab-supervised":
        print(
            f"Evaluating AB engine vs Random ({int(args.eval_games)} games, "
            f"time_limit={int(args.ab_time_limit)}ms, depth={int(args.ab_depth)})..."
        )
        record = evaluate_ab_vs_random(
            eval_games=int(args.eval_games),
            eval_max_moves=int(args.eval_max_moves),
            eval_seed=int(args.eval_seed),
            eval_randomize_start=bool(args.eval_randomize_start),
            iteration=int(i),
            ab_time_limit_ms=int(args.ab_time_limit),
            ab_depth=int(args.ab_depth),
            ab_max_depth=int(args.ab_max_depth),
            ab_move_cap=int(args.ab_move_cap),
            ab_use_heuristic=bool(args.ab_use_heuristic),
            native_model=str(args.ab_native_model),
            nn_ordering_depth=int(args.ab_nn_ordering_depth),
            randomize_max_sticks=int(args.randomize_max_sticks),
        )
    else:
        try:
            node_dim = SAMPLE_ENC.data.x.size(1)  # type: ignore
            global_dim = SAMPLE_ENC.data.global_feats.size(1)
            load_model(str(gnn_eval_ckpt), node_dim, global_dim, device=args.device)
            print(f"Loaded evaluator {gnn_eval_ckpt} for evaluation")
        except Exception as e:
            print(f"Failed to load evaluator for evaluation: {e}")
            return None

        record = evaluate_vs_random(
            backend=str(args.backend),
            device=str(args.device),
            eval_games=int(args.eval_games),
            eval_rollouts=int(args.eval_rollouts),
            eval_max_moves=int(args.eval_max_moves),
            eval_seed=int(args.eval_seed),
            eval_randomize_start=bool(args.eval_randomize_start),
            iteration=int(i),
            model_path=str(gnn_eval_ckpt),
            policy_path=str(policy_ckpt) if (str(args.backend) == "cpp" and policy_ckpt is not None) else None,
            cpp_verbose=int(args.cpp_verbose),
            cpp_use_nn_value=bool(args.cpp_use_nn_value),
            eval_jobs=int(args.eval_jobs),
            prior_scale=float(args.prior_scale),
            prior_mix_uniform=float(args.prior_mix_uniform),
            filter_near_area=bool(args.filter_near_area),
            filter_probe_rollouts=int(args.filter_probe_rollouts),
            filter_probe_moves=int(args.filter_probe_moves),
            randomize_max_sticks=int(args.randomize_max_sticks),
        )

    return record


def _run_optional_evals(
    i: int, args, record: dict, gnn_eval_ckpt: Path, policy_ckpt: Path | None, out_dir: Path,
) -> None:
    """Low-rollout eval, heavy eval, and ladder (vs-prev) eval."""
    # Low-rollout eval
    if int(args.eval_games_low) > 0 and int(args.eval_rollouts_low) > 0:
        low = evaluate_vs_random(
            backend=str(args.backend),
            device=str(args.device),
            eval_games=int(args.eval_games_low),
            eval_rollouts=int(args.eval_rollouts_low),
            eval_max_moves=int(args.eval_max_moves),
            eval_seed=int(args.eval_seed) + 777777,
            eval_randomize_start=bool(args.eval_randomize_start),
            iteration=int(i),
            model_path=str(gnn_eval_ckpt),
            policy_path=str(policy_ckpt) if (str(args.backend) == "cpp" and policy_ckpt is not None) else None,
            cpp_verbose=int(args.cpp_verbose),
            cpp_use_nn_value=bool(args.cpp_use_nn_value),
            prior_scale=float(args.prior_scale),
            prior_mix_uniform=float(args.prior_mix_uniform),
            filter_near_area=bool(args.filter_near_area),
            filter_probe_rollouts=int(args.filter_probe_rollouts),
            filter_probe_moves=int(args.filter_probe_moves),
            randomize_max_sticks=int(args.randomize_max_sticks),
        )
        record["vs_random_low_rollouts"] = {
            "eval_games": int(low.get("eval_games", 0)),
            "eval_rollouts": int(low.get("eval_rollouts", 0)),
            "win_rate": float(low.get("win_rate", 0.0)),
            "wins": int(low.get("wins", 0)),
            "losses": int(low.get("losses", 0)),
            "draws": int(low.get("draws", 0)),
            "avg_moves": float(low.get("avg_moves", 0.0)),
            "seconds": float(low.get("seconds", 0.0)),
            "first_agent_moves_top": low.get("first_agent_moves_top", []),
        }

    # Ladder: vs previous checkpoints
    if int(args.eval_vs_prev) > 0:
        if str(args.backend) != "cpp":
            print("Note: --eval-vs-prev requires --backend=cpp.")
        elif i <= 1:
            pass
        elif int(args.eval_vs_prev_every) > 1 and (int(i) % int(args.eval_vs_prev_every) != 0):
            pass
        else:
            prev_n = int(args.eval_vs_prev)
            start_opp = max(0, int(i) - prev_n)
            vs_prev: dict[str, dict] = {}
            for opp_iter in range(start_opp, int(i)):
                opp_ckpt = out_dir / f"gnn_eval_iter_{opp_iter}.pt" if opp_iter > 0 else out_dir / "gnn_eval_balanced.pt"
                if not opp_ckpt.exists():
                    continue
                print(f"Evaluating vs checkpoint iter {opp_iter}...")
                res = evaluate_cpp_model_vs_model(
                    device=str(args.device),
                    eval_games=int(args.eval_prev_games),
                    eval_rollouts=int(args.eval_prev_rollouts),
                    eval_max_moves=int(args.eval_max_moves),
                    eval_seed=int(args.eval_seed) + 100000 * int(opp_iter),
                    iteration=int(i),
                    model_a_path=str(gnn_eval_ckpt),
                    model_b_path=str(opp_ckpt),
                    policy_a_path=str(policy_ckpt) if (policy_ckpt is not None and policy_ckpt.exists()) else None,
                    policy_b_path=str(out_dir / f"gnn_az_iter_{opp_iter}.pt")
                    if (opp_iter > 0 and (out_dir / f"gnn_az_iter_{opp_iter}.pt").exists())
                    else None,
                    model_a_label=f"iter_{i}",
                    model_b_label=f"iter_{opp_iter}",
                    cpp_verbose=int(args.cpp_verbose),
                    cpp_use_nn_value=bool(args.cpp_use_nn_value),
                    eval_jobs=int(args.eval_jobs),
                    filter_near_area=bool(args.filter_near_area),
                    filter_probe_rollouts=int(args.filter_probe_rollouts),
                    filter_probe_moves=int(args.filter_probe_moves),
                    randomize_max_sticks=int(args.randomize_max_sticks),
                )
                vs_prev[str(opp_iter)] = {
                    "wins": int(res["wins"]),
                    "losses": int(res["losses"]),
                    "draws": int(res["draws"]),
                    "win_rate": float(res["win_rate"]),
                    "eval_games": int(res["eval_games"]),
                    "eval_rollouts": int(res["eval_rollouts"]),
                    "seconds": float(res["seconds"]),
                }
            record["vs_prev_checkpoints"] = vs_prev

    # Heavy vs-Random eval
    if int(args.eval_heavy_every) > 0 and int(args.eval_heavy_games) > 0 and int(args.eval_heavy_rollouts) > 0:
        if int(i) % int(args.eval_heavy_every) == 0:
            heavy = evaluate_vs_random(
                backend=str(args.backend),
                device=str(args.device),
                eval_games=int(args.eval_heavy_games),
                eval_rollouts=int(args.eval_heavy_rollouts),
                eval_max_moves=int(args.eval_max_moves),
                eval_seed=int(args.eval_seed) + 424242,
                eval_randomize_start=bool(args.eval_randomize_start),
                iteration=int(i),
                model_path=str(gnn_eval_ckpt),
                prior_scale=float(args.prior_scale),
                prior_mix_uniform=float(args.prior_mix_uniform),
                cpp_verbose=int(args.cpp_verbose),
                cpp_use_nn_value=bool(args.cpp_use_nn_value),
                eval_jobs=int(args.eval_jobs),
                filter_near_area=bool(args.filter_near_area),
                filter_probe_rollouts=int(args.filter_probe_rollouts),
                filter_probe_moves=int(args.filter_probe_moves),
                randomize_max_sticks=int(args.randomize_max_sticks),
            )
            record["vs_random_heavy"] = {
                "eval_games": int(heavy.get("eval_games", 0)),
                "eval_rollouts": int(heavy.get("eval_rollouts", 0)),
                "win_rate": float(heavy.get("win_rate", 0.0)),
                "wins": int(heavy.get("wins", 0)),
                "losses": int(heavy.get("losses", 0)),
                "draws": int(heavy.get("draws", 0)),
                "avg_moves": float(heavy.get("avg_moves", 0.0)),
                "seconds": float(heavy.get("seconds", 0.0)),
                "first_agent_moves_top": heavy.get("first_agent_moves_top", []),
            }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Detect existing checkpoints to resume numbering.
    max_existing = 0
    for ckpt in out_dir.glob("gnn_az_iter_*.pt"):
        m = re.search(r"gnn_az_iter_(\d+)\.pt$", ckpt.name)
        if m:
            try:
                n = int(m.group(1))
            except ValueError:
                continue
            if n > max_existing:
                max_existing = n

    start_iter = max_existing + 1
    end_iter = start_iter + args.iters - 1
    if max_existing > 0:
        print(f"Resuming training: existing checkpoints up to iteration {max_existing}.")
    else:
        print("Starting training from iteration 1.")

    out_dir_path = Path(args.out_dir)
    default_ckpt_dir = Path("checkpoints")

    for i in range(start_iter, end_iter + 1):
        print(f"=== Iteration {i}/{end_iter} ===")

        iter_seed = int(args.seed) + (100_000 * int(i))
        saved_games_path = Path(args.saved_games_dir.format(iter=i))
        saved_games_path.mkdir(parents=True, exist_ok=True)

        evaluator_path = _resolve_evaluator(i, args, out_dir_path, default_ckpt_dir)
        model_path = str(evaluator_path) if evaluator_path is not None else None
        gnn_eval_ckpt = out_dir / f"gnn_eval_iter_{i}.pt"
        policy_path = _resolve_policy_path(i, args, out_dir_path)
        temp_eff, eps_eff = _compute_exploration(i, args)

        replay_window = int(args.replay_window)
        policy_ckpt: Path | None = None
        train_dataset_paths: list[str] = []

        if str(args.loop_mode) == "ab-supervised":
            train_dataset_paths, _ = _run_ab_supervised_iteration(
                i, args, saved_games_path, iter_seed, gnn_eval_ckpt, out_dir, replay_window,
            )
        else:
            train_dataset_paths, policy_ckpt = _run_az_iteration(
                i, args, saved_games_path, iter_seed, gnn_eval_ckpt, out_dir, data_dir,
                model_path, policy_path, temp_eff, eps_eff, replay_window, start_iter, end_iter,
            )

        # --- Provenance ---
        policy_sha256: str | None = None
        try:
            if policy_ckpt is not None and policy_ckpt.exists() and policy_ckpt.is_file():
                h = hashlib.sha256()
                with policy_ckpt.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        h.update(chunk)
                policy_sha256 = h.hexdigest()
        except Exception:
            policy_sha256 = None

        # --- Evaluation ---
        record = _run_evaluation(i, args, gnn_eval_ckpt, policy_ckpt, out_dir)
        if record is None:
            continue

        record["loop_mode"] = str(args.loop_mode)
        record["policy_ckpt"] = str(policy_ckpt) if policy_ckpt is not None else None
        record["policy_sha256"] = policy_sha256
        record["policy_init_from"] = None  # tracked inside AZ path
        record["train_datasets"] = list(train_dataset_paths)
        record["replay_window"] = int(replay_window)

        _run_optional_evals(i, args, record, gnn_eval_ckpt, policy_ckpt, out_dir)

        print(
            "Evaluation vs RandomPlayer: "
            f"{record['wins']}/{record['eval_games']} wins, "
            f"{record['losses']} losses, {record['draws']} draws "
            f"(win_rate={record['win_rate']:.2f}, avg_moves={record['avg_moves']})"
        )
        try:
            append_jsonl(Path(str(args.strength_log)), record)
            print(f"Appended strength record to {args.strength_log}")
        except Exception as e:
            print(f"Warning: failed to write strength log {args.strength_log}: {e}")

    print("All iterations completed")


if __name__ == "__main__":
    main()
