import argparse
import gc
import hashlib
import json
import math
import multiprocessing as mp
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch

from game import Game
from gnn.encode import SAMPLE_ENC
from gnn.game_generation import randomize_start
from gnn.model import load_model
from players.base import RandomPlayer
from players.mcts import MCTSPlayer
from rl.convert import convert_games_to_dataset
from rl.play_games import play_self_play_games


def _scheduled_int(*, cur_iter: int, start_iter: int, end_iter: int, start_val: int, end_val: int, curve: str) -> int:
    cur_iter = int(cur_iter)
    start_iter = int(start_iter)
    end_iter = int(end_iter)
    start_val = int(start_val)
    end_val = int(end_val)
    curve = str(curve or "linear").lower()

    if end_iter <= start_iter:
        return int(end_val if cur_iter >= end_iter else start_val)

    t = (float(cur_iter) - float(start_iter)) / float(end_iter - start_iter)
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)

    if curve in ("cos", "cosine"):
        # smoothstep-like easing using cosine
        t = 0.5 - 0.5 * math.cos(math.pi * t)
    elif curve in ("linear", "lin"):
        pass
    else:
        raise ValueError(f"Unknown schedule curve: {curve}")

    v = float(start_val) + (float(end_val) - float(start_val)) * t
    return int(round(v))


def export_gnn_eval_from_policy(policy_ckpt: Path, out_path: Path):
    """Load policy+value checkpoint and copy conv/norm weights into GNNEval then save."""
    from gnn.encode import SAMPLE_ENC
    from gnn.model import GNNEval

    device = torch.device('cpu')
    ck = torch.load(policy_ckpt, map_location=device)
    state = ck if isinstance(ck, dict) and 'state_dict' not in ck else ck.get('state_dict', ck)

    # infer feature dims from sample encoding
    node_feat_dim = SAMPLE_ENC.data.x.shape[1] # type: ignore
    global_feat_dim = SAMPLE_ENC.data.global_feats.shape[1]
    # instantiate GNNEval with inferred dims
    model = GNNEval(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim)
    model_sd = model.state_dict()

    # Copy matching trunk keys (convs/norms/etc).
    new_sd: dict[str, torch.Tensor] = {}
    for k in model_sd.keys():
        if k in state:
            new_sd[k] = state[k]

    # If this checkpoint came from PolicyValueNet, map its value head into GNNEval.head.
    # PolicyValueNet has: value_mlp = [Linear, ReLU, Linear]
    # GNNEval has: head = [Linear, ReLU, Dropout, Linear]
    if isinstance(state, dict) and any(str(k).startswith('value_mlp.') for k in state.keys()):
        # First linear
        if 'value_mlp.0.weight' in state and 'head.0.weight' in model_sd:
            new_sd['head.0.weight'] = state['value_mlp.0.weight']
        if 'value_mlp.0.bias' in state and 'head.0.bias' in model_sd:
            new_sd['head.0.bias'] = state['value_mlp.0.bias']
        # Final linear
        if 'value_mlp.2.weight' in state and 'head.3.weight' in model_sd:
            new_sd['head.3.weight'] = state['value_mlp.2.weight']
        if 'value_mlp.2.bias' in state and 'head.3.bias' in model_sd:
            new_sd['head.3.bias'] = state['value_mlp.2.bias']

    model_sd.update(new_sd)
    model.load_state_dict(model_sd)
    torch.save(model.state_dict(), out_path)
    print(f"Saved GNNEval checkpoint to {out_path}")


def _cleanup_old_datasets(data_dir: Path, *, keep_last: int, current_iter: int) -> None:
    keep_last = max(0, int(keep_last))
    cutoff = current_iter - keep_last
    if cutoff <= 0:
        return

    for k in range(1, cutoff + 1):
        pt_path = data_dir / f"alpha_dataset_iter_{k}.pt"
        shards_dir = data_dir / f"alpha_dataset_iter_{k}.pt.shards"
        try:
            if shards_dir.exists() and shards_dir.is_dir():
                shutil.rmtree(shards_dir)
        except Exception as e:
            print(f"Warning: failed to delete {shards_dir}: {e}")
        try:
            if pt_path.exists() and pt_path.is_file():
                pt_path.unlink()
        except Exception as e:
            print(f"Warning: failed to delete {pt_path}: {e}")


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def _merge_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = int(out.get(k, 0)) + int(v)
    return out


def _split_indices(n: int, parts: int) -> list[list[int]]:
    n = int(n)
    parts = max(1, int(parts))
    idxs = list(range(n))
    if parts <= 1 or n <= 1:
        return [idxs]
    out: list[list[int]] = [[] for _ in range(parts)]
    for j, i in enumerate(idxs):
        out[j % parts].append(i)
    return [c for c in out if c]


def _eval_vs_random_worker(payload: dict) -> dict:
    torch_threads = int(payload.get("torch_threads") or 0)
    if torch_threads > 0:
        torch.set_num_threads(int(torch_threads))
        torch.set_num_interop_threads(1)

    backend = str(payload["backend"])
    device = str(payload["device"])
    eval_rollouts = int(payload["eval_rollouts"])
    eval_max_moves = int(payload["eval_max_moves"])
    eval_seed = int(payload["eval_seed"])
    eval_randomize_start = bool(payload["eval_randomize_start"])
    filter_near_area = bool(payload.get("filter_near_area", False))
    filter_probe_rollouts = int(payload.get("filter_probe_rollouts") or 200)
    filter_probe_moves = int(payload.get("filter_probe_moves") or 6)
    randomize_max_sticks = int(payload.get("randomize_max_sticks") or 5)
    model_path = str(payload["model_path"])
    policy_path = str(payload.get("policy_path") or "")
    prior_scale = float(payload.get("prior_scale") or 1.0)
    prior_mix_uniform = float(payload.get("prior_mix_uniform") or 0.04)
    cpp_verbose = int(payload.get("cpp_verbose") or 0)
    cpp_use_nn_value = bool(payload["cpp_use_nn_value"])
    gidxs: list[int] = list(payload["gidxs"])

    if backend != "cpp":
        node_dim = SAMPLE_ENC.data.x.size(1)  # type: ignore
        global_dim = SAMPLE_ENC.data.global_feats.size(1)
        load_model(model_path, node_dim, global_dim, device=device)

    wins = 0
    losses = 0
    draws = 0
    max_moves_reached = 0
    move_counts: list[int] = []
    first_agent_move_counts: dict[str, int] = {}
    cpp_profile_sum: dict[str, float | int | bool] = {}

    if backend == "cpp":
        import players_ext

        from players.game_total import GameTotal
        from players.mcts_cpp import MCTSPlayerCPP

        # Reuse engines across games to avoid re-loading the model per game.
        # Use the same PUCT constant as the Python MCTSPlayer (1.0) for parity.
        mcts0 = MCTSPlayerCPP(0, c_puct=1.0, n_rollouts=eval_rollouts, seed=eval_seed + 17, verbose=bool(cpp_verbose), use_nn_value=bool(cpp_use_nn_value))
        mcts1 = MCTSPlayerCPP(1, c_puct=1.0, n_rollouts=eval_rollouts, seed=eval_seed + 23, verbose=bool(cpp_verbose), use_nn_value=bool(cpp_use_nn_value))
        mcts0.set_model_checkpoint(str(model_path), device=str(device))
        mcts1.set_model_checkpoint(str(model_path), device=str(device))
        if policy_path:
            try:
                mcts0.set_policy_checkpoint(str(policy_path), device=str(device))
                mcts1.set_policy_checkpoint(str(policy_path), device=str(device))
            except Exception as e:
                print(f"Warning: failed to load policy checkpoint {policy_path}: {e}")
        mcts0.set_exploration(dirichlet_alpha=0.0, dirichlet_epsilon=0.0, temperature=0.0, temperature_moves=0)
        mcts1.set_exploration(dirichlet_alpha=0.0, dirichlet_epsilon=0.0, temperature=0.0, temperature_moves=0)
        try:
            mcts0.engine.set_prior_params(float(prior_mix_uniform), float(prior_scale))
            mcts1.engine.set_prior_params(float(prior_mix_uniform), float(prior_scale))
        except Exception:
            pass

    # Build probe function for filtering randomized starts (MCTS quick-win probe).
    _probe_fn = None
    _probe_engines: dict[int, object] | None = None
    if filter_near_area and backend == "cpp":
        from players.mcts_cpp import MCTSPlayerCPP as _ProbeMCTS

        _probe_engines = {
            0: _ProbeMCTS(0, c_puct=1.0, n_rollouts=filter_probe_rollouts, seed=0, verbose=False, use_nn_value=bool(cpp_use_nn_value)),
            1: _ProbeMCTS(1, c_puct=1.0, n_rollouts=filter_probe_rollouts, seed=0, verbose=False, use_nn_value=bool(cpp_use_nn_value)),
        }
        for _pe in _probe_engines.values():
            _pe.set_model_checkpoint(str(model_path), device=str(device))
            _pe.set_exploration(dirichlet_alpha=0.0, dirichlet_epsilon=0.0, temperature=0.0, temperature_moves=0)

        def _probe_fn_impl(game_state, player):
            pe = _probe_engines[player]
            return pe.get_move(game_state)
        _probe_fn = _probe_fn_impl

    for gidx in gidxs:
        random.seed(int(eval_seed) + int(gidx))

        agent_player = 0 if (gidx % 2 == 0) else 1
        random_player = 1 - agent_player

        if backend == "cpp":
            import players_ext

            from players.game_total import GameTotal
            mcts = mcts0 if agent_player == 0 else mcts1  # type: ignore[name-defined]
            mcts.reset_search()
            rnd = RandomPlayer(random_player, seed=int(eval_seed) + int(gidx) + 1)
            players = [mcts, rnd] if agent_player == 0 else [rnd, mcts]
            game = GameTotal(Game(players=players), players_ext.GameState())
            if bool(eval_randomize_start):
                if _probe_engines is not None:
                    for _pe in _probe_engines.values():
                        _pe.reset_search()
                randomize_start(game, max_sticks=randomize_max_sticks, filter_near_area=bool(filter_near_area), probe_fn=_probe_fn, probe_max_moves=filter_probe_moves)
        else:
            mcts = MCTSPlayer(agent_player, use_gnn=True, n_rollouts=eval_rollouts, seed=int(eval_seed) + int(gidx))
            rnd = RandomPlayer(random_player, seed=int(eval_seed) + int(gidx) + 1)
            players = [mcts, rnd] if agent_player == 0 else [rnd, mcts]
            game = Game(players=players)
            if bool(eval_randomize_start):
                randomize_start(game, max_sticks=randomize_max_sticks, filter_near_area=bool(filter_near_area))

        move_count = 0
        seen_agent_move = False
        while game.winner is None and move_count < int(eval_max_moves):
            p = game.players[game.current_player]
            if backend == "cpp":
                mv = p.get_move(game)
            else:
                if isinstance(p, MCTSPlayer):
                    mv = p.get_move(game, reuse_tree=True)
                else:
                    mv = p.get_move(game)

            if (not seen_agent_move) and (game.current_player == agent_player):
                seen_agent_move = True
                mk = f"{mv.c[0]},{mv.c[1]},{mv.t}"
                first_agent_move_counts[mk] = first_agent_move_counts.get(mk, 0) + 1

            game.do_move(game.current_player, mv)
            move_count += 1

            # Reuse MCTS tree across moves (C++ backend needs explicit root advance).
            # Fixed: advance_root now properly clears root_priors for the new position.
            if backend == "cpp":
                for pl in game.players:
                    adv = getattr(pl, "advance_root", None)
                    if adv is not None:
                        adv(mv, game)

        if backend == "cpp":
            try:
                prof = dict(mcts.engine.get_profile_stats())  # type: ignore[attr-defined]
                for k, v in prof.items():
                    if isinstance(v, bool):
                        cpp_profile_sum[k] = bool(v)
                    elif isinstance(v, (int, float)):
                        cpp_profile_sum[k] = float(cpp_profile_sum.get(k, 0.0)) + float(v)
            except Exception:
                pass

        move_counts.append(move_count)
        if game.winner == agent_player:
            wins += 1
        elif game.winner == random_player:
            losses += 1
        else:
            draws += 1
            if move_count >= int(eval_max_moves):
                max_moves_reached += 1

    return {
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "max_moves_reached": int(max_moves_reached),
        "move_counts": move_counts,
        "first_agent_move_counts": first_agent_move_counts,
        "cpp_profile_sum": cpp_profile_sum,
    }


def _eval_cpp_model_vs_model_worker(payload: dict) -> dict:
    torch_threads = int(payload.get("torch_threads") or 0)
    if torch_threads > 0:
        try:
            import torch

            torch.set_num_threads(int(torch_threads))
            try:
                torch.set_num_interop_threads(1)
            except Exception:
                pass
        except Exception:
            pass

    import players_ext

    from players.game_total import GameTotal
    from players.mcts_cpp import MCTSPlayerCPP

    device = str(payload["device"])
    eval_rollouts = int(payload["eval_rollouts"])
    eval_max_moves = int(payload["eval_max_moves"])
    eval_seed = int(payload["eval_seed"])
    model_a_path = str(payload["model_a_path"])
    model_b_path = str(payload["model_b_path"])
    policy_a_path = str(payload.get("policy_a_path") or "")
    policy_b_path = str(payload.get("policy_b_path") or "")
    cpp_verbose = int(payload.get("cpp_verbose") or 0)
    cpp_use_nn_value = bool(payload["cpp_use_nn_value"])
    gidxs: list[int] = list(payload["gidxs"])
    filter_near_area = bool(payload.get("filter_near_area", False))
    filter_probe_rollouts = int(payload.get("filter_probe_rollouts") or 200)
    filter_probe_moves = int(payload.get("filter_probe_moves") or 6)
    randomize_max_sticks = int(payload.get("randomize_max_sticks") or 5)

    # Reuse engines across games to avoid re-loading checkpoints per game.
    # Ensure PUCT parity with Python MCTSPlayer by using c_puct=1.0.
    a0 = MCTSPlayerCPP(0, c_puct=1.0, n_rollouts=eval_rollouts, seed=eval_seed + 101, verbose=bool(cpp_verbose), use_nn_value=bool(cpp_use_nn_value))
    a1 = MCTSPlayerCPP(1, c_puct=1.0, n_rollouts=eval_rollouts, seed=eval_seed + 103, verbose=bool(cpp_verbose), use_nn_value=bool(cpp_use_nn_value))
    b0 = MCTSPlayerCPP(0, c_puct=1.0, n_rollouts=eval_rollouts, seed=eval_seed + 107, verbose=bool(cpp_verbose), use_nn_value=bool(cpp_use_nn_value))
    b1 = MCTSPlayerCPP(1, c_puct=1.0, n_rollouts=eval_rollouts, seed=eval_seed + 109, verbose=bool(cpp_verbose), use_nn_value=bool(cpp_use_nn_value))

    for p in (a0, a1):
        p.set_exploration(dirichlet_alpha=0.0, dirichlet_epsilon=0.0, temperature=0.0, temperature_moves=0)
        p.set_model_checkpoint(model_a_path, device=device)
        if policy_a_path:
            try:
                p.set_policy_checkpoint(policy_a_path, device=device)
            except Exception:
                pass
    for p in (b0, b1):
        p.set_exploration(dirichlet_alpha=0.0, dirichlet_epsilon=0.0, temperature=0.0, temperature_moves=0)
        p.set_model_checkpoint(model_b_path, device=device)
        if policy_b_path:
            try:
                p.set_policy_checkpoint(policy_b_path, device=device)
            except Exception:
                pass

    wins = 0
    losses = 0
    draws = 0
    max_moves_reached = 0
    move_counts: list[int] = []

    # Build probe function for filtering randomized starts.
    _probe_fn_mm = None
    _probe_engines_mm: dict[int, object] | None = None
    if filter_near_area:
        _probe_engines_mm = {
            0: MCTSPlayerCPP(0, c_puct=1.0, n_rollouts=filter_probe_rollouts, seed=0, verbose=False, use_nn_value=bool(cpp_use_nn_value)),
            1: MCTSPlayerCPP(1, c_puct=1.0, n_rollouts=filter_probe_rollouts, seed=0, verbose=False, use_nn_value=bool(cpp_use_nn_value)),
        }
        for _pe in _probe_engines_mm.values():
            _pe.set_model_checkpoint(model_a_path, device=device)
            _pe.set_exploration(dirichlet_alpha=0.0, dirichlet_epsilon=0.0, temperature=0.0, temperature_moves=0)

        def _probe_fn_mm_impl(game_state, player):
            pe = _probe_engines_mm[player]
            return pe.get_move(game_state)
        _probe_fn_mm = _probe_fn_mm_impl

    for gidx in gidxs:
        random.seed(int(eval_seed) + int(gidx))
        a_as_p0 = (gidx % 2 == 0)

        p0 = a0 if a_as_p0 else b0
        p1 = b1 if a_as_p0 else a1
        p0.reset_search()
        p1.reset_search()

        game = GameTotal(Game(players=[p0, p1]), players_ext.GameState())
        if _probe_engines_mm is not None:
            for _pe in _probe_engines_mm.values():
                _pe.reset_search()
        randomize_start(game, max_sticks=randomize_max_sticks, filter_near_area=bool(filter_near_area), probe_fn=_probe_fn_mm, probe_max_moves=filter_probe_moves)

        move_count = 0
        while game.winner is None and move_count < int(eval_max_moves):
            pl = game.players[game.current_player]
            mv = pl.get_move(game)
            game.do_move(game.current_player, mv)
            move_count += 1

            for pl2 in game.players:
                adv = getattr(pl2, "advance_root", None)
                if adv is not None:
                    adv(mv, game)

        move_counts.append(move_count)
        if game.winner is None:
            draws += 1
            if move_count >= int(eval_max_moves):
                max_moves_reached += 1
        else:
            a_player_num = 0 if a_as_p0 else 1
            if int(game.winner) == a_player_num:
                wins += 1
            else:
                losses += 1

    return {
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "max_moves_reached": int(max_moves_reached),
        "move_counts": move_counts,
    }


def _evaluate_vs_random(
    *,
    backend: str,
    device: str,
    eval_games: int,
    eval_rollouts: int,
    eval_max_moves: int,
    eval_seed: int,
    eval_randomize_start: bool,
    iteration: int,
    model_path: str,
    policy_path: str | None = None,
    cpp_verbose: int,
    cpp_use_nn_value: bool,
    eval_jobs: int = 1,
    prior_scale: float = 1.0,
    prior_mix_uniform: float = 0.04,
    filter_near_area: bool = False,
    filter_probe_rollouts: int = 200,
    filter_probe_moves: int = 6,
    randomize_max_sticks: int = 5,
) -> dict:
    # For the Python backend, ensure we're actually evaluating the requested checkpoint.
    # For the C++ backend, the model must be loaded via MCTSPlayerCPP.set_model_checkpoint().
    if backend != 'cpp':
        node_dim = SAMPLE_ENC.data.x.size(1)  # type: ignore
        global_dim = SAMPLE_ENC.data.global_feats.size(1)
        load_model(model_path, node_dim, global_dim, device=device)

    model_sha256: str | None = None
    try:
        p = Path(model_path)
        if p.exists() and p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            model_sha256 = h.hexdigest()
    except Exception:
        model_sha256 = None

    t0 = time.time()
    eval_jobs = max(1, int(eval_jobs))
    chunks = _split_indices(int(eval_games), eval_jobs)
    summaries: list[dict] = []
    # When using multiple processes, cap Torch threads per process to avoid CPU oversubscription.
    torch_threads = 1 if int(eval_jobs) > 1 else 0

    if eval_jobs <= 1:
        summaries.append(
            _eval_vs_random_worker(
                {
                    "backend": str(backend),
                    "device": str(device),
                    "eval_rollouts": int(eval_rollouts),
                    "eval_max_moves": int(eval_max_moves),
                    "eval_seed": int(eval_seed),
                    "eval_randomize_start": bool(eval_randomize_start),
                    "model_path": str(model_path),
                    "policy_path": str(policy_path) if policy_path else "",
                    "prior_scale": float(prior_scale),
                    "prior_mix_uniform": float(prior_mix_uniform),
                    "cpp_verbose": int(cpp_verbose),
                    "cpp_use_nn_value": bool(cpp_use_nn_value),
                    "torch_threads": int(torch_threads),
                    "filter_near_area": bool(filter_near_area),
                    "filter_probe_rollouts": int(filter_probe_rollouts),
                    "filter_probe_moves": int(filter_probe_moves),
                    "randomize_max_sticks": int(randomize_max_sticks),
                    "gidxs": list(range(int(eval_games))),
                }
            )
        )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=eval_jobs, mp_context=ctx) as ex:
            futs = []
            for chunk in chunks:
                futs.append(
                    ex.submit(
                        _eval_vs_random_worker,
                        {
                            "backend": str(backend),
                            "device": str(device),
                            "eval_rollouts": int(eval_rollouts),
                            "eval_max_moves": int(eval_max_moves),
                            "eval_seed": int(eval_seed),
                            "eval_randomize_start": bool(eval_randomize_start),
                            "model_path": str(model_path),
                            "policy_path": str(policy_path) if policy_path else "",
                            "prior_scale": float(prior_scale),
                            "prior_mix_uniform": float(prior_mix_uniform),
                            "cpp_verbose": int(cpp_verbose),
                            "cpp_use_nn_value": bool(cpp_use_nn_value),
                            "torch_threads": int(torch_threads),
                            "filter_near_area": bool(filter_near_area),
                            "filter_probe_rollouts": int(filter_probe_rollouts),
                            "filter_probe_moves": int(filter_probe_moves),
                            "randomize_max_sticks": int(randomize_max_sticks),
                            "gidxs": chunk,
                        },
                    )
                )
            for f in as_completed(futs):
                summaries.append(f.result())

    wins = sum(int(s.get("wins", 0)) for s in summaries)
    losses = sum(int(s.get("losses", 0)) for s in summaries)
    draws = sum(int(s.get("draws", 0)) for s in summaries)
    max_moves_reached = sum(int(s.get("max_moves_reached", 0)) for s in summaries)
    move_counts: list[int] = []
    first_agent_move_counts: dict[str, int] = {}
    cpp_profile_sum: dict[str, float | int | bool] = {}
    for s in summaries:
        move_counts.extend(list(s.get("move_counts", [])))
        first_agent_move_counts = _merge_counts(first_agent_move_counts, dict(s.get("first_agent_move_counts", {})))
        # Profiling (C++ backend only).
        prof = dict(s.get("cpp_profile_sum", {}))
        for k, v in prof.items():
            if isinstance(v, bool):
                cpp_profile_sum[k] = bool(v)
            elif isinstance(v, (int, float)):
                cpp_profile_sum[k] = float(cpp_profile_sum.get(k, 0.0)) + float(v)

    dt = time.time() - t0
    games = max(1, int(eval_games))
    avg_moves = float(sum(move_counts)) / float(max(1, len(move_counts)))
    win_rate = float(wins) / float(games)

    top_first_moves = sorted(first_agent_move_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    first_agent_moves_top = [{"move": k, "count": int(v)} for k, v in top_first_moves]

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "iter": int(iteration),
        "backend": str(backend),
        "device": str(device),
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "first_agent_moves_top": first_agent_moves_top,
        "cpp_profile": cpp_profile_sum
        if backend == 'cpp'
        else None,
        "eval_games": int(eval_games),
        "eval_rollouts": int(eval_rollouts),
        "eval_max_moves": int(eval_max_moves),
        "eval_seed": int(eval_seed),
        "eval_randomize_start": bool(eval_randomize_start),
        "eval_alternate_roles": True,
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "win_rate": win_rate,
        "avg_moves": f"{avg_moves:.3f}",
        "max_moves_reached": int(max_moves_reached),
        "seconds": float(dt),
    }


def _evaluate_cpp_model_vs_model(
    *,
    device: str,
    eval_games: int,
    eval_rollouts: int,
    eval_max_moves: int,
    eval_seed: int,
    iteration: int,
    model_a_path: str,
    model_b_path: str,
    policy_a_path: str | None = None,
    policy_b_path: str | None = None,
    model_a_label: str,
    model_b_label: str,
    cpp_verbose: int,
    cpp_use_nn_value: bool,
    eval_jobs: int = 1,
    filter_near_area: bool = False,
    filter_probe_rollouts: int = 200,
    filter_probe_moves: int = 6,
    randomize_max_sticks: int = 5,
) -> dict:
    t0 = time.time()
    eval_jobs = max(1, int(eval_jobs))
    chunks = _split_indices(int(eval_games), eval_jobs)
    summaries: list[dict] = []
    torch_threads = 1 if int(eval_jobs) > 1 else 0
    if eval_jobs <= 1:
        summaries.append(
            _eval_cpp_model_vs_model_worker(
                {
                    "device": str(device),
                    "eval_rollouts": int(eval_rollouts),
                    "eval_max_moves": int(eval_max_moves),
                    "eval_seed": int(eval_seed),
                    "model_a_path": str(model_a_path),
                    "model_b_path": str(model_b_path),
                    "policy_a_path": str(policy_a_path) if policy_a_path else "",
                    "policy_b_path": str(policy_b_path) if policy_b_path else "",
                    "cpp_verbose": int(cpp_verbose),
                    "cpp_use_nn_value": bool(cpp_use_nn_value),
                    "torch_threads": int(torch_threads),
                    "filter_near_area": bool(filter_near_area),
                    "filter_probe_rollouts": int(filter_probe_rollouts),
                    "filter_probe_moves": int(filter_probe_moves),
                    "randomize_max_sticks": int(randomize_max_sticks),
                    "gidxs": list(range(int(eval_games))),
                }
            )
        )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=eval_jobs, mp_context=ctx) as ex:
            futs = []
            for chunk in chunks:
                futs.append(
                    ex.submit(
                        _eval_cpp_model_vs_model_worker,
                        {
                            "device": str(device),
                            "eval_rollouts": int(eval_rollouts),
                            "eval_max_moves": int(eval_max_moves),
                            "eval_seed": int(eval_seed),
                            "model_a_path": str(model_a_path),
                            "model_b_path": str(model_b_path),
                            "policy_a_path": str(policy_a_path) if policy_a_path else "",
                            "policy_b_path": str(policy_b_path) if policy_b_path else "",
                            "cpp_verbose": int(cpp_verbose),
                            "cpp_use_nn_value": bool(cpp_use_nn_value),
                            "torch_threads": int(torch_threads),
                            "filter_near_area": bool(filter_near_area),
                            "filter_probe_rollouts": int(filter_probe_rollouts),
                            "filter_probe_moves": int(filter_probe_moves),
                            "randomize_max_sticks": int(randomize_max_sticks),
                            "gidxs": chunk,
                        },
                    )
                )
            for f in as_completed(futs):
                summaries.append(f.result())

    wins = sum(int(s.get("wins", 0)) for s in summaries)
    losses = sum(int(s.get("losses", 0)) for s in summaries)
    draws = sum(int(s.get("draws", 0)) for s in summaries)
    max_moves_reached = sum(int(s.get("max_moves_reached", 0)) for s in summaries)
    move_counts: list[int] = []
    for s in summaries:
        move_counts.extend(list(s.get("move_counts", [])))

    dt = time.time() - t0
    games = max(1, int(eval_games))
    avg_moves = float(sum(move_counts)) / float(max(1, len(move_counts)))
    win_rate = float(wins) / float(games)

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "iter": int(iteration),
        "backend": "cpp",
        "device": str(device),
        "match": "checkpoint_vs_checkpoint",
        "model_a": str(model_a_path),
        "model_b": str(model_b_path),
        "model_a_label": str(model_a_label),
        "model_b_label": str(model_b_label),
        "eval_games": int(eval_games),
        "eval_rollouts": int(eval_rollouts),
        "eval_max_moves": int(eval_max_moves),
        "eval_seed": int(eval_seed),
        "eval_alternate_roles": True,
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "win_rate": win_rate,
        "avg_moves": avg_moves,
        "max_moves_reached": int(max_moves_reached),
        "seconds": float(dt),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0, help='Base RNG seed for self-play/training determinism (0 is allowed).')
    p.add_argument('--iters', type=int, default=3)
    p.add_argument('--games', type=int, default=50)
    p.add_argument('--rollouts', type=int, default=100)
    p.add_argument('--selfplay-temp', type=float, default=1.1, help='Self-play temperature for sampling from visit counts (C++ backend only).')
    p.add_argument('--selfplay-temp-moves', type=int, default=20, help='Number of opening moves to apply temperature sampling (C++ backend only).')
    p.add_argument('--selfplay-dirichlet-alpha', type=float, default=0.3, help='Dirichlet alpha for root noise (C++ backend only).')
    p.add_argument('--selfplay-dirichlet-epsilon', type=float, default=0.25, help='Dirichlet epsilon for root noise (C++ backend only).')
    p.add_argument('--selfplay-temp-decay', type=float, default=1.0, help='Multiply self-play temperature by this each iteration after --selfplay-decay-start.')
    p.add_argument('--selfplay-epsilon-decay', type=float, default=1.0, help='Multiply self-play Dirichlet epsilon by this each iteration after --selfplay-decay-start.')
    p.add_argument('--selfplay-decay-start', type=int, default=0, help='Iteration index (1-based) after which to start decaying self-play exploration params. 0 means start immediately.')
    p.add_argument('--selfplay-min-temp', type=float, default=0.0, help='Floor for decayed self-play temperature.')
    p.add_argument('--selfplay-min-epsilon', type=float, default=0.0, help='Floor for decayed self-play Dirichlet epsilon.')
    p.add_argument('--selfplay-vs-random-prob', type=float, default=0.0, help='Probability a self-play game uses a random opponent (targets recorded only for MCTS moves).')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--value-weight', type=float, default=1.0, help='Weight applied to value loss during training')
    p.add_argument('--value-lr-mult', type=float, default=1.0, help='LR multiplier for the value head during training')
    p.add_argument('--device', default='cpu')
    p.add_argument('--train-num-workers', type=int, default=0, help='DataLoader workers for training subprocess')
    p.add_argument('--train-prefetch-factor', type=int, default=2, help='Prefetch factor for training subprocess (requires train-num-workers>0)')
    p.add_argument('--train-persistent-workers', action='store_true', help='Enable persistent DataLoader workers in training subprocess (requires train-num-workers>0)')
    p.add_argument('--train-steps-per-epoch', type=int, default=0, help='If >0, cap training batches per epoch in subprocess (stability on MPS)')
    p.add_argument('--out-dir', default='checkpoints')
    p.add_argument('--init-eval-model', type=str, default=None, help='Evaluator checkpoint to use for iteration 1 self-play, If not provided, a random-initialized GNNEval checkpoint is created/used in --out-dir.')
    p.add_argument('--init-policy-from', type=str, default=None, help='Checkpoint to warm-start the PolicyValueNet from at iteration 1. Accepts a GNNEval or PolicyValueNet checkpoint (GNNEval value head keys are remapped automatically).')
    p.add_argument(
        '--auto-init-policy',
        action='store_true',
        default=False,
        help='If set and using --backend=cpp, create a random PolicyValueNet checkpoint for priors when no previous policy checkpoint exists (opt-in).',
    )
    p.add_argument('--saved-games-dir', default='rl_self_play/iter_{iter}')
    p.add_argument('--data-dir', default='data')
    p.add_argument('--dataset-shard-size', type=int, default=0, help='If >0, write training dataset in shards of this many samples (reduces peak memory). Suggested: 20000-50000 for ~300k+ samples.')
    p.add_argument(
        '--dataset-grouped-policy',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If true, store one graph per decision with K moves (faster training; avoids duplicating graphs).',
    )
    p.add_argument('--dataset-policy-topk', type=int, default=0, help='If >0, keep only the top-K policy targets per decision when converting games (reduces dataset size & training time).')
    p.add_argument('--dataset-policy-min-prob', type=float, default=0.0, help='If >0, drop policy targets below this probability when converting games (after normalization).')
    p.add_argument('--dataset-policy-topk-start', type=int, default=0, help='If >0 and --dataset-policy-topk is 0, schedule top-K starting from this value (gradual schedule).')
    p.add_argument('--dataset-policy-topk-end', type=int, default=0, help='Scheduled top-K value at --dataset-policy-topk-end-iter. Use 0 to disable filtering by the end of the schedule.')
    p.add_argument('--dataset-policy-topk-start-iter', type=int, default=0, help='Iteration (1-based) to start scheduling dataset-policy-topk. 0 means start at the run\'s first iteration.')
    p.add_argument('--dataset-policy-topk-end-iter', type=int, default=0, help='Iteration (1-based) to finish scheduling dataset-policy-topk. 0 means end at the run\'s last iteration.')
    p.add_argument('--dataset-policy-topk-curve', type=str, default='cosine', choices=['linear', 'cosine'], help='Curve for scheduled dataset-policy-topk (cosine is smooth).')
    p.add_argument('--replay-window', type=int, default=1, help='How many most-recent iteration datasets to train on each iteration (replay buffer). 1 means only current iteration.')
    p.add_argument('--keep-last-datasets', type=int, default=1, help='How many most-recent iteration datasets to keep on disk (pt + .shards). Older iterations are deleted after a successful iteration.')
    p.add_argument('--eval-games', type=int, default=100, help='Number of games to evaluate vs RandomPlayer after each iteration')
    p.add_argument('--eval-rollouts', type=int, default=500, help='MCTS rollouts to use during evaluation')
    p.add_argument('--eval-jobs', type=int, default=1, help='Parallel worker processes for evaluation (CPU). 1 disables parallelism.')
    p.add_argument('--eval-heavy-every', type=int, default=0, help='If >0, run a heavier vs-Random eval every N iterations (in addition to the light eval).')
    p.add_argument('--eval-heavy-games', type=int, default=0, help='Games for the heavy vs-Random eval (requires --eval-heavy-every>0).')
    p.add_argument('--eval-heavy-rollouts', type=int, default=0, help='Rollouts for the heavy vs-Random eval (requires --eval-heavy-every>0).')
    p.add_argument('--eval-max-moves', type=int, default=256, help='Max moves per evaluation game before counting as draw')
    p.add_argument('--eval-seed', type=int, default=12345, help='Base RNG seed for deterministic evaluation starting positions')
    p.add_argument(
        '--eval-randomize-start',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If true, call randomize_start() before each eval game.',
    )
    p.add_argument('--eval-games-low', type=int, default=0, help='If >0, run an additional vs-Random eval at --eval-rollouts-low')
    p.add_argument('--eval-rollouts-low', type=int, default=0, help='Rollouts for the additional low-rollouts eval (requires --eval-games-low>0)')
    p.add_argument('--strength-log', default='logs/strength_curve.jsonl', help='Append JSONL evaluation records here each iteration')
    p.add_argument('--eval-vs-prev', type=int, default=0, help='If >0 and --backend=cpp: also evaluate vs the last N previous evaluator checkpoints')
    p.add_argument('--eval-vs-prev-every', type=int, default=1, help='Run --eval-vs-prev only every N iterations (C++ backend only).')
    p.add_argument('--eval-prev-games', type=int, default=40, help='Games per previous-checkpoint opponent')
    p.add_argument('--eval-prev-rollouts', type=int, default=200, help='MCTS rollouts per move for previous-checkpoint evaluation')
    p.add_argument('--no-augment', action='store_true', help='Disable symmetric augmentation during conversion')
    p.add_argument('--backend', type=str, default='python', choices=['python', 'cpp'], help='MCTS backend for self-play/eval')
    p.add_argument(
        '--cpp-verbose',
        type=int,
        default=0,
        help='If --backend=cpp: integer verbosity for C++ MCTS (0=silent,1=summaries,2=debug).',
    )
    p.add_argument(
        '--cpp-use-nn-value',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='If --backend=cpp: use the neural net for leaf value evaluation (AlphaZero-style).',
    )
    p.add_argument('--prior-scale', type=float, default=1.0, help='Scale applied to policy priors before mixing (C++ backend only).')
    p.add_argument('--prior-mix-uniform', type=float, default=0.04, help='Weight of uniform mix applied to policy priors (0.0 disables) (C++ backend only).')
    p.add_argument(
        '--filter-near-area',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Reject randomized starting positions where a short MCTS probe finds a quick win. '
             'Uses --filter-probe-rollouts and --filter-probe-moves to control the probe. '
             'Requires --backend=cpp.',
    )
    p.add_argument('--filter-probe-rollouts', type=int, default=200, help='MCTS rollouts per move for the quick-win probe (requires --filter-near-area).')
    p.add_argument('--filter-probe-moves', type=int, default=6, help='Max half-moves for the quick-win probe (requires --filter-near-area).')
    p.add_argument('--randomize-max-sticks', type=int, default=5, help='Maximum number of sticks placed during randomize_start (default 5). Lower values reduce near-forced starting positions.')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # detect existing checkpoints and choose starting iteration so numbering continues
    az_ckpts = list(out_dir.glob('gnn_az_iter_*.pt'))
    max_existing = 0
    for p in az_ckpts:
        m = re.search(r'gnn_az_iter_(\d+)\.pt$', p.name)
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

    for i in range(start_iter, end_iter + 1):
        print(f"=== Iteration {i}/{end_iter} ===")

        base_seed = int(args.seed)
        # Per-iteration seed namespace (stable across resume when iteration number is the same).
        iter_seed = base_seed + (100_000 * int(i))

        saved_games_dir = args.saved_games_dir.format(iter=i)
        saved_games_path = Path(saved_games_dir)
        saved_games_path.mkdir(parents=True, exist_ok=True)

        def _ensure_random_init_eval(out_dir_path: Path) -> Path:
            from gnn.model import GNNEval

            init_path = out_dir_path / 'gnn_eval_init_random.pt'
            if init_path.exists() and init_path.is_file():
                return init_path

            node_dim = int(SAMPLE_ENC.data.x.size(1))  # type: ignore
            global_dim = int(SAMPLE_ENC.data.global_feats.size(1))

            # Deterministic random init (makes runs reproducible unless user supplies a checkpoint).
            torch.manual_seed(0)
            model = GNNEval(node_feat_dim=node_dim, global_feat_dim=global_dim)
            torch.save(model.state_dict(), init_path)
            print(f"Created random-init GNNEval evaluator at {init_path}")
            return init_path

        # Resolve the evaluator checkpoint used for self-play.
        # - Iteration > 1: prefer previous iteration's evaluator; otherwise fall back.
        # - Iteration == 1: use --init-eval-model if provided; else random-init.
        out_dir_path = Path(args.out_dir)
        default_ckpt_dir = Path('checkpoints')

        evaluator_path: Path | None = None
        if i == 1:
            if args.init_eval_model is not None:
                cand = Path(str(args.init_eval_model))
                if not cand.exists():
                    raise FileNotFoundError(f"--init-eval-model not found: {cand}")
                evaluator_path = cand
            else:
                evaluator_path = _ensure_random_init_eval(out_dir_path)
        else:
            candidates: list[Path] = [
                out_dir_path / f'gnn_eval_iter_{i-1}.pt',
                default_ckpt_dir / f'gnn_eval_iter_{i-1}.pt',
            ]
            evaluator_path = next((p for p in candidates if p.exists()), None)
            if evaluator_path is None:
                # If we can't find the previous evaluator, fall back to init model if provided,
                # else random-init, else the shipped balanced checkpoint.
                if args.init_eval_model is not None and Path(str(args.init_eval_model)).exists():
                    evaluator_path = Path(str(args.init_eval_model))
                else:
                    evaluator_path = _ensure_random_init_eval(out_dir_path)
                    if evaluator_path is None:
                        for fb in (out_dir_path / 'gnn_eval_balanced.pt', default_ckpt_dir / 'gnn_eval_balanced.pt'):
                            if fb.exists():
                                evaluator_path = fb
                                break

        model_path = str(evaluator_path) if evaluator_path is not None else None

        # For C++ self-play priors, prefer the previous PolicyValueNet checkpoint when available.
        policy_path = None
        if i > 1:
            cand = out_dir_path / f'gnn_az_iter_{i-1}.pt'
            if cand.exists():
                policy_path = str(cand)

        # If using the C++ backend, require a policy checkpoint for priors.
        # If `--auto-init-policy` is set, create a random-initialized PolicyValueNet
        # checkpoint (deterministic seed) and use it as the policy path. Otherwise
        # fail fast to avoid unsafe fallbacks in the C++ engine.
        if str(args.backend) == 'cpp' and not policy_path:
            if bool(args.auto_init_policy):
                try:
                    from rl.train import PolicyValueNet

                    node_dim = int(SAMPLE_ENC.data.x.size(1))  # type: ignore
                    global_dim = int(SAMPLE_ENC.data.global_feats.size(1))  # type: ignore
                    move_feat_dim = 16  # must match move feature tuple produced by C++ encoder
                    torch.manual_seed(0)
                    model = PolicyValueNet(node_feat_dim=node_dim, global_feat_dim=global_dim, move_feat_dim=move_feat_dim)
                    cand = out_dir_path / f'gnn_az_iter_{i-1}.pt'
                    torch.save(model.state_dict(), cand)
                    policy_path = str(cand)
                    print(f"Auto-initialized random policy checkpoint at {cand}")
                except Exception as e:
                    raise RuntimeError(f"Failed to auto-initialize policy checkpoint: {e}")
            else:
                raise RuntimeError(
                    "C++ backend requires a policy AZ checkpoint for priors."
                    " Ensure 'gnn_az_iter_{i-1}.pt' exists in the output directory before running, or pass --auto-init-policy to create a temporary random policy."
                )

        decay_start = int(args.selfplay_decay_start)
        age = int(i) - 1
        if decay_start > 0:
            age = max(0, int(i) - int(decay_start))

        temp_eff = float(args.selfplay_temp) * (float(args.selfplay_temp_decay) ** float(age))
        eps_eff = float(args.selfplay_dirichlet_epsilon) * (float(args.selfplay_epsilon_decay) ** float(age))
        temp_eff = float(max(float(args.selfplay_min_temp), temp_eff))
        eps_eff = float(max(float(args.selfplay_min_epsilon), eps_eff))

        if str(args.backend) == 'cpp':
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

        dataset_path = data_dir / f'alpha_dataset_iter_{i}.pt'
        shard_size = int(args.dataset_shard_size) if int(args.dataset_shard_size) > 0 else None
        # Determine effective top-K policy target filtering for dataset conversion.
        fixed_topk = int(args.dataset_policy_topk)
        if fixed_topk > 0:
            topk_eff = fixed_topk
        else:
            start_k = int(args.dataset_policy_topk_start)
            end_k = int(args.dataset_policy_topk_end)
            if start_k > 0 or end_k > 0:
                sched_start = int(args.dataset_policy_topk_start_iter) if int(args.dataset_policy_topk_start_iter) > 0 else int(start_iter)
                sched_end = int(args.dataset_policy_topk_end_iter) if int(args.dataset_policy_topk_end_iter) > 0 else int(end_iter)
                topk_eff = _scheduled_int(
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
        if topk is not None:
            print(f"Dataset conversion: policy_topk={topk} (iter={i})")
        else:
            print(f"Dataset conversion: policy_topk=ALL (iter={i})")
        convert_games_to_dataset(
            str(saved_games_path),
            str(dataset_path),
            augment=not args.no_augment,
            shard_size=shard_size,
            policy_topk=topk,
            policy_min_prob=float(args.dataset_policy_min_prob),
            grouped_policy=bool(args.dataset_grouped_policy),
        )

        policy_ckpt = out_dir / f'gnn_az_iter_{i}.pt'

        # Train on a replay buffer of recent datasets.
        replay_window = max(1, int(args.replay_window))
        train_dataset_paths: list[str] = []
        for j in range(max(1, int(i) - replay_window + 1), int(i) + 1):
            dp = data_dir / f'alpha_dataset_iter_{j}.pt'
            if dp.exists():
                train_dataset_paths.append(str(dp))
        if not train_dataset_paths:
            train_dataset_paths = [str(dataset_path)]

        # Warm-start from the previous iteration's policy checkpoint when available.
        init_from = None
        if i > 1:
            prev_ckpt = out_dir / f'gnn_az_iter_{i-1}.pt'
            if prev_ckpt.exists():
                init_from = str(prev_ckpt)
        elif i == 1 and args.init_policy_from is not None:
            cand = Path(str(args.init_policy_from))
            if not cand.exists():
                raise FileNotFoundError(f"--init-policy-from not found: {cand}")
            init_from = str(cand)
            print(f"Iteration 1: warm-starting PolicyValueNet from {init_from}")

        # Run training in a fresh process so macOS/MPS cached memory is released when training finishes.
        gc.collect()
        train_cmd = [
            sys.executable,
            '-m',
            'rl.train',
            '--dataset',
            *train_dataset_paths,
            '--epochs',
            str(args.epochs),
            '--batch-size',
            str(args.batch_size),
            '--lr',
            str(args.lr),
            '--value-weight',
            str(args.value_weight),
            '--value-lr-mult',
            str(args.value_lr_mult),
            '--device',
            str(args.device),
            '--out',
            str(policy_ckpt),
            '--seed',
            str(int(iter_seed) + 7),
            '--init-from',
            str(init_from) if init_from is not None else '',
        ]

        dev = str(args.device)

        # Short smoke diagnostic run: 1 epoch, capped steps (100), small batch,
        # write per-batch diagnostics to logs/run_loop_diagnostics.json. Run this
        # before the full training to capture training signal quickly.
        try:
            diag_out = Path('logs/run_loop_diagnostics.json')
            diag_cmd = [
                sys.executable,
                '-m',
                'rl.train',
                '--dataset',
                *train_dataset_paths,
                '--epochs',
                '1',
                '--steps-per-epoch',
                '100',
                '--batch-size',
                str(max(1, min(16, int(args.batch_size)))),
                '--lr',
                str(args.lr),
                '--value-weight',
                str(args.value_weight),
                '--value-lr-mult',
                str(args.value_lr_mult),
                '--device',
                str(args.device),
                '--out',
                str(out_dir / f'gnn_az_diag_iter_{i}.pt'),
                '--seed',
                str(int(iter_seed) + 7),
                '--diagnostics-out',
                str(diag_out),
            ]
            # include init-from only when warm-starting
            if init_from is not None:
                diag_cmd += ['--init-from', str(init_from)]
            if dev == 'mps' or dev.startswith('cuda'):
                diag_cmd.append('--amp')
            print('Running short diagnostic training subprocess:', ' '.join(diag_cmd))
            subprocess.run(diag_cmd, check=True)
        except Exception as e:
            print(f"Diagnostic training run failed (continuing): {e}")

        # Drop the flag entirely when not warm-starting.
        if init_from is None:
            # remove the last two items: ['--init-from', '']
            train_cmd = train_cmd[:-2]
        # Speed: enable AMP automatically on mps/cuda.
        if dev == 'mps' or dev.startswith('cuda'):
            train_cmd.append('--amp')
        # Speed: allow DataLoader workers for faster collation.
        if int(args.train_num_workers) > 0:
            train_cmd += ['--num-workers', str(int(args.train_num_workers))]
            train_cmd += ['--prefetch-factor', str(int(args.train_prefetch_factor))]
            if args.train_persistent_workers:
                train_cmd += ['--persistent-workers']
        if int(args.train_steps_per_epoch) > 0:
            train_cmd += ['--steps-per-epoch', str(int(args.train_steps_per_epoch))]
        print('Running training subprocess:', ' '.join(train_cmd))
        try:
            subprocess.run(train_cmd, check=True)
        except subprocess.CalledProcessError as e:
            # On macOS, DataLoader multiprocessing can be fragile and can get SIGKILLed under memory pressure,
            # leaving leaked semaphores. Fall back to single-process loading to keep the loop going.
            if e.returncode in (-9, 137):
                print('Training subprocess was SIGKILLed; retrying with --num-workers 0...')
                cleaned: list[str] = []
                skip_next = False
                for idx, c in enumerate(train_cmd):
                    if skip_next:
                        skip_next = False
                        continue
                    if c in ('--num-workers', '--prefetch-factor'):
                        skip_next = True
                        continue
                    if c == '--persistent-workers':
                        continue
                    cleaned.append(c)
                print('Retry command:', ' '.join(cleaned))
                try:
                    subprocess.run(cleaned, check=True)
                except subprocess.CalledProcessError as e2:
                    if e2.returncode in (-9, 137):
                        # Last-resort: disable AMP and cap steps-per-epoch + batch size.
                        print('Retry was SIGKILLed; retrying with smaller batch and capped steps (no AMP)...')
                        safe_bs = max(1, int(args.batch_size) // 2)
                        safe_steps = int(args.train_steps_per_epoch) if int(args.train_steps_per_epoch) > 0 else 2000
                        fallback_cmd = [
                            sys.executable,
                            '-m',
                            'rl.train',
                            '--dataset',
                            *train_dataset_paths,
                            '--epochs',
                            str(args.epochs),
                            '--batch-size',
                            str(safe_bs),
                            '--lr',
                            str(args.lr),
                            '--device',
                            str(args.device),
                            '--out',
                            str(policy_ckpt),
                            '--steps-per-epoch',
                            str(safe_steps),
                            '--clear-cache-interval',
                            '50',
                        ]

                        if init_from is not None:
                            fallback_cmd.extend(['--init-from', str(init_from)])
                        print('Fallback command:', ' '.join(fallback_cmd))
                        subprocess.run(fallback_cmd, check=True)
                    else:
                        raise
            else:
                raise

        print("Training completed for iteration", i, "cleaning up old datasets...")

        _cleanup_old_datasets(
            data_dir,
            keep_last=max(int(args.keep_last_datasets), int(replay_window)),
            current_iter=i,
        )

        gc.collect()
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

        gnn_eval_ckpt = out_dir / f'gnn_eval_iter_{i}.pt'
        export_gnn_eval_from_policy(policy_ckpt, gnn_eval_ckpt)

        # Collect proof fields for the strength log.
        policy_sha256: str | None = None
        try:
            if policy_ckpt.exists() and policy_ckpt.is_file():
                h = hashlib.sha256()
                with policy_ckpt.open('rb') as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                        h.update(chunk)
                policy_sha256 = h.hexdigest()
        except Exception:
            policy_sha256 = None

        if args.eval_games and args.eval_games > 0:
            try:
                node_dim = SAMPLE_ENC.data.x.size(1) # type: ignore
                global_dim = SAMPLE_ENC.data.global_feats.size(1)
                load_model(str(gnn_eval_ckpt), node_dim, global_dim, device=args.device)
                print(f"Loaded evaluator {gnn_eval_ckpt} for evaluation")
            except Exception as e:
                print(f"Failed to load evaluator for evaluation: {e}")
                continue

            record = _evaluate_vs_random(
                backend=str(args.backend),
                device=str(args.device),
                eval_games=int(args.eval_games),
                eval_rollouts=int(args.eval_rollouts),
                eval_max_moves=int(args.eval_max_moves),
                eval_seed=int(args.eval_seed),
                eval_randomize_start=bool(args.eval_randomize_start),
                iteration=int(i),
                model_path=str(gnn_eval_ckpt),
                policy_path=str(policy_ckpt) if str(args.backend) == 'cpp' else None,
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

            record["policy_ckpt"] = str(policy_ckpt)
            record["policy_sha256"] = policy_sha256
            record["policy_init_from"] = str(init_from) if init_from is not None else None
            record["train_datasets"] = list(train_dataset_paths)
            record["replay_window"] = int(replay_window)

            # Optional: low-rollout eval (more sensitive to evaluator differences).
            if int(args.eval_games_low) > 0 and int(args.eval_rollouts_low) > 0:
                low = _evaluate_vs_random(
                    backend=str(args.backend),
                    device=str(args.device),
                    eval_games=int(args.eval_games_low),
                    eval_rollouts=int(args.eval_rollouts_low),
                    eval_max_moves=int(args.eval_max_moves),
                    eval_seed=int(args.eval_seed) + 777777,
                    eval_randomize_start=bool(args.eval_randomize_start),
                    iteration=int(i),
                    model_path=str(gnn_eval_ckpt),
                    policy_path=str(policy_ckpt) if str(args.backend) == 'cpp' else None,
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

            # Optional: ladder eval vs prior checkpoints (C++ backend only).
            if int(args.eval_vs_prev) > 0:
                if str(args.backend) != 'cpp':
                    print('Note: --eval-vs-prev requires --backend=cpp (needs per-engine checkpoint loading).')
                elif i <= 1:
                    pass
                elif int(args.eval_vs_prev_every) > 1 and (int(i) % int(args.eval_vs_prev_every) != 0):
                    pass
                else:
                    prev_n = int(args.eval_vs_prev)
                    prev_games = int(args.eval_prev_games)
                    prev_rollouts = int(args.eval_prev_rollouts)
                    start_opp = max(0, int(i) - prev_n)
                    vs_prev: dict[str, dict] = {}
                    for opp_iter in range(start_opp, int(i)):
                        opp_ckpt = out_dir / f'gnn_eval_iter_{opp_iter}.pt' if opp_iter > 0 else out_dir / 'gnn_eval_balanced.pt'
                        if not opp_ckpt.exists():
                            continue
                        print(f"Evaluating vs checkpoint iter {opp_iter}...")
                        res = _evaluate_cpp_model_vs_model(
                            device=str(args.device),
                            eval_games=prev_games,
                            eval_rollouts=prev_rollouts,
                            eval_max_moves=int(args.eval_max_moves),
                            eval_seed=int(args.eval_seed) + 100000 * int(opp_iter),
                            iteration=int(i),
                            model_a_path=str(gnn_eval_ckpt),
                            model_b_path=str(opp_ckpt),
                            policy_a_path=str(policy_ckpt) if policy_ckpt.exists() else None,
                            policy_b_path=str(out_dir / f'gnn_az_iter_{opp_iter}.pt') if (opp_iter > 0 and (out_dir / f'gnn_az_iter_{opp_iter}.pt').exists()) else None,
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

            # Optional: heavy vs-Random eval every N iterations.
            if int(args.eval_heavy_every) > 0 and int(args.eval_heavy_games) > 0 and int(args.eval_heavy_rollouts) > 0:
                if int(i) % int(args.eval_heavy_every) == 0:
                    heavy = _evaluate_vs_random(
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

            print(
                "Evaluation vs RandomPlayer: "
                f"{record['wins']}/{record['eval_games']} wins, "
                f"{record['losses']} losses, {record['draws']} draws "
                f"(win_rate={record['win_rate']:.2f}, avg_moves={record['avg_moves']})"
            )
            try:
                _append_jsonl(Path(str(args.strength_log)), record)
                print(f"Appended strength record to {args.strength_log}")
            except Exception as e:
                print(f"Warning: failed to write strength log {args.strength_log}: {e}")

    print('All iterations completed')


if __name__ == '__main__':
    main()
