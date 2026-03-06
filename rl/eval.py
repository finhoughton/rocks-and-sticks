from __future__ import annotations

import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import torch

from gnn.encode import SAMPLE_ENC
from gnn.game_generation import randomize_start
from gnn.model import load_model
from players.base import RandomPlayer
from rl.utils import merge_counts, split_indices


# ---------------------------------------------------------------------------
# AB engine vs Random
# ---------------------------------------------------------------------------

def evaluate_ab_vs_random(
    *,
    eval_games: int,
    eval_max_moves: int,
    eval_seed: int,
    eval_randomize_start: bool,
    iteration: int,
    ab_time_limit_ms: int,
    ab_depth: int,
    ab_max_depth: int,
    ab_move_cap: int,
    ab_use_heuristic: bool,
    native_model: str = "",
    nn_ordering_depth: int = 3,
    randomize_max_sticks: int = 5,
) -> dict:
    """Evaluate the AB engine vs RandomPlayer (matching ab-supervised generation settings)."""
    import players_ext

    from players.alphabeta_cpp import AlphaBetaPlayerCPP
    from players.game_total import GameTotal

    t0 = time.time()
    wins = 0
    losses = 0
    draws = 0
    max_moves_reached = 0
    move_counts: list[int] = []
    first_agent_move_counts: dict[str, int] = {}

    for gidx in range(int(eval_games)):
        random.seed(int(eval_seed) + int(gidx))
        agent_player = 0 if (gidx % 2 == 0) else 1
        random_player = 1 - agent_player

        base_seed = int(eval_seed) + int(gidx) * 17

        ab = AlphaBetaPlayerCPP(
            agent_player,
            seed=base_seed,
            depth=int(ab_depth),
            move_cap=int(ab_move_cap),
            time_limit_ms=int(ab_time_limit_ms),
            max_depth=int(ab_max_depth),
            use_heuristic=bool(ab_use_heuristic),
            native_model=str(native_model),
            nn_ordering_depth=int(nn_ordering_depth),
        )
        rnd = RandomPlayer(random_player, seed=int(eval_seed) + int(gidx) + 1)
        players_list = [ab, rnd] if agent_player == 0 else [rnd, ab]

        from game import Game
        game = GameTotal(Game(players=players_list), players_ext.GameState())

        if bool(eval_randomize_start):
            randomize_start(game, max_sticks=int(randomize_max_sticks))

        move_count = 0
        seen_agent_move = False
        while game.winner is None and move_count < int(eval_max_moves):
            cp = game.current_player
            mv = game.players[cp].get_move(game)

            if (not seen_agent_move) and (cp == agent_player):
                seen_agent_move = True
                mk = f"{mv.c[0]},{mv.c[1]},{mv.t}"
                first_agent_move_counts[mk] = first_agent_move_counts.get(mk, 0) + 1

            game.do_move(cp, mv)
            move_count += 1

        move_counts.append(move_count)
        if game.winner == agent_player:
            wins += 1
        elif game.winner == random_player:
            losses += 1
        else:
            draws += 1
            if move_count >= int(eval_max_moves):
                max_moves_reached += 1

    dt = time.time() - t0
    games = max(1, int(eval_games))
    avg_moves = float(sum(move_counts)) / float(max(1, len(move_counts)))
    win_rate = float(wins) / float(games)

    top_first_moves = sorted(first_agent_move_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]
    first_agent_moves_top = [{"move": k, "count": int(v)} for k, v in top_first_moves]

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "iter": int(iteration),
        "backend": "cpp",
        "device": "cpu",
        "match": "ab_vs_random",
        "eval_engine": "alphabeta",
        "model_path": None,
        "model_sha256": None,
        "ab_time_limit_ms": int(ab_time_limit_ms),
        "ab_depth": int(ab_depth),
        "ab_max_depth": int(ab_max_depth),
        "ab_move_cap": int(ab_move_cap),
        "ab_use_heuristic": bool(ab_use_heuristic),
        "first_agent_moves_top": first_agent_moves_top,
        "eval_games": int(eval_games),
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


# ---------------------------------------------------------------------------
# MCTS+NN vs Random — worker (runs in a subprocess or inline)
# ---------------------------------------------------------------------------

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
    _probe_engines: dict | None = None
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
            pe: _ProbeMCTS = _probe_engines[player]  # type: ignore[index]
            return pe.get_move(game_state)
        _probe_fn = _probe_fn_impl

    from game import Game
    from players.mcts import MCTSPlayer

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


# ---------------------------------------------------------------------------
# MCTS+NN vs Random — dispatcher (optionally multi-process)
# ---------------------------------------------------------------------------

def evaluate_vs_random(
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
    if backend != "cpp":
        node_dim = SAMPLE_ENC.data.x.size(1)  # type: ignore
        global_dim = SAMPLE_ENC.data.global_feats.size(1)
        load_model(model_path, node_dim, global_dim, device=device)

    from rl.utils import sha256_file
    model_sha256 = sha256_file(Path(model_path))

    t0 = time.time()
    eval_jobs = max(1, int(eval_jobs))
    chunks = split_indices(int(eval_games), eval_jobs)
    summaries: list[dict] = []
    torch_threads = 1 if int(eval_jobs) > 1 else 0

    base_payload = {
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
    }

    if eval_jobs <= 1:
        summaries.append(
            _eval_vs_random_worker({**base_payload, "gidxs": list(range(int(eval_games)))})
        )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=eval_jobs, mp_context=ctx) as ex:
            futs = [ex.submit(_eval_vs_random_worker, {**base_payload, "gidxs": chunk}) for chunk in chunks]
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
        first_agent_move_counts = merge_counts(first_agent_move_counts, dict(s.get("first_agent_move_counts", {})))
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
        "cpp_profile": cpp_profile_sum if backend == "cpp" else None,
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


# ---------------------------------------------------------------------------
# Checkpoint vs checkpoint — worker
# ---------------------------------------------------------------------------

def _eval_cpp_model_vs_model_worker(payload: dict) -> dict:
    torch_threads = int(payload.get("torch_threads") or 0)
    if torch_threads > 0:
        try:
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

    from game import Game

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

    _probe_fn_mm = None
    _probe_engines_mm: dict | None = None
    if filter_near_area:
        _probe_engines_mm = {
            0: MCTSPlayerCPP(0, c_puct=1.0, n_rollouts=filter_probe_rollouts, seed=0, verbose=False, use_nn_value=bool(cpp_use_nn_value)),
            1: MCTSPlayerCPP(1, c_puct=1.0, n_rollouts=filter_probe_rollouts, seed=0, verbose=False, use_nn_value=bool(cpp_use_nn_value)),
        }
        for _pe in _probe_engines_mm.values():
            _pe.set_model_checkpoint(model_a_path, device=device)
            _pe.set_exploration(dirichlet_alpha=0.0, dirichlet_epsilon=0.0, temperature=0.0, temperature_moves=0)

        def _probe_fn_mm_impl(game_state, player):
            pe: MCTSPlayerCPP = _probe_engines_mm[player]  # type: ignore[index]
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


# ---------------------------------------------------------------------------
# Checkpoint vs checkpoint — dispatcher
# ---------------------------------------------------------------------------

def evaluate_cpp_model_vs_model(
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
    chunks = split_indices(int(eval_games), eval_jobs)
    summaries: list[dict] = []
    torch_threads = 1 if int(eval_jobs) > 1 else 0

    base_payload = {
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
    }

    if eval_jobs <= 1:
        summaries.append(
            _eval_cpp_model_vs_model_worker({**base_payload, "gidxs": list(range(int(eval_games)))})
        )
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=eval_jobs, mp_context=ctx) as ex:
            futs = [ex.submit(_eval_cpp_model_vs_model_worker, {**base_payload, "gidxs": chunk}) for chunk in chunks]
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
