from __future__ import annotations

import argparse
import json
import os
import random
from glob import glob
from typing import Any, Callable, List, cast

from game import Game
from gnn.encode import SAMPLE_ENC
from gnn.game_generation import randomize_start
from gnn.model import load_model
from models import Move, move_key
from players import MCTSPlayer, StateKey, _game_key


def _next_save_path(save_dir: str) -> Callable[[int], str]:
    os.makedirs(save_dir, exist_ok=True)
    existing = [p for p in glob(os.path.join(save_dir, "game_*.json"))]
    start_idx = 0
    if existing:
        def _idx(p: str) -> int:
            stem = os.path.basename(p)
            try:
                return int(stem.split("_")[1].split(".")[0])
            except Exception:
                return -1
        start_idx = max(map(_idx, existing)) + 1
    return lambda i: os.path.join(save_dir, f"game_{start_idx + i:05d}.json")


def _visits_to_policy(mcts: MCTSPlayer, root_key: StateKey) -> List[dict]:
    """Return normalized visit distribution for legal root moves.

    Each entry is a dict: {"x":int,"y":int,"t":str,"visits":int,"prob":float}
    """
    legal = mcts._legal_moves.get(root_key, [])
    out: list[dict] = []
    total = 0
    for m in legal:
        v = mcts.Nsa.get((root_key, move_key(m)), 0)
        total += v
        out.append({"x": m.c[0], "y": m.c[1], "t": m.t, "visits": int(v)})
    if total <= 0:
        # fallback: give uniform tiny mass
        n = max(1, len(out))
        for e in out:
            e["prob"] = 1.0 / n
    else:
        for e in out:
            e["prob"] = float(e["visits"]) / float(max(1, total))
    return out


def _visits_to_policy_cpp(raw: list[dict]) -> list[dict]:
    """Normalize visit distribution returned by the C++ backend.

    Input entries are dicts: {"x":int,"y":int,"t":str,"visits":int}
    Output adds "prob".
    """
    total = 0
    for e in raw:
        total += int(e.get("visits", 0))
    if total <= 0:
        n = max(1, len(raw))
        for e in raw:
            e["prob"] = 1.0 / n
    else:
        for e in raw:
            e["prob"] = float(int(e.get("visits", 0))) / float(total)
    return raw


def play_self_play_games(
    num_games: int,
    mcts_rollouts: int | None,
    mcts_time_limit: float | None,
    save_games_dir: str,
    model_path: str | None,
    policy_path: str | None = None,
    device: str = "cpu",
    prior_scale: float = 1.0,
    prior_mix_uniform: float = 0.04,
    temp: float = 1.0,
    temperature_moves: int = 20,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
    vs_random_prob: float = 0.0,
    swap_roles: bool = True,
    max_moves: int = 256,
    seed_base: int | None = None,
    prune_size: int | None = None,
    backend: str = "python",
    cpp_verbose: int = 0,
    cpp_use_nn_value: bool = True,
    filter_near_area: bool = False,
    filter_probe_rollouts: int = 200,
    filter_probe_moves: int = 6,
    randomize_max_sticks: int = 5,
) -> None:
    save_path_factory = _next_save_path(save_games_dir)
    if seed_base is None:
        seed_base = random.Random().randrange(1_000_000_000) ^ int(random.random() * 1e6)

    # For the Python backend, the NN lives in-process via load_model().
    # For the C++ backend, the NN must be loaded per-engine via set_model_checkpoint().
    if backend == "cpp":
        if not model_path:
            raise ValueError("--model is required when --backend=cpp (GNN evaluation cannot be disabled)")
    else:
        if model_path:
            node_dim = SAMPLE_ENC.data.x.size(1) # type: ignore
            global_dim = SAMPLE_ENC.data.global_feats.size(1)
            load_model(model_path, node_dim, global_dim, device=device)
            print(f"Loaded GNN eval from {model_path} on {device}")

    vs_random_prob = float(max(0.0, min(1.0, float(vs_random_prob))))

    def _random_legal_move_for_player(game: Any, player_number: int, rng: random.Random) -> Move:
        from models import PASS

        moves = sorted((m for m in game.get_possible_moves(player_number) if m is not PASS), key=move_key)
        if not moves:
            return PASS
        return rng.choice(moves)

    # For the C++ backend, reuse engines across games to avoid re-loading the model.
    # We reseed + reset per game to keep the run deterministic.
    mcts_players: dict[int, Any] | None = None
    if backend == "cpp":
        from players.mcts_cpp import MCTSPlayerCPP

        mcts_players = {
            0: MCTSPlayerCPP(
                0,
                n_rollouts=mcts_rollouts if mcts_rollouts is not None else 1000,
                seed=0,
                verbose=cpp_verbose,
                use_nn_value=cpp_use_nn_value,
            ),
            1: MCTSPlayerCPP(
                1,
                n_rollouts=mcts_rollouts if mcts_rollouts is not None else 1000,
                seed=0,
                verbose=cpp_verbose,
                use_nn_value=cpp_use_nn_value,
            ),
        }

        if not model_path:
            raise ValueError("--model is required when --backend=cpp")
        for p in mcts_players.values():
            p.set_model_checkpoint(str(model_path), device=str(device))
            if policy_path:
                try:
                    p.set_policy_checkpoint(str(policy_path), device=str(device))
                except Exception as e:
                    print(f"Warning: failed to load policy checkpoint {policy_path}: {e}")
            p.set_exploration(
                dirichlet_alpha=float(dirichlet_alpha),
                dirichlet_epsilon=float(dirichlet_epsilon),
                temperature=float(temp),
                temperature_moves=int(temperature_moves),
            )
            try:
                # Apply policy prior scaling/mixing parameters to C++ engine
                p.engine.set_prior_params(float(prior_mix_uniform), float(prior_scale))
            except Exception:
                pass

    # Build a probe function for filtering randomized starts (MCTS-based quick-win detection).
    _probe_fn = None
    _probe_engines: dict[int, object] | None = None
    if filter_near_area and backend == "cpp" and model_path:
        from players.mcts_cpp import MCTSPlayerCPP as _ProbeMCTS

        PROBE_ROLLOUTS = int(filter_probe_rollouts)
        _probe_engines = {
            0: _ProbeMCTS(0, c_puct=1.0, n_rollouts=PROBE_ROLLOUTS, seed=0, verbose=False, use_nn_value=cpp_use_nn_value),
            1: _ProbeMCTS(1, c_puct=1.0, n_rollouts=PROBE_ROLLOUTS, seed=0, verbose=False, use_nn_value=cpp_use_nn_value),
        }
        for pe in _probe_engines.values():
            pe.set_model_checkpoint(str(model_path), device=str(device))
            pe.set_exploration(dirichlet_alpha=0.0, dirichlet_epsilon=0.0, temperature=0.0, temperature_moves=0)

        def _probe_fn_impl(game_state, player):
            pe = _probe_engines[player]
            return pe.get_move(game_state)
        _probe_fn = _probe_fn_impl

    for i in range(num_games):
        print(f"Generating game {i+1}/{num_games} (seed={seed_base + i})...")
        # Ensure deterministic starting position / move ordering (randomize_start uses the global RNG).
        random.seed(int(seed_base) + int(i))
        if backend == "cpp":
            import players_ext

            from players.game_total import GameTotal

            game = GameTotal(Game(), players_ext.GameState())
        else:
            game = Game()
        moves_log: list[Move] = []
        movers_log: list[int] = []
        # Reset probe engines per game so their state doesn't leak between positions.
        if _probe_engines is not None:
            for pe in _probe_engines.values():
                pe.reset_search()
        randomize_start(
            cast(Any, game),
            move_log=moves_log,
            mover_log=movers_log,
            filter_near_area=bool(filter_near_area),
            probe_fn=_probe_fn,
            probe_max_moves=int(filter_probe_moves),
            max_sticks=int(randomize_max_sticks),
        )

        # Store per-move policy targets aligned with `moves_log`.
        # We may not have a policy target for every move (e.g. random-opponent moves).
        policy_targets: list[list[dict] | None] = [None for _ in range(len(moves_log))]

        # Optional: mix in games where one side plays random.
        # We still record policy targets for MCTS moves, but set None for random moves.
        game_rng = random.Random(int(seed_base) + int(i) + 99991)
        use_random_opp = (vs_random_prob > 0.0 and game_rng.random() < vs_random_prob)
        random_side = 1
        if use_random_opp:
            # Alternate which side is random for balance (if desired).
            random_side = (i % 2) if bool(swap_roles) else 1
        if backend != "cpp":
            mcts_players = {
                0: MCTSPlayer(0, check_forced_losses=False, use_gnn=bool(model_path), n_rollouts=mcts_rollouts if mcts_rollouts is not None else 1000, time_limit=mcts_time_limit, seed=seed_base + i),
                1: MCTSPlayer(1, check_forced_losses=False, use_gnn=bool(model_path), n_rollouts=mcts_rollouts if mcts_rollouts is not None else 1000, time_limit=mcts_time_limit, seed=seed_base + i + 1),
            }

        # Reset and reseed C++ engines per game to keep behavior deterministic.
        if backend == "cpp":
            assert mcts_players is not None
            for pid, p_mcts in mcts_players.items():
                # Make tie-breaks / sampling deterministic per game.
                reseed = getattr(p_mcts, "set_seed", None)
                if reseed is not None:
                    reseed(int(seed_base) + int(i) + int(pid))
                p_mcts.reset_search()
                try:
                    p_mcts.engine.clear_root_priors()
                except Exception:
                    pass
        while game.winner is None and len(game.moves) < max_moves:
            player_idx = game.current_player
            assert mcts_players is not None
            mcts: Any = mcts_players[player_idx]
            key = _game_key(game)
            if use_random_opp and player_idx == int(random_side):
                move = _random_legal_move_for_player(game, int(player_idx), game_rng)
                policy_targets.append(None)
            else:
                move = mcts.get_move(game, reuse_tree=True)
                if backend == "cpp":
                    policy = _visits_to_policy_cpp(mcts.get_root_visit_stats(game))
                else:
                    policy = _visits_to_policy(mcts, mcts._root_key)
                policy_targets.append(policy)
            key_after = _game_key(game)
            if key != key_after:
                raise Exception(f"keys not equal OLD: {key} \n\n NEW: {key_after}")
            moves_log.append(move)
            movers_log.append(player_idx)

            game.do_move(game.current_player, move)
            # Fixed: advance_root now properly clears root_priors for the new position.
            for p_mcts in mcts_players.values():
                adv = getattr(p_mcts, "advance_root", None)
                if adv is not None:
                    adv(move, game)

            if prune_size is not None and prune_size > 0:
                for p_mcts in mcts_players.values():
                    prune = getattr(p_mcts, "prune_tables", None)
                    if prune is not None:
                        prune(prune_size)

        out_path = save_path_factory(i)
        payload = {
            "winner": game.winner,
            "moves": [
                {"x": move.c[0], "y": move.c[1], "t": move.t, "p": int(movers_log[j])}
                for j, move in enumerate(moves_log)
            ],
            "max_moves_reached": (len(game.moves) >= max_moves and game.winner is None),
            "policy_targets": policy_targets,
            "vs_random": bool(use_random_opp),
            "random_side": int(random_side) if bool(use_random_opp) else None,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        print(f"Saved self-play with policies to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MCTS self-play generator (AlphaZero-style targets)")
    parser.add_argument("--games", type=int, default=10)
    mcts_group = parser.add_mutually_exclusive_group()
    mcts_group.add_argument("--mcts-time-limit", type=float, default=None)
    mcts_group.add_argument("--mcts-rollouts", type=int, default=None)
    parser.add_argument("--save-games-dir", type=str, default="saved_games_mcts_alpha")
    parser.add_argument("--model", type=str, default=None, help="Path to GNN eval to enable NN priors/values")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for sampling moves from visit counts")
    parser.add_argument("--temp-moves", type=int, default=20, help="Number of opening moves to apply temperature sampling")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3, help="Dirichlet alpha for root noise (C++ backend only)")
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25, help="Dirichlet epsilon for root noise (C++ backend only)")
    parser.add_argument("--vs-random-prob", type=float, default=0.0, help="Probability a self-play game uses a random opponent (targets only recorded for MCTS moves)")
    parser.add_argument("--max-moves", type=int, default=256)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--prune-size", type=int, default=None, help="optional max MCTS state table size to prune to (per-player)")
    parser.add_argument("--backend", type=str, default="python", choices=["python", "cpp"], help="MCTS backend: pure Python or C++ (players_ext)")
    parser.add_argument("--filter-near-area", action="store_true", default=False, help="Reject randomized starts where a quick MCTS probe finds a win (reduces near-forced positions)")
    parser.add_argument("--filter-probe-rollouts", type=int, default=200, help="MCTS rollouts for the quick-win probe")
    parser.add_argument("--filter-probe-moves", type=int, default=6, help="Max moves for the quick-win probe")
    parser.add_argument("--randomize-max-sticks", type=int, default=5, help="Max random sticks in randomize_start")
    args = parser.parse_args()
    play_self_play_games(
        num_games=args.games,
        mcts_rollouts=args.mcts_rollouts,
        mcts_time_limit=args.mcts_time_limit,
        save_games_dir=args.save_games_dir,
        model_path=args.model,
        device=args.device,
        temp=args.temp,
        temperature_moves=int(args.temp_moves),
        dirichlet_alpha=float(args.dirichlet_alpha),
        dirichlet_epsilon=float(args.dirichlet_epsilon),
        vs_random_prob=float(args.vs_random_prob),
        max_moves=args.max_moves,
        seed_base=args.seed_base,
        prune_size=args.prune_size,
        backend=args.backend,
        filter_near_area=bool(args.filter_near_area),
        filter_probe_rollouts=int(args.filter_probe_rollouts),
        filter_probe_moves=int(args.filter_probe_moves),
        randomize_max_sticks=int(args.randomize_max_sticks),
    )


if __name__ == "__main__":
    main()
