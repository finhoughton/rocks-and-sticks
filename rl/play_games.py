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
    device: str = "cpu",
    temp: float = 1.0,
    swap_roles: bool = True,
    max_moves: int = 256,
    seed_base: int | None = None,
    prune_size: int | None = None,
    backend: str = "python",
    cpp_verbose: bool = False,
) -> None:
    save_path_factory = _next_save_path(save_games_dir)
    if seed_base is None:
        seed_base = random.Random().randrange(1_000_000_000) ^ int(random.random() * 1e6)

    if model_path:
        node_dim = SAMPLE_ENC.data.x.size(1) # type: ignore
        global_dim = SAMPLE_ENC.data.global_feats.size(1)
        load_model(model_path, node_dim, global_dim, device=device)
        print(f"Loaded GNN eval from {model_path} on {device}")
    elif backend == "cpp":
        raise ValueError("--model is required when --backend=cpp (GNN evaluation cannot be disabled)")

    for i in range(num_games):
        print(f"Generating game {i+1}/{num_games} (seed={seed_base + i})...")
        if backend == "cpp":
            import players_ext

            from players.game_total import GameTotal

            game = GameTotal(Game(), players_ext.GameState())
        else:
            game = Game()
        moves_log: list[Move] = []
        movers_log: list[int] = []
        randomize_start(cast(Any, game), move_log=moves_log, mover_log=movers_log)
        if backend == "cpp":
            from players.mcts_cpp import MCTSPlayerCPP
            mcts_players = {
                0: MCTSPlayerCPP(
                    0,
                    n_rollouts=mcts_rollouts if mcts_rollouts is not None else 1000,
                    seed=seed_base + i,
                    verbose=cpp_verbose,
                ),
                1: MCTSPlayerCPP(
                    1,
                    n_rollouts=mcts_rollouts if mcts_rollouts is not None else 1000,
                    seed=seed_base + i + 1,
                    verbose=cpp_verbose,
                ),
            }

            for p in mcts_players.values():
                p.set_exploration(
                    dirichlet_alpha=0.3,
                    dirichlet_epsilon=0.25,
                    temperature=float(temp),
                    temperature_moves=20,
                )
        else:
            mcts_players = {
                0: MCTSPlayer(0, check_forced_losses=False, use_gnn=bool(model_path), n_rollouts=mcts_rollouts if mcts_rollouts is not None else 1000, time_limit=mcts_time_limit, seed=seed_base + i),
                1: MCTSPlayer(1, check_forced_losses=False, use_gnn=bool(model_path), n_rollouts=mcts_rollouts if mcts_rollouts is not None else 1000, time_limit=mcts_time_limit, seed=seed_base + i + 1),
            }
        policy_targets: list[list[dict]] = []

        while game.winner is None and len(game.moves) < max_moves:
            player_idx = game.current_player
            mcts: Any = mcts_players[player_idx]
            key = _game_key(game)
            move = mcts.get_move(game, reuse_tree=True)
            key_after = _game_key(game)
            if key != key_after:
                raise Exception(f"keys not equal OLD: {key} \n\n NEW: {key_after}")
            if backend == "cpp":
                policy = _visits_to_policy_cpp(mcts.get_root_visit_stats(game))
            else:
                policy = _visits_to_policy(mcts, mcts._root_key)
            policy_targets.append(policy)
            moves_log.append(move)
            movers_log.append(player_idx)

            game.do_move(game.current_player, move)
            for p_mcts in mcts_players.values():
                p_mcts.advance_root(move, game)

            if prune_size is not None and prune_size > 0:
                for p_mcts in mcts_players.values():
                    p_mcts.prune_tables(prune_size)

        out_path = save_path_factory(i)
        payload = {
            "winner": game.winner,
            "moves": [
                {"x": move.c[0], "y": move.c[1], "t": move.t, "p": int(movers_log[j])}
                for j, move in enumerate(moves_log)
            ],
            "max_moves_reached": (len(game.moves) >= max_moves and game.winner is None),
            "policy_targets": policy_targets,
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
    parser.add_argument("--max-moves", type=int, default=256)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--prune-size", type=int, default=None, help="optional max MCTS state table size to prune to (per-player)")
    parser.add_argument("--backend", type=str, default="python", choices=["python", "cpp"], help="MCTS backend: pure Python or C++ (players_ext)")
    args = parser.parse_args()
    play_self_play_games(
        num_games=args.games,
        mcts_rollouts=args.mcts_rollouts,
        mcts_time_limit=args.mcts_time_limit,
        save_games_dir=args.save_games_dir,
        model_path=args.model,
        device=args.device,
        temp=args.temp,
        max_moves=args.max_moves,
        seed_base=args.seed_base,
        prune_size=args.prune_size,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()
