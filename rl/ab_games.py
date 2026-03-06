from __future__ import annotations

import json
import random
from pathlib import Path

from gnn.game_generation import randomize_start
from players.base import Player


def generate_cpp_ab_games(
    *,
    num_games: int,
    save_games_dir: Path,
    seed_base: int,
    ab_time_limit_ms: int,
    ab_depth: int,
    ab_max_depth: int,
    ab_move_cap: int,
    ab_max_moves: int,
    ab_use_heuristic: bool,
    native_model: str,
    nn_ordering_depth: int,
    randomize_start_enabled: bool,
    randomize_max_sticks: int,
) -> None:
    import players_ext

    from game import Game
    from players.alphabeta_cpp import AlphaBetaPlayerCPP
    from players.game_total import GameTotal

    save_games_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(save_games_dir.glob("game_*.json"))
    start_idx = 0
    if existing:
        def _idx(path_obj: Path) -> int:
            try:
                return int(path_obj.stem.split("_")[1])
            except Exception:
                return -1

        start_idx = max(_idx(p) for p in existing) + 1

    for g in range(int(num_games)):
        base_seed = int(seed_base) + int(g) * 17
        p0 = AlphaBetaPlayerCPP(
            0,
            seed=base_seed,
            depth=int(ab_depth),
            move_cap=int(ab_move_cap),
            time_limit_ms=int(ab_time_limit_ms),
            max_depth=int(ab_max_depth),
            use_heuristic=bool(ab_use_heuristic),
            native_model=str(native_model),
            nn_ordering_depth=int(nn_ordering_depth),
        )
        p1 = AlphaBetaPlayerCPP(
            1,
            seed=base_seed + 1,
            depth=int(ab_depth),
            move_cap=int(ab_move_cap),
            time_limit_ms=int(ab_time_limit_ms),
            max_depth=int(ab_max_depth),
            use_heuristic=bool(ab_use_heuristic),
            native_model=str(native_model),
            nn_ordering_depth=int(nn_ordering_depth),
        )
    
        players: list[Player] = [p0, p1] if (g % 2 == 0) else [p1, p0]
        game = GameTotal(Game(players=players), players_ext.GameState())
        move_log = []
        mover_log: list[int] = []

        if bool(randomize_start_enabled):
            random.seed(base_seed)
            randomize_start(
                game,
                max_sticks=int(randomize_max_sticks),
                move_log=move_log,
                mover_log=mover_log,
            )

        move_count = 0
        while game.winner is None and move_count < int(ab_max_moves):
            cp = game.current_player
            mv = game.players[cp].get_move(game)
            move_log.append(mv)
            mover_log.append(int(cp))
            game.do_move(cp, mv)
            move_count += 1

        payload: dict[str, object] = {
            "winner": game.winner,
            "moves": [
                {"x": m.c[0], "y": m.c[1], "t": m.t, "p": int(pnum)}
                for m, pnum in zip(move_log, mover_log)
            ],
            "max_moves_reached": bool(game.winner is None and move_count >= int(ab_max_moves)),
            "meta": {
                "source": "cpp-ab-self-play",
                "ab_time_limit_ms": int(ab_time_limit_ms),
                "ab_depth": int(ab_depth),
                "nn_ordering_depth": int(nn_ordering_depth),
            },
        }

        out_path = save_games_dir / f"game_{start_idx + g:05d}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)

        print(
            f"[{g + 1}/{num_games}] winner={game.winner} moves={len(game.moves)} "
            f"saved={out_path}"
        )
