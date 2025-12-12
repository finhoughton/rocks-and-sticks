from __future__ import annotations

import argparse
import cProfile
import pstats
from dataclasses import dataclass
from typing import Dict

from game import Game
from players import AlphaBetaPlayer, MCTSPlayer, RandomPlayer


@dataclass(frozen=True)
class EvalConfig:
    games: int = 50
    max_turns: int = 200
    mcts_rollouts: int = 400
    mcts_sim_depth: int = 60
    mcts0_seed: int | None = 0
    mcts1_seed: int | None = 1
    random_seed: int | None = 1
    show: bool = False
    pause_s: float = 0.0
    mode: str = "mcts-vs-random"  # or "mcts-vs-mcts" / "mcts-vs-alphabeta" / "alphabeta-vs-mcts"
    profile: bool = False
    profile_top: int = 40
    profile_out: str | None = None


def play_one_game(cfg: EvalConfig, game_index: int) -> int | None:
    # Keep seeds deterministic but varied per game.
    if cfg.mode == "mcts-vs-mcts":
        p0 = MCTSPlayer(
            0,
            seed=None if cfg.mcts0_seed is None else cfg.mcts0_seed + game_index,
            n_rollouts=cfg.mcts_rollouts,
            max_sim_depth=cfg.mcts_sim_depth,
        )
        p1 = MCTSPlayer(
            1,
            seed=None if cfg.mcts1_seed is None else cfg.mcts1_seed + game_index,
            n_rollouts=cfg.mcts_rollouts,
            max_sim_depth=cfg.mcts_sim_depth,
        )
    elif cfg.mode == "mcts-vs-alphabeta":
        p0 = AlphaBetaPlayer(0)
        p1 = MCTSPlayer(
            1,
            seed=None if cfg.mcts1_seed is None else cfg.mcts1_seed + game_index,
            n_rollouts=cfg.mcts_rollouts,
            max_sim_depth=cfg.mcts_sim_depth,
        )
    elif cfg.mode == "alphabeta-vs-mcts":
        p0 = MCTSPlayer(
            0,
            seed=None if cfg.mcts0_seed is None else cfg.mcts0_seed + game_index,
            n_rollouts=cfg.mcts_rollouts,
            max_sim_depth=cfg.mcts_sim_depth,
        )
        p1 = AlphaBetaPlayer(1)
    else:
        p0 = RandomPlayer(
            0, seed=None if cfg.random_seed is None else cfg.random_seed + game_index
        )
        p1 = MCTSPlayer(
            1,
            seed=None if cfg.mcts1_seed is None else cfg.mcts1_seed + game_index,
            n_rollouts=cfg.mcts_rollouts,
            max_sim_depth=cfg.mcts_sim_depth,
        )

    g = Game(players=[p0, p1])

    plt = None
    if cfg.show:
        import matplotlib.pyplot as plt  # type: ignore

        plt.ion()  # type: ignore[reportUnknownMemberType]
        g.render(block=False)

    while g.winner is None and g.turn_number < cfg.max_turns:
        for player in g.players:
            move = player.get_move(g)
            g.do_move(player, move)
            if cfg.show:
                g.render(block=False)
                if cfg.pause_s > 0 and plt is not None:
                    plt.pause(cfg.pause_s)
            if g.winner is not None:
                break

    # Close the figure so multiple games don't spawn a pile of windows.
    if cfg.show and plt is not None:
        try:
            fig = getattr(g, "_render_fig", None)
            if fig is not None:
                plt.close(fig)  # type: ignore[reportPrivateUsage]
            else:
                plt.close("all")
        except Exception:
            pass

    return g.winner


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MCTS vs Random")
    parser.add_argument(
        "--mode",
        choices=[
            "mcts-vs-random",
            "mcts-vs-mcts",
            "mcts-vs-alphabeta",
            "alphabeta-vs-mcts",
        ],
        default="mcts-vs-random",
        help="Which matchup to run",
    )
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--rollouts", type=int, default=400)
    parser.add_argument("--sim-depth", type=int, default=60)
    parser.add_argument("--mcts0-seed", type=int, default=0)
    parser.add_argument("--mcts1-seed", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--show", action="store_true", help="Render games as they play")
    parser.add_argument(
        "--pause", type=float, default=0.0, help="Extra pause between moves (seconds)"
    )
    parser.add_argument(
        "--profile", action="store_true", help="Run under cProfile and print hotspots"
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=40,
        help="How many lines of profile output to print",
    )
    parser.add_argument(
        "--profile-out",
        type=str,
        default=None,
        help="Optional path to write a .prof file",
    )
    args = parser.parse_args()

    cfg = EvalConfig(
        mode=args.mode,
        games=args.games,
        max_turns=args.max_turns,
        mcts_rollouts=args.rollouts,
        mcts_sim_depth=args.sim_depth,
        mcts0_seed=None if args.mcts0_seed < 0 else args.mcts0_seed,
        mcts1_seed=None if args.mcts1_seed < 0 else args.mcts1_seed,
        random_seed=None if args.random_seed < 0 else args.random_seed,
        show=args.show,
        pause_s=args.pause,
        profile=args.profile,
        profile_top=args.profile_top,
        profile_out=args.profile_out,
    )
    wins: Dict[int | None, int] = {0: 0, 1: 0, None: 0}

    def run_eval() -> None:
        for i in range(cfg.games):
            w = play_one_game(cfg, i)
            wins[w] += 1

    if cfg.profile:
        pr = cProfile.Profile()
        pr.enable()
        try:
            run_eval()
        finally:
            pr.disable()

        if cfg.profile_out:
            pr.dump_stats(cfg.profile_out)
            print(f"Wrote profile to: {cfg.profile_out}")

        stats = pstats.Stats(pr)
        stats.strip_dirs().sort_stats(pstats.SortKey.CUMULATIVE)
        print("\n=== cProfile (top cumulative) ===")
        stats.print_stats(max(1, cfg.profile_top))
    else:
        run_eval()

    total = cfg.games
    print(f"Games: {total}")
    if cfg.mode == "mcts-vs-mcts":
        print(f"MCTS P1 (player 1) wins: {wins[0]} ({wins[0] / total:.1%})")
        print(f"MCTS P2 (player 2) wins: {wins[1]} ({wins[1] / total:.1%})")
    elif cfg.mode == "mcts-vs-alphabeta":
        print(f"AlphaBeta (player 1) wins: {wins[0]} ({wins[0] / total:.1%})")
        print(f"MCTS (player 2) wins: {wins[1]} ({wins[1] / total:.1%})")
    elif cfg.mode == "alphabeta-vs-mcts":
        print(f"MCTS (player 1) wins: {wins[0]} ({wins[0] / total:.1%})")
        print(f"AlphaBeta (player 2) wins: {wins[1]} ({wins[1] / total:.1%})")
    else:
        print(f"Random (player 1) wins: {wins[0]} ({wins[0] / total:.1%})")
        print(f"MCTS (player 2) wins: {wins[1]} ({wins[1] / total:.1%})")
    print(f"No winner (max_turns reached): {wins[None]} ({wins[None] / total:.1%})")


if __name__ == "__main__":
    main()
