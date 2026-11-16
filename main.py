import argparse
import glob
import json
import os
import time

import torch

from game import Game, GameProtocol
from gnn.encode import SAMPLE_ENC
from gnn.model import load_model
from players import AlphaBetaPlayer, HumanPlayer, MCTSPlayer, Player
from players.base import Resigned
from players.alphabeta_cpp import AlphaBetaPlayerCPP
from players.game_total import GameTotal
from players.mcts_cpp import MCTSPlayerCPP
from rl.PPO import PPOGNNPolicy, PPOPlayer


def _next_save_path(save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    existing = [p for p in glob.glob(os.path.join(save_dir, "game_*.json"))]
    start_idx = 0
    if existing:
        def _idx(p: str) -> int:
            stem = os.path.basename(p)
            try:
                return int(stem.split("_")[1].split(".")[0])
            except Exception:
                return -1
        start_idx = max(map(_idx, existing)) + 1
    return os.path.join(save_dir, f"game_{start_idx:05d}.json")

def _save_game(game: GameProtocol, opponent: Player, save_dir: str = "saved_games_human", evals: list | None = None,
               human_player: int | None = None, bot_spec: str | None = None, resigned: int | None = None) -> str:
    path = _next_save_path(save_dir)
    payload: dict[str, object] = {
        "winner": game.winner,
        "moves": [{"x": m.c[0], "y": m.c[1], "t": m.t} for m in game.moves],
        "max_moves_reached": False,
        # one entry per move: the rs engine's search result for its own moves, None otherwise
        "evals": evals if evals is not None else [],
        "meta": {
            "mode": "human-vs-bot",
            "bot": opponent.__class__.__name__,
            "time_limit": getattr(opponent, "time_limit", None),
            "human_player": human_player,
            "bot_spec": bot_spec,
            # the player (0 or 1) who resigned, if the game ended that way
            "resigned": resigned,
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"Saved game to {path}")
    return path


def _rate_game(game: GameProtocol, name: str, human_num: int, bot_spec: str, game_path: str) -> None:
    """Record the result in ratings.jsonl and show the updated ratings of both sides."""
    import ratings

    p1, p2 = (name, bot_spec) if human_num == 0 else (bot_spec, name)
    rec = ratings.record_game(p1, p2, game.winner, plies=len(game.moves), source="human", note=game_path)
    fit = ratings.fit(ratings.load_games())
    for r in fit.ratings:
        if r.name in (rec["p1"], rec["p2"]):
            print(f"  rating  {r.name:14s} {r.elo:6.0f} ± {r.ci95:.0f}   ({r.wins}-{r.losses} in {r.games} rated games)")
    print("  (full table: python3 tools/rate.py)")

def _load_gnn(model_path: str, device: str) -> None:
    node_dim, global_dim = SAMPLE_ENC.data.x.size(1), SAMPLE_ENC.data.global_feats.size(1) # type: ignore
    load_model(model_path, node_dim, global_dim, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Rocks and Sticks")
    parser.add_argument("--ai", choices=["rs", "mcts", "alphabeta", "ppo", "mcts-cpp", "alphabeta-cpp", "none"], default="none",
                        help="opponent: rs = the strong alpha-beta engine (recommended); the others are the older bots")
    parser.add_argument("--bot", type=str, default=None,
                        help="play a bot by spec, e.g. rs:5s:4t, rs:100k, ab:3, mcts:3000 or random (see players/specs.py); "
                             "overrides --ai. Games against --bot and --ai rs opponents are rated (python3 tools/rate.py)")
    parser.add_argument("--name", type=str, default="you", help="your name in the ratings table (default: you)")
    parser.add_argument("--no-rate", action="store_true", help="do not record this game in the ratings")
    parser.add_argument("--rs-time", type=float, default=5.0, help="thinking time per move in seconds for --ai rs (default: 5)")
    parser.add_argument("--rs-threads", type=int, default=4, help="search threads for --ai rs (default: 4)")
    parser.add_argument("--no-ponder", action="store_true", help="--ai rs: do not let the engine think while you do")
    parser.add_argument("--show-eval", action="store_true", help="draw the engine's evaluation bar while you play (--ai rs)")
    parser.add_argument("--rs-weights", type=str, default=None, help="evaluation file for --ai rs (default: chosen by thinking time, see players/rs_player.py)")
    parser.add_argument("--mcts-time-limit", type=float, default=None, help="time limit (seconds) for MCTS simulations")
    parser.add_argument("--mcts-rollouts",type=int,default=None,help="number of rollouts for MCTS simulations")
    parser.add_argument("--model", type=str, default=None, help="path to GNN weights to enable NN eval or PPO model for --ai ppo")
    parser.add_argument("--policy", type=str, default=None, help="path to PolicyValueNet checkpoint for policy priors (mcts-cpp)")
    parser.add_argument("--device", type=str, default="cpu", help="device for GNN)")
    parser.add_argument("--ab-depth", type=int, default=3, help="search depth for alphabeta-cpp (default: 3)")
    parser.add_argument("--ab-time-limit", type=int, default=0, help="time limit (ms) for alphabeta-cpp iterative deepening (0=use fixed depth)")
    parser.add_argument("--native-model", type=str, default=None, help="path to native NN weights (.bin) for AB move ordering")
    parser.add_argument("--nn-ordering-depth", type=int, default=2, help="min AB depth for NN move ordering (default: 2)")
    parser.add_argument("--human-player", type=int, default=0, choices=[0, 1], help="which player the human controls (0 or 1)")
    args = parser.parse_args()

    cpp = args.ai in {"mcts-cpp", "alphabeta-cpp"}
    human_num = int(args.human_player)
    bot_num = 1 - human_num

    if args.ai == "ppo":
        if not args.model:
            raise ValueError("--model (path to PPO model) is required for --ai ppo")
        node_feat_dim = SAMPLE_ENC.data.x.size(1) # type: ignore
        global_feat_dim = SAMPLE_ENC.data.global_feats.size(1)
        model = PPOGNNPolicy(node_feat_dim=node_feat_dim, global_feat_dim=global_feat_dim)
        state = torch.load(args.model, map_location=args.device)
        model.load_state_dict(state)
        model.eval()
        opponent = PPOPlayer(bot_num, model, device=args.device)
        print(f"Loaded PPO agent from {args.model} on device {args.device}.")
    elif args.model:
        print(f"Using GNN evaluator from {args.model} on device {args.device}.")
        _load_gnn(args.model, args.device)

    # bot_spec names the opponent exactly (players/specs.py); games against it are rated
    bot_spec: str | None = None
    if args.ai == "rs" and args.bot is None:
        if args.rs_weights:
            from players.rs_player import RSPlayer

            opponent = RSPlayer(bot_num, time_ms=int(args.rs_time * 1000), weights=args.rs_weights, threads=args.rs_threads, tt_bits=23, verbose=True)
            print("Custom weights: this game will not be rated.")
        else:
            args.bot = f"rs:{int(round(args.rs_time * 1000))}ms:{args.rs_threads}t" + ("" if args.no_ponder else ":ponder")
    if args.bot is not None:
        from players.specs import canonical, describe, make_bot

        bot_spec = canonical(args.bot)
        opponent = make_bot(bot_spec, bot_num, seed=int.from_bytes(os.urandom(3), "little"), verbose=True)
        print(f"Opponent: {bot_spec} ({describe(bot_spec)})")
        cpp = True  # the old bots need the C++ game state alongside; harmless for the others
    elif args.ai == "rs":
        pass
    elif args.ai == "mcts":
        opponent = MCTSPlayer(bot_num, time_limit=args.mcts_time_limit, n_rollouts=args.mcts_rollouts, use_gnn=bool(args.model), check_forced_losses=not bool(args.model))
    elif args.ai == "alphabeta":
        opponent = AlphaBetaPlayer(bot_num, use_gnn=bool(args.model))
    elif args.ai == "mcts-cpp":
        rollouts = args.mcts_rollouts or 1200
        if args.model:
            opponent = MCTSPlayerCPP(bot_num, n_rollouts=rollouts, use_nn_value=True)
            opponent.set_model_checkpoint(args.model, device=args.device)
            if args.policy:
                opponent.set_policy_checkpoint(args.policy, device=args.device)
                try:
                    opponent.engine.set_prior_params(0.04, 1.0)
                except Exception:
                    pass
            opponent.set_exploration(dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
                                    temperature=0.0, temperature_moves=0)
            print(f"C++ MCTS: {rollouts} rollouts, model={args.model}, policy={args.policy}")
        else:
            # No model: use heuristic evaluation at leaf nodes (much better than random rollouts)
            opponent = MCTSPlayerCPP(bot_num, n_rollouts=rollouts,
                                      use_nn_value=False, use_heuristic_rollout=True)
            opponent.set_exploration(dirichlet_alpha=0.0, dirichlet_epsilon=0.0,
                                    temperature=0.0, temperature_moves=0)
            print(f"C++ MCTS: {rollouts} rollouts, heuristic eval (no GNN)")
    elif args.ai == "alphabeta-cpp":
        ab_depth = args.ab_depth
        ab_time = args.ab_time_limit
        native_model = getattr(args, 'native_model', None)
        nn_ord_depth = getattr(args, 'nn_ordering_depth', 2)
        if args.model:
            opponent = AlphaBetaPlayerCPP(bot_num, depth=ab_depth, time_limit_ms=ab_time, max_depth=20,
                                          native_model=native_model or "", nn_ordering_depth=nn_ord_depth)
            opponent.set_model_checkpoint(args.model, device=args.device)
            print(f"C++ AlphaBeta: depth={ab_depth}, time={ab_time}ms, GNN model={args.model}")
        else:
            opponent = AlphaBetaPlayerCPP(bot_num, depth=ab_depth, use_heuristic=True,
                                          time_limit_ms=ab_time, max_depth=20,
                                          native_model=native_model or "", nn_ordering_depth=nn_ord_depth)
            if native_model:
                print(f"C++ AlphaBeta: heuristic eval + native NN ordering (depth>={nn_ord_depth})")
            if ab_time > 0:
                print(f"C++ AlphaBeta: iterative deepening, {ab_time}ms/move, heuristic eval")
            else:
                print(f"C++ AlphaBeta: depth={ab_depth}, heuristic eval (no GNN)")
    else:
        opponent = HumanPlayer(bot_num)

    if bot_spec is None and args.ai not in ("none", "rs"):
        print("This opponent is not rated; use --bot (e.g. --bot ab:3 or --bot mcts:3000) for rated games.")

    player_list: list[Player | None] = [None, None]
    player_list[human_num] = HumanPlayer(human_num)
    player_list[bot_num] = opponent
    players: list[Player] = [p for p in player_list if p is not None]  # type: ignore
    game: GameProtocol
    if cpp:
        print("Using C++ GameState backend via pybind11.")
        import players_ext
        game = GameTotal(Game(players=players), players_ext.GameState()) # type: ignore
    else:
        game = Game(players=players)



    evals: list = []
    resigned: int | None = None
    while True:
        print(f"Turn {game.turn_number}")
        for player in players:
            game.render(block=False)
            t0 = time.perf_counter()
            try:
                m = player.get_move(game)
            except Resigned as r:
                # the other player wins; the game is saved and rated like any other loss
                resigned = r.player
                game.winner = 1 - r.player
                getattr(game, "py", game).result_note = f"Player {r.player + 1} resigned"
                print(f"Player {r.player + 1} resigns.")
                break
            dt = time.perf_counter() - t0
            info = getattr(player, "last_info", None) if player is opponent else None
            evals.append({"score": int(info["score"]), "depth": int(info["depth"]), "mover": player.number} if info else None)
            # Only print timing for non-human players (bots).
            if player.__class__.__name__ != "HumanPlayer":
                print(f"Player {player.number + 1} ({player.__class__.__name__}) move time: {dt:.3f}s")
            if player.__class__.__name__ == "AlphaBetaPlayerCPP":
                depth = int(getattr(player.engine, "last_depth_completed", -1)) # type: ignore
                nodes = int(getattr(player.engine, "last_nodes_searched", -1)) # type: ignore
                print(f"Player {player.number + 1} AlphaBeta stats: depth={depth}, nodes={nodes}")
            print(f"Player {player.number + 1} plays {m}")
            game.do_move(player.number, m)
            game.render(block=False)
            if info and args.show_eval:
                import matplotlib.pyplot as plt

                from players.rs_display import describe, draw_eval_bar

                draw_eval_bar(plt.gcf(), describe(info["score"], player.number))

        if game.winner is not None:
            if resigned is None:
                print(f"player {game.winner + 1} wins with area {game.players_scores[game.winner] / 2}")
            else:
                print(f"player {game.winner + 1} wins by resignation")
            # save (and rate) before showing the final board: closing the terminal instead of
            # the window must not lose the game
            stop = getattr(opponent, "stop_pondering", None)
            if stop is not None:
                stop()
            path = _save_game(game, opponent, save_dir="saved_games_human", evals=evals, human_player=human_num, bot_spec=bot_spec, resigned=resigned)
            if bot_spec is not None and not args.no_rate:
                _rate_game(game, args.name, human_num, bot_spec, path)
            game.render(block=True)
            break
