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

def _save_game(game: GameProtocol, opponent: Player, save_dir: str = "saved_games_human") -> None:
    path = _next_save_path(save_dir)
    payload: dict[str, object] = {
        "winner": game.winner,
        "moves": [{"x": m.c[0], "y": m.c[1], "t": m.t} for m in game.moves],
        "max_moves_reached": False,
        "meta": {
            "mode": "human-vs-bot",
            "bot": opponent.__class__.__name__,
            "time_limit": getattr(opponent, "time_limit", None),
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    print(f"Saved game to {path}")

def _load_gnn(model_path: str, device: str) -> None:
    node_dim, global_dim = SAMPLE_ENC.data.x.size(1), SAMPLE_ENC.data.global_feats.size(1) # type: ignore
    load_model(model_path, node_dim, global_dim, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Rocks and Sticks")
    parser.add_argument("--ai", choices=["mcts", "alphabeta", "ppo", "mcts-cpp", "alphabeta-cpp", "none"], default="none")
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

    if args.ai == "mcts":
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



    while True:
        print(f"Turn {game.turn_number}")
        for player in players:
            game.render(block=False)
            t0 = time.perf_counter()
            m = player.get_move(game)
            dt = time.perf_counter() - t0
            # Only print timing for non-human players (bots).
            if player.__class__.__name__ != "HumanPlayer":
                print(f"Player {player.number + 1} ({player.__class__.__name__}) move time: {dt:.3f}s")
            print(f"Player {player.number + 1} plays {m}")
            game.do_move(player.number, m)
            game.render(block=False)

        if game.winner is not None:
            print(f"player {game.winner + 1} wins with area {game.players_scores[game.winner] / 2}")
            game.render(block=True)
            break

    _save_game(game, opponent, save_dir="saved_games_human")
