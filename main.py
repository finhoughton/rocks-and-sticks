import argparse

from game import Game
from players import AlphaBetaPlayer, HumanPlayer, MCTSPlayer, Player


def _load_gnn(game: Game, model_path: str, device: str) -> None:
    from gnn_eval import encode_game_to_graph, load_model

    enc = encode_game_to_graph(game)

    node_dim, global_dim = enc.data.x.size(1), enc.data.global_feats.size(1) # type: ignore
    load_model(model_path, node_dim, global_dim, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Rocks and Sticks")
    parser.add_argument("--ai", choices=["mcts", "alphabeta"], default="mcts")
    parser.add_argument("--time-limit", type=float, default=2.0, help="MCTS time limit (seconds)")
    parser.add_argument("--gnn-model", type=str, default=None, help="path to GNN weights to enable NN eval")
    parser.add_argument("--device", type=str, default="cpu", help="device for GNN (cpu/cuda)")
    args = parser.parse_args()

    opponent = MCTSPlayer(1, time_limit=args.time_limit, use_gnn=bool(args.gnn_model)) if args.ai == "mcts" else AlphaBetaPlayer(1, use_gnn=bool(args.gnn_model))
    players: list[Player] = [HumanPlayer(0), opponent]
    game = Game(players=players)

    if args.gnn_model:
        print(f"Using GNN evaluator from {args.gnn_model} on device {args.device}.")
        _load_gnn(game, args.gnn_model, args.device)

    game.run(display=True)
