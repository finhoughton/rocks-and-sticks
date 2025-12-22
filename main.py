import argparse

from game import Game
from players import AIPlayer, AlphaBetaPlayer, HumanPlayer, MCTSPlayer, Player


def _maybe_load_gnn(game: Game, model_path: str, device: str) -> None:
    try:
        from gnn_eval import encode_game_to_graph, load_model
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"GNN modules not available: {exc}")

    node_dim, global_dim = _infer_dims(game, encode_game_to_graph)
    load_model(model_path, node_dim, global_dim, device=device)


def _infer_dims(game: Game, encoder) -> tuple[int, int]: # type: ignore
    enc = encoder(game) # type: ignore
    return enc.data.x.size(1), enc.data.global_feats.size(1) # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Rocks and Sticks")
    parser.add_argument("--ai", choices=["mcts", "alphabeta"], default="mcts")
    parser.add_argument("--time-limit", type=float, default=2.0, help="MCTS time limit (seconds)")
    parser.add_argument("--gnn-model", type=str, default=None, help="path to GNN weights to enable NN eval")
    parser.add_argument("--device", type=str, default="cpu", help="device for GNN (cpu/cuda)")
    args = parser.parse_args()

    opponent = MCTSPlayer(1, time_limit=args.time_limit) if args.ai == "mcts" else AlphaBetaPlayer(1)
    players: list[Player] = [HumanPlayer(0), opponent]
    game = Game(players=players)

    if args.gnn_model:
        AIPlayer.use_gnn_eval = True
        AIPlayer.require_gnn_eval = True
        _maybe_load_gnn(game, args.gnn_model, args.device)
        print(f"Using GNN evaluator from {args.gnn_model} on device {args.device}.")

    game.run(display=True)
