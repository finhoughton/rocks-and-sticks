import argparse
import glob
import json
import os

from game import Game
from players import AlphaBetaPlayer, HumanPlayer, MCTSPlayer, Player


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

def _save_game(game: Game, opponent: Player, save_dir: str = "saved_games_human") -> None:
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

def _load_gnn(game: Game, model_path: str, device: str) -> None:
    from gnn.gnn_encode import encode_game_to_graph
    from gnn.gnn_model import load_model

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
    _save_game(game, opponent, save_dir="saved_games_human")
