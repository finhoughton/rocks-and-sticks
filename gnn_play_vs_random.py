from __future__ import annotations

import argparse
import sys

from game import Game
from gnn_eval import encode_game_to_graph, evaluate_game, load_model
from players import HumanPlayer, RandomPlayer


def _infer_dims(game: Game) -> tuple[int, int]:
    enc = encode_game_to_graph(game)
    return enc.data.x.size(1), enc.data.global_feats.size(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play vs random and print GNN win probs")
    parser.add_argument("--model", type=str, default="gnn_eval.pt", help="path to GNN weights")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="disable matplotlib UI and read moves from stdin",
    )
    parser.add_argument("--rand-seed", type=int, default=1, help="seed for random opponent")
    args = parser.parse_args()

    players = [HumanPlayer(0), RandomPlayer(1, seed=1)]
    players[1] = RandomPlayer(1, seed=args.rand_seed)
    game = Game(players)

    if args.no_gui:
        def _stdin_input(prompt: str) -> str:
            return input(prompt)
        # Replace the GUI input loop with plain stdin.
        game.wait_for_move_input = _stdin_input  # type: ignore[attr-defined]
        print("CLI mode: enter moves as 'x y DIR' or 'P' to pass.")

    try:
        node_dim, global_dim = _infer_dims(game)
        load_model(args.model, node_dim, global_dim, device=args.device)
    except Exception as e:  # pragma: no cover
        print(f"Failed to load model: {e}")
        sys.exit(1)

    def print_prob() -> None:
        try:
            prob_cur = float(evaluate_game(game))
        except Exception as e:  # pragma: no cover
            print(f"Eval failed: {e}")
            return
        prob_human = prob_cur if game.current_player == 0 else (1.0 - prob_cur)
        print(
            f"Turn {game.turn_number}, to-move player {game.current_player + 1}: "
            f"P(human wins)={prob_human:.3f} | P(current wins)={prob_cur:.3f}"
        )

    print("Starting game: you are Player 1, opponent is random Player 2.")
    print_prob()

    while game.winner is None:
        p = game.players[game.current_player]
        mv = p.get_move(game)
        if isinstance(p, RandomPlayer):
            print(f"Random move: {mv}")
        game.do_move(p, mv)
        print_prob()

    if game.winner is None:
        print("Game ended in a draw.")
    elif game.winner == 0:
        print("You win!")
    else:
        print("Random player wins.")


if __name__ == "__main__":
    main()
