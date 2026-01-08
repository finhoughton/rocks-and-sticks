import argparse
import json
from typing import List, Optional, Tuple

from game import Game
from models import PASS, Move
from players import Player, RandomPlayer


def load_moves_from_json(path: str) -> Tuple[List[Move], Optional[str]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    moves: List[Move] = []
    for mv in payload["moves"]:
        if mv["t"] == "P":
            moves.append(PASS)
        else:
            moves.append(Move(int(mv["x"]), int(mv["y"]), str(mv["t"])))
    winner: Optional[str] = payload.get("winner", None)
    return moves, winner


def rewatch_game(path: str, delay: float = 1.0) -> None:
    moves, winner = load_moves_from_json(path)
    players: List[Player] = [RandomPlayer(0), RandomPlayer(1)]
    game = Game(players)

    import matplotlib.pyplot as plt

    game.render(block=False)
    for i, mv in enumerate(moves):
        print(f"Move {i+1}: Player {game.current_player+1} plays {mv}")
        game.do_move(game.current_player, mv)
        game.render(block=False)
        plt.pause(delay)
    print(f"Game over. Winner: {winner}")

    plt.show() # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rewatch a saved Rocks and Sticks game")
    parser.add_argument("game_file", type=str, help="Path to saved game JSON")
    parser.add_argument("--delay", type=float, default=0.8, help="Delay between moves (seconds)")
    args = parser.parse_args()
    rewatch_game(args.game_file, delay=args.delay)
