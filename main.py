from game import Game
from players import AlphaBetaPlayer, HumanPlayer, MCTSPlayer  # type: ignore

if __name__ == "__main__":
    game = Game(players=[HumanPlayer(0), MCTSPlayer(1, time_limit=2.0)])
    game.run(display=True)
