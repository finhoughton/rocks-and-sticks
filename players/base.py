from __future__ import annotations

import random
from abc import abstractmethod
from contextlib import contextmanager
from typing import TYPE_CHECKING

from models import Move

if TYPE_CHECKING:
    from game import Game
    from models import Node

StateKey = tuple[object, ...]


def _game_key(game: Game) -> StateKey:
    # Canonical, hashable description of game state for MCTS
    sticks = tuple(sorted(s.ordered for s in game.sticks))
    rocks = tuple(sorted(r.c for r in game.rocks))
    return (
        game.winner,
        game.turn_number,
        game.current_player,
        tuple(game.players_scores),
        tuple(game.num_rocks),
        rocks,
        sticks,
    )


@contextmanager
def applied_move(game: Game, player: "Player", move: Move):
    game.do_move(player, move)
    yield
    game.undo_move()


@contextmanager
def rollback_to(game: Game):
    start_len = len(game.moves)
    yield
    while len(game.moves) > start_len:
        game.undo_move()


class Player:
    def __init__(self, player_number: int):
        self.use_gnn_eval = False
        self.number = player_number

    @abstractmethod
    def get_move(self, game: "Game") -> Move:
        raise NotImplementedError

    def can_place(self, point: "Node") -> bool:
        # Identity comparison avoids calling Player.__eq__ in hot paths.
        return point.rocked_by is None or point.rocked_by is self

    def __hash__(self) -> int:
        return self.number

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Player):
            return False
        return self.number == value.number

    def __deepcopy__(self, memo: dict[int, object]) -> "Player":
        # Players are effectively immutable identifiers
        return self


class HumanPlayer(Player):
    def get_move(self, game: Game) -> Move:
        inp = game.wait_for_move_input(f"Player {self.number + 1}, enter move: ")
        if inp == "P":
            from models import PASS

            return PASS
        try:
            x_s, y_s, t = inp.split()
            x = int(x_s)
            y = int(y_s)

            m = Move(x, y, t)

            if not game.valid_move(m, self):
                raise ValueError("Invalid move")

        except Exception as e:
            print(f"{str(e)}, try again. got input: {inp}")
            return self.get_move(game)
        return m


class RandomPlayer(Player):
    def __init__(self, player_number: int, seed: int | None = None):
        super().__init__(player_number)
        self._rng = random.Random(seed)

    def get_move(self, game: Game) -> Move:
        from models import PASS, move_key

        moves = sorted((m for m in game.get_possible_moves(self) if m is not PASS), key=move_key)
        if not moves:
            from models import PASS

            return PASS
        return self._rng.choice(moves)


class RockBiasedRandomPlayer(RandomPlayer):
    # Random player with a bias toward rock moves to diversify play

    def get_move(self, game: Game) -> Move:
        from models import PASS, move_key

        moves = sorted((m for m in game.get_possible_moves(self) if m is not PASS), key=move_key)
        rock_moves = [m for m in moves if m.t == "R"]
        if rock_moves and self._rng.random() < 0.6:
            return self._rng.choice(rock_moves)
        return self._rng.choice(moves)
