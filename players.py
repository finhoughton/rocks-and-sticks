from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from constants import AI_DEPTH
from models import Move, PASS, calculate_end

if TYPE_CHECKING:
    from game import Game
    from models import Node


class Player:
    def __init__(self, player_number: int):
        self.number = player_number

    @abstractmethod
    def get_move(self, game: Game) -> Move:
        raise NotImplementedError

    def can_place(self, point: Node) -> bool:
        return point.rocked_by is None or point.rocked_by == self

    def __hash__(self) -> int:
        return self.number

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Player):
            return False
        return self.number == value.number


class HumanPlayer(Player):
    def get_move(self, game: Game) -> Move:
        inp = input(f"Player {self.number + 1}, enter move: ")
        if inp == "P":
            return PASS
        try:
            x_s, y_s, t = inp.split()
            x = int(x_s)
            y = int(y_s)

            m = Move(x, y, t)

            if not game.valid_move(m, self):
                raise ValueError("Invalid move")

        except Exception as e:
            print(f"{str(e)}, try again")
            return self.get_move(game)
        return m


class AIPlayer(Player):
    def get_move(self, game: Game) -> Move:
        if game.num_players != 2:
            raise ValueError("AI is only implemented for 2 players")
        move, _ = self.alpha_beta(game, AI_DEPTH, float("-inf"), float("inf"), True)
        return move

    def alpha_beta(self, game: Game, depth: int, a: float, b: float, maximising: bool) -> tuple[Move, float]:
        if depth == 0 or game.winner is not None:
            return (PASS, self.evaluate_position_heuristic(game))

        best_move = PASS

        if maximising:
            value = float("-inf")
            for move in game.get_possible_moves(self):
                game.do_move(self, move)
                _, v2 = self.alpha_beta(game, depth - 1, a, b, False)
                game.undo_move(self)

                if v2 > value:
                    value = v2
                    best_move = move
                if value > b:
                    break
                a = max(a, value)
            return (best_move, value)
        value = float("inf")
        opp = game.players[1 - self.number]
        for move in game.get_possible_moves(opp):
            game.do_move(opp, move)
            _, v2 = self.alpha_beta(game, depth - 1, a, b, True)
            game.undo_move(opp)
            if v2 < value:
                value = v2
                best_move = move
            if value < a:
                break
            b = min(b, value)
        return (best_move, value)

    w1 = 2.0
    w2 = 1.0
    w3 = 1.0
    w4 = 2.0
    w5 = 5.0

    def evaluate_position_heuristic(self, game: Game) -> float:
        if game.winner is not None:
            if game.winner == self.number:
                return float("inf")
            return float("-inf")

        opp = game.players[1 - self.number]

        potential_area = self.calculate_potential_area(game, self)
        blocking_opponent = self.calculate_blocking_opponent(game, self)
        stick_opportunities = self.count_stick_opportunities(game, self)
        opponent_progress = self.calculate_potential_area(game, opp)

        return (
            self.w1 * potential_area
            + self.w2 * blocking_opponent
            + self.w3 * stick_opportunities
            - self.w4 * opponent_progress
            - self.w5 * game.num_rocks[self.number]
        )

    @classmethod
    def calculate_potential_area(cls, game: Game, player: Player) -> float:
        potential_area = 0
        for point in game.connected_points:
            if not player.can_place(point):
                continue
            for d in point.empty_directions:
                end_coords = calculate_end(point.c, d)
                end = game.points.get(end_coords)
                if end and player.can_place(end):
                    potential_area += 1
        return potential_area

    @classmethod
    def calculate_blocking_opponent(cls, game: Game, player: Player) -> float:
        return 0

    @classmethod
    def count_stick_opportunities(cls, game: Game, player: Player) -> float:
        total = 0
        for point in game.connected_points:
            if not player.can_place(point):
                continue
            total += 8 - len(point.neighbours)

        return float(total)
