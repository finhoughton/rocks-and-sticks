from __future__ import annotations
from abc import abstractmethod
from enum import Enum
from operator import itemgetter
from typing import Iterable, Iterator
from collections import deque
from functools import cache, cached_property

second = itemgetter(1)

HALF_AREA_COUNTS = False
STARTING_STICK = True

N_ROCKS = 2
AI_DEPTH = 3


class Player:
    def __init__(self, player_number: int):
        self.number = player_number
        # could maintain legal moves per player in Node: list[Move]

    @abstractmethod
    def get_move(self, game: Game) -> Move:
        pass

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
            value = float('-inf')
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
        else:
            value = float('inf')
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

    # weights for the heuristic evaluation function
    w1 = 2.0
    w2 = 1.0
    w3 = 1.0
    w4 = 2.0
    w5 = 5.0

    def evaluate_position_heuristic(self, game: Game) -> float:
        # This is the real difficult part of the AI, very hard to know what a good/bad heuristic looks like
        if game.winner is not None:
            if game.winner == self.number:
                return float('inf')
            else:
                return float('-inf')

        opp = game.players[1 - self.number]

        potential_area = self.calculate_potential_area(game, self)
        blocking_opponent = self.calculate_blocking_opponent(game, self)
        stick_opportunities = self.count_stick_opportunities(game, self)
        opponent_progress = self.calculate_potential_area(game, opp)
 
        return (
            self.w1 * potential_area +
            self.w2 * blocking_opponent +
            self.w3 * stick_opportunities -
            self.w4 * opponent_progress -
            self.w5 * game.num_rocks[self.number]
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
        # Count how many potential areas of the opponent are disrupted
        return 0

    @classmethod
    def count_stick_opportunities(cls, game: Game, player: Player) -> float:
        total = 0
        for point in game.connected_points:
            if not player.can_place(point):
                continue
            total += 8 - len(point.neighbours)

        return float(total)

class Direction(Enum):
    N =  0, ( 0,  1)
    E =  1, ( 1,  0)
    NE = 2, ( 1,  1)
    SE = 3, ( 1, -1)
    NW = 4, (-1,  1)
    SW = 5, (-1, -1)
    W =  6, (-1,  0)
    S =  7, ( 0, -1)

    @cached_property
    def as_int(self) -> int:
        return self.value[0]
    
    @cached_property
    def delta(self) -> tuple[int, int]:
        return self.value[1]

    @cached_property
    def reversed(self) -> Direction:
        return list(Direction)[7 - self.as_int]

class Move:
    @cache
    def __new__(cls, x: int, y: int, t: str) -> Move:
        return super().__new__(cls)

    def __init__(self, x: int, y: int, t: str):
        self.c = (x, y)
        self.t = t

    def __repr__(self) -> str:
        return f"Move({self.c[0]}, {self.c[1]}, {self.t})"

    def __bool__(self) -> bool: return True

PASS = Move(0, 0, "P")

class Node:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.c = (x, y)
        self.rocked_by: Player | None = None
        self.neighbours: list[Node | None] = [None for _ in Direction]
        self.empty_directions: set[Direction] = set(Direction)
        self.neighbour_count: int = 0

    @property
    def neighbours_list(self) -> Iterable[Node]:
        return [n for n in self.neighbours if n is not None]

    @property
    def connected(self) -> bool:
        return self.neighbour_count > 0

    def set_neighbour(self, d: Direction, neighbour: Node) -> None:
        self.neighbours[d.as_int] = neighbour
        self.empty_directions.remove(d)
        self.neighbour_count += 1

    def clear_neighbour(self, d: Direction) -> None:
        self.neighbours[d.as_int] = None
        self.empty_directions.add(d)
        self.neighbour_count -= 1

    def __hash__(self) -> int:
        return hash(self.c)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.c == other.c

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.x}, {self.y})"

    def find_path(self, end: Node, visited: set[Node]) -> list[Node]:
        if not self.connected or not end.connected:
            return []

        queue: deque[tuple[Node, list[Node]]] = deque([(self, [self])])
        visited.add(self)
        while queue:
            current, path = queue.popleft()
            if current == end:
                return path
            for neighbor in current.neighbours_list:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, [*path, neighbor]))
        return []

@cache
def calculate_end(p: tuple[int, int], d: Direction) -> tuple[int, int]:
    x, y = p
    dx, dy = d.delta
    return (x + dx, y + dy)

def calculate_area(ps: Iterable[Node]) -> int:
    # shoelace formula

    ps = iter(ps)
    first = next(ps)
    area = 0

    prev = first
    for current in ps:
        area += prev.x * current.y - current.x * prev.y
        prev = current

    # Close the polygon by connecting the last point to the first
    area += prev.x * first.y - first.x * prev.y
    return abs(area)

class Stick:
    def __init__(self, start: Node, end: Node, d: Direction):
        self.start = start
        self.d = d
        self.end = end

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Stick):
            return False
        return (self.start == value.start and self.end == value.end
                ) or (self.start == value.end and self.end == value.start)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.start}, {self.end}, Direction.{self.d.name})"
    
    def __str__(self) -> str:
        return f"{self.start} -> {self.end}"

class Game:
    def __init__(self):
        self.turn_number = 0

        self.players: list[Player] = [AIPlayer(0), AIPlayer(1)]
        self.players_scores: list[int] = [0 for _ in self.players]
        self.num_rocks: list[int] = [N_ROCKS for _ in self.players]
        self.num_players = len(self.players)

        self.current_player: int = 0
        self.winner: int | None = None

        self.points: dict[tuple[int, int], Node] = dict()
        self.connected_points: set[Node] = set()
        self.rocks: list[Node] = []
        self.sticks: list[Stick] = []

        self.moves: list[Move] = []
        self._history: list[tuple[int, int, list[int], list[int], int | None]] = [] # (turn_number, current_player, player_scores, num_rocks, winner)

        self.add_node_and_neighbours((0, 0))
        if STARTING_STICK:
            self.add_stick(self.points[(0, 0)], Direction.N)

    def valid_move(self, m: Move, player: Player) -> bool:
        if m is PASS:
            return True
        t = m.t
        c = m.c
        if not (t == "R" or t in Direction._member_names_):
            raise ValueError("Invalid move type")

        point = self.points.get(c)
        if point is None:
            return False

        if t == "R": 
            if self.turn_number == 0:
                return False
            if point.rocked_by is not None:
                return False
            if self.num_rocks[player.number] <= 0:
                return False
            return True

        if not point.connected:
            return False
        d = Direction[t]
        if point.neighbours[d.as_int] is not None:
            return False
        if not player.can_place(point):
            return False
        return True

    def get_node(self, c: tuple[int, int]) -> Node:
        if c in self.points:
            return self.points[c]
        p = Node(*c)
        self.points[c] = p
        return p

    def add_node_and_neighbours(self, c: tuple[int, int]) -> Node:
        p = self.get_node(c)
        for d in Direction:
            end_c = calculate_end(c, d)
            if end_c not in self.points:
                self.points[end_c] = Node(*end_c)
        return p

    def remove_connected_point(self, point: Node) -> None:
        if point.c != (0, 0):
            self.connected_points.remove(point)

    def add_stick(self, start: Node, d: Direction) -> int:
        end_coords = calculate_end(start.c, d)
        end = self.add_node_and_neighbours(end_coords)

        path: list[Node] = list(start.find_path(end, set()))

        stick = Stick(start, end, d)

        stick.start.set_neighbour(d, end)
        stick.end.set_neighbour(d.reversed, start)

        self.connected_points.add(stick.start)
        self.connected_points.add(stick.end)

        self.sticks.append(stick)

        if not path:
            return 0
        return calculate_area(path)

    def remove_stick(self, stick: Stick) -> None:
        # doesn't remove area
        stick.start.clear_neighbour(stick.d)
        stick.end.clear_neighbour(stick.d.reversed)

        if not stick.start.connected:
            self.remove_connected_point(stick.start)
        if not stick.end.connected:
            self.remove_connected_point(stick.end)

    def do_move(self, player: Player, m: Move) -> None:
        # assert (player.number == self.current_player)
        self._history.append((self.turn_number, self.current_player, self.players_scores.copy(), self.num_rocks.copy(), self.winner))

        if m is PASS:
            self.num_rocks[player.number] = N_ROCKS

        elif m.t == "R":
            # point = self.add_node_and_neighbours(m.c)
            # theoretically we should add neighbours, but it increases the search space too much
            point = self.get_node(m.c)
            point.rocked_by = player
            self.rocks.append(point)
            self.num_rocks[player.number] -= 1

        elif m.t in Direction.__members__:
            self.num_rocks[player.number] = N_ROCKS
            v = self.add_stick(self.points[m.c], Direction[m.t])
            if HALF_AREA_COUNTS or v != 1:
                self.players_scores[player.number] += v
        else:
            self._history.pop()
            raise ValueError("Invalid move")

        self.current_player += 1
        if self.current_player == game.num_players:
            self.current_player = 0
            self.turn_number += 1

            players_areas = sorted(enumerate(self.players_scores), key=second)
            players, areas = zip(*players_areas)
            max_a = max(areas)
            if max_a > 0 and areas.count(max_a) == 1:
                self.winner = players[areas.index(max_a)]

        self.moves.append(m)

    def undo_move(self, player: Player) -> None:
        if not self.moves or not self._history:
            raise ValueError("No moves to undo")

        last_move = self.moves.pop()
        point = self.points[last_move.c]

        self.turn_number, self.current_player, self.players_scores, self.num_rocks, self.winner = self._history.pop()

        if last_move is PASS:
            pass

        elif last_move.t == "R":
            point.rocked_by = None
            q = self.rocks.pop()
            assert q == point, "Undoing rock move failed"

        elif last_move.t in Direction.__members__:
            stick = self.sticks.pop()
            assert stick.start.c == last_move.c and stick.d == Direction[last_move.t], "Undoing stick move failed"
            self.remove_stick(stick)

        else:
            raise ValueError("Invalid move type for undo")

    def get_possible_moves(self, player: Player) -> Iterator[Move]:

        for p in list(self.connected_points):
            if not player.can_place(p):
                continue
            for d in p.empty_directions:
                yield Move(p.x, p.y, d.name)

        can_rock = (self.turn_number != 0) and (self.num_rocks[player.number] > 0)

        if can_rock:
            for p in list(self.points.values()):
                if p.rocked_by is None:
                    yield Move(p.x, p.y, "R")
        yield PASS

    # def get_possible_moves(self, player: Player) -> Iterator[Move]:
    #     # testing only. has bad impact on performance. obviously
    #     for move in self._get_possible_moves(player):
    #         assert self.valid_move(move, player)
    #         yield move

    def run(self):
        while True:
            print(f"Turn {self.turn_number}") 
            for player in self.players:
                m: Move = player.get_move(self)
                print(f"Player {player.number + 1} plays {m}")
                self.do_move(player, m)

            if self.turn_number == 2:
                return

            if self.winner is not None:
                print(f"player {self.winner + 1} wins with area {self.players_scores[self.winner] / 2}")
                break

if __name__ == "__main__":
    game = Game()
    game.run()
