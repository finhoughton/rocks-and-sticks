from __future__ import annotations

from typing import Iterator

from constants import HALF_AREA_COUNTS, STARTING_STICK, N_ROCKS, second
from models import Direction, Move, Node, PASS, Stick, calculate_area, calculate_end
from players import AIPlayer, HumanPlayer, Player # type: ignore


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
        self._history: list[tuple[int, int, list[int], list[int], int | None]] = []

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
            self.connected_points.discard(point)

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
        stick.start.clear_neighbour(stick.d)
        stick.end.clear_neighbour(stick.d.reversed)

        if not stick.start.connected:
            self.remove_connected_point(stick.start)
        if not stick.end.connected:
            self.remove_connected_point(stick.end)

    def do_move(self, player: Player, m: Move) -> None:
        self._history.append(
            (
                self.turn_number,
                self.current_player,
                self.players_scores.copy(),
                self.num_rocks.copy(),
                self.winner,
            )
        )

        if m is PASS:
            self.num_rocks[player.number] = N_ROCKS

        elif m.t == "R":
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
        if self.current_player == self.num_players:
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
            return

        if last_move.t == "R":
            point.rocked_by = None
            q = self.rocks.pop()
            assert q == point, "Undoing rock move failed"
            return

        if last_move.t in Direction.__members__:
            stick = self.sticks.pop()
            assert stick.start.c == last_move.c and stick.d == Direction[last_move.t], "Undoing stick move failed"
            self.remove_stick(stick)
            return

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

    def run(self) -> None:
        while True:
            print(f"Turn {self.turn_number}")
            for player in self.players:
                m: Move = player.get_move(self)
                print(f"Player {player.number + 1} plays {m}")
                self.do_move(player, m)

            if self.winner is not None:
                print(
                    f"player {self.winner + 1} wins with area {self.players_scores[self.winner] / 2}"
                )
                break
