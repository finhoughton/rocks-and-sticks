from __future__ import annotations

from typing import Iterator, Any

from constants import HALF_AREA_COUNTS, STARTING_STICK, N_ROCKS, second
from models import D, Move, Node, PASS, Stick, calculate_area, calculate_end
from players import AlphaBetaPlayer, MCTSPlayer, HumanPlayer, Player # type: ignore

from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Game:
    def __init__(self):
        self.turn_number = 0

        self.players: list[Player] = [HumanPlayer(0), MCTSPlayer(1)]
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

        self._render_fig: Figure | None = None
        self._render_ax: Axes | None = None
        self._render_warned = False

        self.add_node_and_neighbours((0, 0))
        if STARTING_STICK:
            self.add_stick(self.points[(0, 0)], D.N)

    def get_node(self, c: tuple[int, int]) -> Node:
        if c in self.points:
            return self.points[c]
        p = Node(*c)
        self.points[c] = p
        return p

    def add_node_and_neighbours(self, c: tuple[int, int]) -> Node:
        p = self.get_node(c)
        for d in D:
            end_c = calculate_end(c, d)
            if end_c not in self.points:
                self.points[end_c] = Node(*end_c)
        return p

    def remove_connected_point(self, point: Node) -> None:
        if point.c != (0, 0):
            self.connected_points.discard(point)

    def add_stick(self, start: Node, d: D) -> int:
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

    def is_stick(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        if start not in self.points:
            return False
        if end not in self.points:
            return False
        return Stick(self.points[start], self.points[end], D.NO_DIR) in self.sticks

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

        elif m.t in D.__members__:
            self.num_rocks[player.number] = N_ROCKS
            v = self.add_stick(self.points[m.c], D[m.t])
            if HALF_AREA_COUNTS or v != 1:
                self.players_scores[player.number] += v
        else:
            self._history.pop()
            raise ValueError("Invalid move")

        self.current_player += 1
        if self.current_player == self.num_players:
            self.current_player = 0
            self.turn_number += 1

            if self.turn_number > 0:
                players_areas = sorted(enumerate(self.players_scores), key=second)
                players, areas = zip(*players_areas)
                max_a = max(areas)

                if max_a > 0 and areas.count(max_a) == 1:
                    leader_idx = areas.index(max_a)
                    leader = players[leader_idx]
                    if leader == 0 and self.turn_number == 1:
                        # Player 1 scored first, player 2 gets this turn
                        pass
                    elif leader == 1 or (leader == 0 and self.turn_number > 1):
                        self.winner = leader

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

        if last_move.t in D.__members__:
            stick = self.sticks.pop()
            assert stick.start.c == last_move.c and stick.d == D[last_move.t], "Undoing stick move failed"
            self.remove_stick(stick)
            return

        raise ValueError("Invalid move type for undo")
    
    def intersects_stick(self, start: tuple[int, int], d: D) -> bool:
        # to check if a diagonal stick intersects an existing stick
        if not d.is_diagonal:
            return False
        (x1, y1) = start
        (x2, y2) = calculate_end(start, d)
        dx = x2 - x1
        dy = y2 - y1
        mx = x1 + x2
        my = y1 + y2
        a = ((mx - dy)//2, (my + dx)//2)
        b = ((mx + dy)//2, (my - dx)//2)
        return self.is_stick(a, b)

    def get_possible_moves(self, player: Player) -> Iterator[Move]:
        for p in list(self.connected_points):
            if not player.can_place(p):
                continue
            for d in list(p.empty_directions):
                if not self.intersects_stick(p.c, d):
                    yield Move(p.x, p.y, d.name)
 
        can_rock = (self.turn_number != 0) and (self.num_rocks[player.number] > 0)

        if can_rock:
            for p in list(self.points.values()):
                if p.rocked_by is None:
                    yield Move(p.x, p.y, "R")
        yield PASS

    def valid_move(self, m: Move, player: Player) -> bool:
        if m is PASS:
            return True
        t = m.t
        c = m.c
        if not (t == "R" or t in D._member_names_):
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
        d = D[t]
        if point.neighbours[d.as_int] is not None:
            return False
        # Check if the end node also doesn't already have this connection
        end_coords = calculate_end(c, d)
        end_point = self.points.get(end_coords)
        if end_point and end_point.neighbours[d.reversed.as_int] is not None:
            return False
        if not player.can_place(point):
            return False
        if self.intersects_stick(c, d):
            return False
        return True

    def render(self, block: bool = False) -> None:
        import matplotlib.pyplot as plt

        if self._render_fig is None or self._render_ax is None:
            fig_ax: tuple[Figure, Axes] = plt.subplots() #type: ignore
            fig, ax = fig_ax
            self._render_fig = fig
            self._render_ax = ax

        ax = self._render_ax
        ax_any: Any = ax
        ax.clear()

        for stick in self.sticks:
            ax_any.plot([stick.start.x, stick.end.x], [stick.start.y, stick.end.y], color="black", linewidth=2, zorder=1)

        for point in self.points.values():
            if point.rocked_by is not None:
                color = f"C{point.rocked_by.number}"
                marker = "o"
                ax_any.scatter(point.x, point.y, c=color, marker=marker, s=80, zorder=2)

        xs = [p.x for p in self.points.values() if p.connected or p.rocked_by is not None]
        ys = [p.y for p in self.points.values() if p.connected or p.rocked_by is not None]
        if xs and ys:
            ax_any.set_xlim(min(xs) - 1, max(xs) + 1)
            ax_any.set_ylim(min(ys) - 1, max(ys) + 1)

        ax_any.set_aspect("equal", "box")
        ax_any.grid(True, alpha=0.2)
        ax_any.set_title(f"Turn {self.turn_number} — Player {self.current_player + 1} to move")

        plt.pause(0.001)
        if block:
            plt.show() # type: ignore

    def run(self, display: bool = True) -> None:
        while True:
            print(f"Turn {self.turn_number}")
            for player in self.players:
                m: Move = player.get_move(self)
                print(f"Player {player.number + 1} plays {m}")
                self.do_move(player, m)
                if display:
                    self.render(block=False)

            if self.winner is not None:
                print(
                    f"player {self.winner + 1} wins with area {self.players_scores[self.winner] / 2}"
                )
                if display:
                    self.render(block=True)
                break
