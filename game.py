from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

from constants import HALF_AREA_COUNTS, MCTS_SEED, N_ROCKS, STARTING_STICK, second
from models import PASS, D, Move, Node, Stick, calculate_area, calculate_end
from players import HumanPlayer, MCTSPlayer, Player  # type: ignore

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.widgets import TextBox


Edge = tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True)
class ClaimedRegion:
    owner: int
    vertices: tuple[tuple[int, int], ...]
    edge_key: frozenset[Edge]
    bbox: tuple[int, int, int, int]  # (minx, maxx, miny, maxy)


def _norm_edge(a: tuple[int, int], b: tuple[int, int]) -> Edge:
    return (a, b) if a <= b else (b, a)


def _region_edge_key(vertices: tuple[tuple[int, int], ...]) -> frozenset[Edge]:
    if len(vertices) < 3:
        return frozenset()
    edges: set[Edge] = set()
    prev = vertices[-1]
    for cur in vertices:
        edges.add(_norm_edge(prev, cur))
        prev = cur
    return frozenset(edges)


def _point_on_segment(p: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> bool:
    (px, py) = p
    (ax, ay) = a
    (bx, by) = b
    # collinear?
    if (bx - ax) * (py - ay) != (by - ay) * (px - ax):
        return False
    return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)


def _point_in_polygon_strict(point: tuple[int, int], poly: tuple[tuple[int, int], ...]) -> bool:
    """Ray-casting point-in-polygon test.

    Returns True only if strictly inside (points on boundary are False).
    """

    if len(poly) < 3:
        return False

    x, y = point

    # Boundary check first.
    prev = poly[-1]
    for cur in poly:
        if _point_on_segment(point, prev, cur):
            return False
        prev = cur

    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


class Game:
    def __init__(self, players: list[Player] | None = None):
        self.turn_number = 0

        self.players: list[Player]

        if players is None:
            self.players = [HumanPlayer(0), MCTSPlayer(1, seed=MCTS_SEED, time_limit=30.0)]
        else:
            self.players = players

        self.players_scores: list[int] = [0 for _ in self.players]
        self.num_rocks: list[int] = [N_ROCKS for _ in self.players]
        self.num_players = len(self.players)

        self.current_player: int = 0
        self.winner: int | None = None

        self.points: dict[tuple[int, int], Node] = dict()
        self.connected_points: set[Node] = set()
        self.rocks: list[Node] = []
        self.sticks: list[Stick] = []
        self.stick_endpoints: set[Edge] = set()
        self.claimed_regions: list[ClaimedRegion] = []

        self.moves: list[Move] = []
        self._history: list[tuple[int, int, list[int], list[int], int | None, int]] = []

        self._render_fig: "Figure | None" = None
        self._render_ax: "Axes | None" = None
        self._render_warned = False
        self._render_input_ax: "Axes | None" = None
        self._render_text_box: "TextBox | None" = None
        self._pending_input_text: str | None = None

        self.add_node_and_neighbours((0, 0))
        if STARTING_STICK:
            self.add_stick(self.points[(0, 0)], D.N, owner=None)

    def _claimed_edge_keys(self) -> set[frozenset[Edge]]:
        return {r.edge_key for r in self.claimed_regions}

    def coord_in_claimed_region(self, c: tuple[int, int]) -> bool:
        """True if coordinate is strictly inside any claimed region."""

        x, y = c
        for r in self.claimed_regions:
            minx, maxx, miny, maxy = r.bbox
            if x <= minx or x >= maxx or y <= miny or y >= maxy:
                continue
            if _point_in_polygon_strict(c, r.vertices):
                return True
        return False

    def _shortest_paths(self, start: Node, end: Node, max_paths: int = 32) -> list[list[Node]]:
        """Enumerate up to `max_paths` shortest paths from start to end in the current stick graph."""

        if start == end:
            return [[start]]
        if not start.connected or not end.connected:
            return []

        dist: dict[Node, int] = {start: 0}
        parents: dict[Node, list[Node]] = {start: []}
        q: deque[Node] = deque([start])

        while q:
            cur = q.popleft()
            dcur = dist[cur]
            if cur == end:
                # Still continue; we want all parents at shortest distance.
                continue
            for nb in cur.neighbours_list:
                nd = dcur + 1
                if nb not in dist:
                    dist[nb] = nd
                    parents[nb] = [cur]
                    q.append(nb)
                elif dist[nb] == nd:
                    parents[nb].append(cur)

        if end not in dist:
            return []

        out: list[list[Node]] = []
        path: list[Node] = [end]

        def backtrack(node: Node) -> None:
            if len(out) >= max_paths:
                return
            if node == start:
                out.append(list(reversed(path)))
                return
            for p in parents.get(node, []):
                path.append(p)
                backtrack(p)
                path.pop()

        backtrack(end)
        # Deterministic ordering.
        out.sort(key=lambda ps: [(n.x, n.y) for n in ps])
        return out

    def _best_unclaimed_cycle(self, start: Node, end: Node) -> tuple[int, tuple[tuple[int, int], ...], frozenset[Edge]] | None:
        """Pick the best (highest area) unclaimed cycle among shortest start->end paths."""

        candidates = self._shortest_paths(start, end, max_paths=48)
        if not candidates:
            return None

        claimed = self._claimed_edge_keys()
        best: tuple[int, tuple[tuple[int, int], ...], frozenset[Edge]] | None = None

        for path in candidates:
            if len(path) < 3:
                continue
            vertices = tuple(n.c for n in path)
            area2 = calculate_area(path)
            if area2 <= 0:
                continue
            if not HALF_AREA_COUNTS and area2 == 1:
                # This closure doesn't score under current rules; don't claim it.
                continue
            edge_key = _region_edge_key(vertices)
            if edge_key in claimed:
                continue
            if best is None or area2 > best[0]:
                best = (area2, vertices, edge_key)

        return best

    def __deepcopy__(self, memo: dict[int, object]) -> Game:
        # Deepcopy is used by MCTS to create a private working game.
        # Never copy matplotlib objects: deepcopying figures/axes can create
        # a second window and is not part of the logical game state.

        result: Game = type(self).__new__(type(self))
        memo[id(self)] = result

        for key, value in self.__dict__.items():
            if "_render_" in key:
                setattr(result, key, None)
                continue
            setattr(result, key, copy.deepcopy(value, memo))

        return result

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

    def add_stick(self, start: Node, d: D, owner: int | None) -> int:
        end_coords = calculate_end(start.c, d)
        end = self.add_node_and_neighbours(end_coords)

        # If there is already a path between the endpoints, adding this stick closes a cycle.
        # There can be multiple shortest paths; choose a cycle that is not already claimed.
        best_cycle = self._best_unclaimed_cycle(start, end)

        stick = Stick(start, end, d)

        stick.start.set_neighbour(d, end)
        stick.end.set_neighbour(d.reversed, start)

        self.connected_points.add(stick.start)
        self.connected_points.add(stick.end)

        self.sticks.append(stick)
        self.stick_endpoints.add(stick.ordered)

        if best_cycle is None:
            return 0

        area2, vertices, edge_key = best_cycle
        if owner is not None:
            xs = [x for (x, _y) in vertices]
            ys = [y for (_x, y) in vertices]
            bbox = (min(xs), max(xs), min(ys), max(ys))
            self.claimed_regions.append(
                ClaimedRegion(owner=owner, vertices=vertices, edge_key=edge_key, bbox=bbox)
            )
        return area2

    def is_stick(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        return tuple(sorted((start, end))) in self.stick_endpoints

    def remove_top_stick(self) -> Stick:
        stick = self.sticks.pop()
        stick.start.clear_neighbour(stick.d)
        stick.end.clear_neighbour(stick.d.reversed)

        if not stick.start.connected:
            self.remove_connected_point(stick.start)
        if not stick.end.connected:
            self.remove_connected_point(stick.end)

        self.stick_endpoints.discard(stick.ordered)
        return stick

    def do_move(self, player: Player, m: Move) -> None:
        self._history.append(
            (
                self.turn_number,
                self.current_player,
                self.players_scores.copy(),
                self.num_rocks.copy(),
                self.winner,
                len(self.claimed_regions),
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
            v = self.add_stick(self.points[m.c], D[m.t], owner=player.number)
            if v > 0:
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

    def undo_move(self) -> None:
        if not self.moves or not self._history:
            raise ValueError("No moves to undo")

        last_move = self.moves.pop()
        point = self.points[last_move.c]
        (
            self.turn_number,
            self.current_player,
            self.players_scores,
            self.num_rocks,
            self.winner,
            claimed_len,
        ) = self._history.pop()

        # Remove any regions claimed by the undone move.
        if len(self.claimed_regions) > claimed_len:
            del self.claimed_regions[claimed_len:]

        if last_move is PASS:
            return

        if last_move.t == "R":
            point.rocked_by = None
            q = self.rocks.pop()
            assert q == point, "Undoing rock move failed"
            return

        if last_move.t in D.__members__:
            stick = self.remove_top_stick()
            assert stick.start.c == last_move.c and stick.d == D[last_move.t], (
                "Undoing stick move failed"
            )
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
        a = ((mx - dy) // 2, (my + dx) // 2)
        b = ((mx + dy) // 2, (my - dx) // 2)
        return self.is_stick(a, b)

    def get_possible_moves(self, player: Player) -> Iterator[Move]:
        for p in list(self.connected_points):
            if not player.can_place(p):
                continue
            if self.coord_in_claimed_region(p.c):
                continue
            for d in list(p.empty_directions):
                if not self.intersects_stick(p.c, d):
                    end_c = calculate_end(p.c, d)
                    if self.coord_in_claimed_region(end_c):
                        continue
                    yield Move(p.x, p.y, d.name)

        can_rock = (self.turn_number != 0) and (self.num_rocks[player.number] > 0)

        if can_rock:
            xs = [p.x for p in self.connected_points] + [r.x for r in self.rocks]
            ys = [p.y for p in self.connected_points] + [r.y for r in self.rocks]
            minx, maxx = min(xs) - 1, max(xs) + 1
            miny, maxy = min(ys) - 1, max(ys) + 1
            for p in list(self.points.values()):
                if p.rocked_by is None and minx <= p.x <= maxx and miny <= p.y <= maxy:
                    if self.coord_in_claimed_region(p.c):
                        continue
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
            if self.coord_in_claimed_region(c):
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
        if self.coord_in_claimed_region(c):
            return False
        if self.intersects_stick(c, d):
            return False
        if self.coord_in_claimed_region(end_coords):
            return False
        return True

    def wait_for_move_input(self, prompt: str) -> str:
        """Block until a move string is supplied via UI or stdin."""

        if self._render_text_box is None:
            # Ensure UI exists; if still absent, fall back to stdin.
            self.render(block=False)
            if self._render_text_box is None:
                return input(prompt)

        import matplotlib.pyplot as plt

        print(prompt, flush=True)
        self._pending_input_text = None
        while self._pending_input_text is None:
            try:
                plt.pause(0.05)  # type: ignore
            except SystemError:
                continue

        text = self._pending_input_text or ""
        self._pending_input_text = None
        return text

    def render(self, block: bool = False, game_index: int | None = None, total_games: int | None = None) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import TextBox

        if self._render_fig is None or self._render_ax is None:
            fig_ax: tuple[Figure, Axes] = plt.subplots()  # type: ignore
            fig, ax = fig_ax
            self._render_fig = fig
            self._render_ax = ax
            # Make room at the bottom for an input box.
            fig.subplots_adjust(bottom=0.22)

            input_ax: Axes = fig.add_axes((0.15, 0.06, 0.35, 0.08)) # type: ignore

            def _on_submit(text: str) -> None:
                cleaned = text.strip()
                if not cleaned:
                    text_box.set_val("")
                    return
                self._pending_input_text = cleaned
                text_box.set_val("")

            text_box = TextBox(input_ax, "Input:", initial="")
            text_box.on_submit(_on_submit)

            self._render_input_ax = input_ax
            self._render_text_box = text_box

        ax = self._render_ax
        ax_any: Any = ax
        ax.clear()

        # Draw claimed regions first (under sticks/rocks).
        for region in self.claimed_regions:
            xs = [c[0] for c in region.vertices]
            ys = [c[1] for c in region.vertices]
            if len(xs) >= 3:
                ax_any.fill(xs, ys, color=f"C{region.owner}", alpha=0.18, zorder=0)

        for stick in self.sticks:
            ax_any.plot(
                [stick.start.x, stick.end.x],
                [stick.start.y, stick.end.y],
                color="black",
                linewidth=2,
                zorder=1,
            )

        for point in self.points.values():
            if point.rocked_by is not None:
                color = f"C{point.rocked_by.number}"
                marker = "o"
                ax_any.scatter(point.x, point.y, c=color, marker=marker, s=80, zorder=2)

        xs = [p.x for p in self.points.values() if p.connected or p.rocked_by is not None]
        ys = [p.y for p in self.points.values() if p.connected or p.rocked_by is not None]
        if self.claimed_regions:
            for r in self.claimed_regions:
                xs.extend([c[0] for c in r.vertices])
                ys.extend([c[1] for c in r.vertices])
        if xs and ys:
            ax_any.set_xlim(min(xs) - 1, max(xs) + 1)
            ax_any.set_ylim(min(ys) - 1, max(ys) + 1)

        ax_any.set_aspect("equal", "box")
        ax_any.grid(True, alpha=0.2)
        prefix = ""
        if game_index is not None:
            if total_games is not None:
                prefix = f"Game {game_index + 1}/{total_games} — "
            else:
                prefix = f"Game {game_index + 1} — "
        ax_any.set_title(f"{prefix}Turn {self.turn_number} — Player {self.current_player + 1} to move")

        try:
            plt.pause(0.01)  # type: ignore
        except SystemError:
            return
        if block:
            plt.show()  # type: ignore

    def run(self, display: bool = True) -> None:
        import time

        while True:
            print(f"Turn {self.turn_number}")
            for player in self.players:
                if display:
                    self.render(block=False)
                t0 = time.perf_counter()
                m: Move = player.get_move(self)
                dt = time.perf_counter() - t0
                # Only print timing for non-human players (bots).
                if player.__class__.__name__ != "HumanPlayer":
                    extra = ""
                    # MCTSPlayer exposes rollout settings; print actual rollouts when present.
                    rollouts_used = getattr(player, "last_rollouts", None)
                    sim_depth = getattr(player, "max_sim_depth", None)
                    if isinstance(rollouts_used, int) and isinstance(sim_depth, int):
                        extra = f" (rollouts={rollouts_used}, sim_depth={sim_depth})"
                    print(f"Player {player.number + 1} ({player.__class__.__name__}) move time: {dt:.3f}s{extra}")
                print(f"Player {player.number + 1} plays {m}")
                self.do_move(player, m)
                if display:
                    self.render(block=False)

            if self.winner is not None:
                print(f"player {self.winner + 1} wins with area {self.players_scores[self.winner] / 2}")
                if display:
                    self.render(block=True)
                break
