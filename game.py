from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional, Protocol

from constants import HALF_AREA_COUNTS, N_ROCKS, STARTING_STICK, second
from models import (
    PASS,
    D,
    Move,
    Node,
    Stick,
    calculate_area,
    calculate_end,
    delta_to_direction,
)
from players import Player  # type: ignore

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

# Directions in anticlockwise order, used to walk round faces.
_ACW = (D.E, D.NE, D.N, D.NW, D.W, D.SW, D.S, D.SE)
_ACW_POS = {d: i for i, d in enumerate(_ACW)}


class GameProtocol(Protocol):
    players: list[Player]
    current_player: int
    turn_number: int
    winner: Optional[int]
    players_scores: list[int]
    num_rocks: list[int]
    num_players: int
    moves: list
    sticks: list[Stick]
    rocks: list[Node]
    points: dict[tuple[int, int], Node]
    connected_points: set[Node]

    # all "public" methods (used by external code)
    def do_move(self, player_number: int, m: Move) -> None: ...

    def undo_move(self) -> None: ...

    def get_possible_moves(self, player_number: int) -> Iterator[Move]: ...

    def set_current_player0(self) -> None: ... # used by randomize_start

    def valid_move(self, m: Move, player_number: int) -> bool: ...

    def wait_for_move_input(self, prompt: str) -> str: ...

    def render(self, block: bool = False, game_index: Optional[int] = None, total_games: Optional[int] = None) -> None: ...

    def intersects_stick(self, start: tuple[int, int], d: D) -> bool: ...

    def coord_in_claimed_region(self, c: tuple[int, int]) -> bool: ...

    def _path_of_smallest_area(self, start: Node, end: Node) -> tuple[int, tuple[tuple[int, int], ...], frozenset[Edge]] | None: ...

class Game:
    def __init__(self, players: list[Player] | None = None) -> None:
        self.turn_number = 0

        self.players: list[Player]

        self.players = players if players is not None else [Player(0), Player(1)]

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

        # Cached reachability map used by coord_in_claimed_region().
        self._reachable_cache_key: int | None = None
        self._reachable_lowx = 0
        self._reachable_highx = -1
        self._reachable_lowy = 0
        self._reachable_highy = -1
        self._reachable_coords: set[tuple[int, int]] = set()

        # Cache for diagonal intersection checks: (x, y) -> bool
        self._intersects_cache: set[tuple[int, int]] = set()

        self.moves: list[Move] = []
        self._history: list[tuple[int, int, list[int], list[int], int | None, int]] = []

        self._render_fig: "Figure | None" = None
        self._render_ax: "Axes | None" = None
        self._render_warned = False
        self._render_input_ax: "Axes | None" = None
        self._render_text_box: "TextBox | None" = None
        self._pending_input_text: str | None = None
        self._render_rock_texts: list[Any] = []
        self._render_resign_button: Any = None
        self._resign_armed = False
        # set by the caller when a game ends some other way than by the rules (a resignation)
        self.result_note: str | None = None

        self.add_node_and_neighbours((0, 0))
        if STARTING_STICK:
            self.add_stick(self.points[(0, 0)], D.N, owner=None)
    
    def set_current_player0(self) -> None:
        self.current_player = 0

    def _claimed_edge_keys(self) -> set[frozenset[Edge]]:
        return {r.edge_key for r in self.claimed_regions}

    def coord_in_claimed_region(self, c: tuple[int, int]) -> bool:

        if not self.connected_points:
            return False

        self._ensure_reachable_cache()

        x, y = c
        if x < self._reachable_lowx or x > self._reachable_highx or y < self._reachable_lowy or y > self._reachable_highy:
            return False

        return c not in self._reachable_coords

    def _invalidate_reachable_cache(self) -> None:
        self._reachable_cache_key = None
        self._reachable_coords.clear()

    def _sticks_cache_key(self) -> int:
        return hash(tuple(sorted(self.stick_endpoints)))

    def _ensure_reachable_cache(self) -> None:
        key = self._sticks_cache_key()
        if self._reachable_cache_key == key:
            return

        minx = min(p.x for p in self.connected_points)
        maxx = max(p.x for p in self.connected_points)
        miny = min(p.y for p in self.connected_points)
        maxy = max(p.y for p in self.connected_points)

        margin = 1
        lowx = minx - margin
        highx = maxx + margin
        lowy = miny - margin
        highy = maxy + margin

        def in_bounds(cx: int, cy: int) -> bool:
            return lowx <= cx <= highx and lowy <= cy <= highy

        def blocked_edge(cx: int, cy: int, nx: int, ny: int, d: D) -> bool:
            p = self.points.get((cx, cy))
            if p is not None and p.neighbours[d.as_int] is not None:
                return True
            q = self.points.get((nx, ny))
            if q is not None and q.neighbours[d.reversed.as_int] is not None:
                return True
            return False

        reachable: set[tuple[int, int]] = set()
        q: deque[tuple[int, int]] = deque()

        for x in range(lowx, highx + 1):
            q.append((x, lowy))
            q.append((x, highy))
        for y in range(lowy, highy + 1):
            q.append((lowx, y))
            q.append((highx, y))

        while q:
            cx, cy = q.popleft()
            if (cx, cy) in reachable:
                continue
            if not in_bounds(cx, cy):
                continue
            reachable.add((cx, cy))

            for d in D:
                dx, dy = d.delta
                nx, ny = cx + dx, cy + dy
                if not in_bounds(nx, ny):
                    continue
                if blocked_edge(cx, cy, nx, ny, d):
                    continue
                if (nx, ny) not in reachable:
                    q.append((nx, ny))

        self._reachable_cache_key = key
        self._reachable_lowx = lowx
        self._reachable_highx = highx
        self._reachable_lowy = lowy
        self._reachable_highy = highy
        self._reachable_coords = reachable
    
    def _face_cycle(self, u: Node, v: Node) -> list[Node] | None:
        """Walk round the face that lies on one side of the stick u-v.

        Starting along u -> v, always turn to the next stick in anticlockwise order after the
        one just walked; that follows the boundary of a single face. Dead-end branches and
        anything hanging off the boundary at a single vertex make the walk revisit a vertex,
        and those loops are cut out, leaving the simple polygon v ... u (closed by u -> v).
        Returns None if the stick has no polygon on this side.
        """
        path: list[Node] = [v]
        index: dict[Node, int] = {v: 0}
        cur = v
        back = _ACW_POS[delta_to_direction((u.x - v.x, u.y - v.y))]  # type: ignore[index]
        for _ in range(4 * len(self.sticks) + 16):
            nxt = None
            for k in range(1, 9):
                d = _ACW[(back + k) % 8]
                nxt = cur.neighbours[d.as_int]
                if nxt is not None:
                    break
            if nxt is None:
                return None
            if cur is u and nxt is v:
                return path if len(path) >= 3 else None
            j = index.get(nxt)
            if j is None:
                index[nxt] = len(path)
                path.append(nxt)
            else:
                for dropped in path[j + 1:]:
                    del index[dropped]
                del path[j + 1:]
            back = _ACW_POS[d.reversed]
            cur = nxt
        return None

    def _path_of_smallest_area(self, start: Node, end: Node) -> tuple[int, tuple[tuple[int, int], ...], frozenset[Edge]] | None:
        """The smallest polygon that the stick start-end completes (or would complete).

        Sticks never cross, so they divide the plane into faces, and every polygon through a
        stick contains the face on one of its two sides. The smallest polygon is therefore the
        smaller of those two faces' outlines, which is found by walking round each.
        Returns (twice the area, polygon vertices, edge key) or None if there is no polygon.
        """
        if start is end or not (start.connected and end.connected):
            return None
        d = delta_to_direction((end.x - start.x, end.y - start.y))
        if d is None:
            return None
        hypothetical = start.neighbours[d.as_int] is None
        if hypothetical:
            if self.intersects_stick(start.c, d):
                return None  # the stick could never be placed
            start.set_neighbour(d, end)
            end.set_neighbour(d.reversed, start)
        try:
            best: tuple[int, tuple[tuple[int, int], ...], frozenset[Edge]] | None = None
            if start.neighbour_count < 2 or end.neighbour_count < 2:
                return None  # one end hangs free: nothing is enclosed
            claimed = self._claimed_edge_keys()
            for u, v in ((start, end), (end, start)):
                cycle = self._face_cycle(u, v)
                if cycle is None:
                    continue
                area2 = calculate_area(cycle)
                if area2 == 0:
                    continue
                vertices = tuple(p.c for p in cycle)
                edge_key = _region_edge_key(vertices)
                if edge_key in claimed:
                    continue
                if best is None or area2 < best[0]:
                    best = (area2, vertices, edge_key)
            return best
        finally:
            if hypothetical:
                start.clear_neighbour(d)
                end.clear_neighbour(d.reversed)

    def __deepcopy__(self, memo: dict[int, object]) -> Game:
        # Deepcopy is used by MCTS to create a private working game.
        # copying matplotlib objects can cause bugs, skip them.

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

        # Invalidate only the cache entries that would have relied on this stick.
        self._flip_intersects_cache_for_stick(start.c, d)

        stick = Stick(start, end, d)

        stick.start.set_neighbour(d, end)
        stick.end.set_neighbour(d.reversed, start)

        self.connected_points.add(stick.start)
        self.connected_points.add(stick.end)

        self.sticks.append(stick)
        self.stick_endpoints.add(stick.ordered)
        self._invalidate_reachable_cache()

        best_cycle = self._path_of_smallest_area(start, end)
        if best_cycle is None:
            return 0
        area2, vertices, edge_key = best_cycle
        if not HALF_AREA_COUNTS and area2 == 1:
            return 0

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

        # Removing a stick flips the answer for any cached crossing checks involving it.
        self._flip_intersects_cache_for_stick(stick.start.c, stick.d)

        if not stick.start.connected:
            self.remove_connected_point(stick.start)
        if not stick.end.connected:
            self.remove_connected_point(stick.end)

        self.stick_endpoints.discard(stick.ordered)
        self._invalidate_reachable_cache()
        return stick

    def do_move(self, player_number: int, m: Move) -> None:
        player_obj = self.players[player_number]

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
            pass

        elif m.t == "R":
            if not self.valid_move(m, player_number):
                self._history.pop()
                raise ValueError(f"Invalid move {m}")
            point = self.get_node(m.c)
            point.rocked_by = player_obj
            self.rocks.append(point)
            self.num_rocks[player_number] -= 1

        elif m.t in D.__members__:
            self.num_rocks[player_number] = N_ROCKS
            v = self.add_stick(self.points[m.c], D[m.t], owner=player_number)
            if v > 0:
                self.players_scores[player_number] += v
        else:
            self._history.pop()
            raise ValueError(f"Invalid move {m}")

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
        (d1, d2) = d.delta
        key = (2 * x1 + d1, 2 * y1 + d2)
        return key in self._intersects_cache

    def _flip_intersects_cache_for_stick(self, start: tuple[int, int], d: D) -> None:
        if not d.is_diagonal:
            return
        (x1, y1) = start
        (d1, d2) = d.delta
        key = (2 * x1 + d1, 2 * y1 + d2)
        if key in self._intersects_cache:
            self._intersects_cache.remove(key)
        else:
            self._intersects_cache.add(key)

    def get_possible_moves(self, player_number: int) -> Iterator[Move]:
        coord_in_region = self.coord_in_claimed_region
        intersects_stick = self.intersects_stick
        end_fn = calculate_end
        player = self.players[player_number]

        for p in list(self.connected_points):
            if not player.can_place(p):
                continue
            pc = p.c
            if coord_in_region(pc):
                continue
            for d in list(p.empty_directions):
                if intersects_stick(pc, d):
                    continue
                end_c = end_fn(pc, d)
                if coord_in_region(end_c):
                    continue
                yield Move(p.x, p.y, d.name)

        can_rock = (self.turn_number != 0) and (self.num_rocks[player_number] > 0)

        if can_rock:
            anchors = list(self.connected_points) + list(self.rocks)
            cand_coords: set[tuple[int, int]] = set()
            for a in anchors:
                ax, ay = a.c
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        cand_coords.add((ax + dx, ay + dy))
            for c in cand_coords:
                if coord_in_region(c):
                    continue
                p = self.points.get(c)
                if p is not None and p.rocked_by is not None:
                    continue
                yield Move(c[0], c[1], "R")
        yield PASS

    def valid_move(self, m: Move, player_number: int) -> bool:
        player = self.players[player_number]
        if m is PASS:
            return True
        t = m.t
        c = m.c
        if not (t == "R" or t in D._member_names_):
            raise ValueError("Invalid move type")

        point = self.points.get(c)
        if t == "R":
            if self.turn_number == 0:
                return False
            if point is not None and point.rocked_by is not None:
                return False
            if self.num_rocks[player_number] <= 0:
                return False
            if self.coord_in_claimed_region(c):
                return False

            x, y = c
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    point = self.points.get((x + dx, y + dy))
                    if point is not None and (point.connected or point.rocked_by is not None):
                        return True
            return False

        if point is None:
            return False

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
        self._set_resign_armed(False)
        while self._pending_input_text is None:
            try:
                plt.pause(0.05)  # type: ignore
            except SystemError:
                continue

        text = self._pending_input_text or ""
        self._pending_input_text = None
        return text

    def _set_resign_armed(self, armed: bool) -> None:
        """The Resign button asks for a second click before it resigns."""
        self._resign_armed = armed
        button = self._render_resign_button
        if button is not None:
            button.label.set_text("Click again\nto resign" if armed else "Resign")
            button.ax.figure.canvas.draw_idle()

    def render(self, block: bool = False, game_index: int | None = None, total_games: int | None = None) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, TextBox

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

            # Rocks each player may still place before their next stick, right of the input box.
            self._render_rock_texts = [
                fig.text(0.525, 0.115 - 0.045 * i, "", color=f"C{i}", fontsize=10, va="center") for i in range(self.num_players)
            ]

            # Resign: the same as typing "resign". Two clicks, so a stray click cannot end the game.
            def _on_resign(_event: Any) -> None:
                if not self._resign_armed:
                    self._set_resign_armed(True)
                    return
                self._set_resign_armed(False)
                self._pending_input_text = "resign"

            resign_button = Button(fig.add_axes((0.81, 0.06, 0.13, 0.08)), "Resign")  # type: ignore
            resign_button.label.set_fontsize(9)
            resign_button.on_clicked(_on_resign)
            self._render_resign_button = resign_button

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
        if self.winner is not None:
            status = f"Player {self.winner + 1} wins" + (f" ({self.result_note})" if self.result_note else "")
        else:
            status = f"Player {self.current_player + 1} to move"
        ax_any.set_title(f"{prefix}Turn {self.turn_number} — {status}")

        humans = [p.__class__.__name__ == "HumanPlayer" for p in self.players]
        for i, text in enumerate(self._render_rock_texts):
            who = " (you)" if humans[i] and sum(humans) == 1 else ""
            left = self.num_rocks[i]
            text.set_text(f"Player {i + 1}{who} rocks  " + "●" * left + "○" * (N_ROCKS - left))

        try:
            plt.pause(0.01)  # type: ignore
        except SystemError:
            return
        if block:
            plt.show()  # type: ignore
