from __future__ import annotations

from collections import deque
from enum import Enum, EnumMeta
from functools import cache, cached_property
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:
    from players import Player


class DirectionMeta(EnumMeta):
    def __iter__(cls) -> Iterator[Direction]:
        v = getattr(cls, "_filtered_members", None)
        if v is None:
            cls._filtered_members = v = tuple(m for m in super().__iter__() if m.value[0] != -1) # type: ignore
        return iter(v) # type: ignore


class Direction(Enum, metaclass=DirectionMeta):
    NO_DIR = -1, (0, 0)
    N = 0, (0, 1)
    E = 1, (1, 0)
    NE = 2, (1, 1)
    SE = 3, (1, -1)
    NW = 4, (-1, 1)
    SW = 5, (-1, -1)
    W = 6, (-1, 0)
    S = 7, (0, -1)

    @cached_property
    def as_int(self) -> int:
        return self.value[0]

    @cached_property
    def delta(self) -> tuple[int, int]:
        return self.value[1]

    @cached_property
    def reversed(self) -> Direction:
        # find the direction whose delta is the negation of this one
        rev_delta = (-self.delta[0], -self.delta[1])
        for m in type(self):
            if m.delta == rev_delta:
                return m
        return Direction.NO_DIR

    @cached_property
    def is_diagonal(self) -> bool:
        return 2 <= self.as_int <= 5

@cache
def delta_to_direction(delta: tuple[int, int]) -> Direction | None:
    for d in Direction:
        if d.delta == delta:
            return d
    return None


D = Direction


class Move:
    __slots__ = ("c", "t")

    def __init__(self, x: int, y: int, t: str):
        self.c = (x, y)
        self.t = t

    def __repr__(self) -> str:
        return f"Move({self.c[0]}, {self.c[1]}, {self.t})"

    def __bool__(self) -> bool:
        return True

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Move):
            return False
        return self.c == other.c and self.t == other.t

    def __deepcopy__(self, memo: dict[int, object]) -> Move:
        return self  # since immutable-ish


PASS = Move(0, 0, "P")


MoveKey = tuple[str, int, int]

def move_key(m: Move) -> MoveKey:
    return (m.t, m.c[0], m.c[1])


class Node:
    __slots__ = (
        "x",
        "y",
        "c",
        "rocked_by",
        "neighbours",
        "empty_directions"
    )

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.c = (x, y)
        self.rocked_by: Player | None = None
        self.neighbours: list[Node | None] = [None for _ in D]
        self.empty_directions: set[D] = set(D)

    @property
    def neighbours_list(self) -> Iterable[Node]:
        return [n for n in self.neighbours if n is not None]

    @property
    def connected(self) -> bool:
        return self.neighbour_count != 0
    
    @property
    def neighbour_count(self) -> int:
        return 8 - len(self.empty_directions)

    def set_neighbour(self, d: D, neighbour: Node) -> None:
        self.neighbours[d.as_int] = neighbour
        self.empty_directions.discard(d)

    def clear_neighbour(self, d: D) -> None:
        self.neighbours[d.as_int] = None
        self.empty_directions.add(d)

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


def calculate_end(p: tuple[int, int], d: D) -> tuple[int, int]:
    x, y = p
    dx, dy = d.delta
    return (x + dx, y + dy)


def calculate_area(ps: Iterable[Node]) -> int:
    ps = iter(ps)
    first = next(ps)
    area = 0

    prev = first
    for current in ps:
        area += prev.x * current.y - current.x * prev.y
        prev = current

    area += prev.x * first.y - first.x * prev.y
    return abs(area)


class Stick:
    __slots__ = ("start", "d", "end", "_ordered_cache")

    def __init__(self, start: Node, end: Node, d: D):
        self.start = start
        self.d = d
        self.end = end
        self._ordered_cache: tuple[tuple[int, int], tuple[int, int]] | None = None

    @property
    def ordered(self) -> tuple[tuple[int, int], tuple[int, int]]:
        cached = self._ordered_cache
        if cached is not None:
            return cached
        a, b = self.start.c, self.end.c
        ordered = (a, b) if a < b else (b, a)
        self._ordered_cache = ordered
        return ordered

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Stick):
            return False
        return self.ordered == value.ordered

    def __hash__(self) -> int:
        return hash(self.ordered)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.start}, {self.end}, D.{self.d.name})"

    def __str__(self) -> str:
        return f"{self.start} -> {self.end}"
