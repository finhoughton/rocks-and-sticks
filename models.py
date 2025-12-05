from __future__ import annotations

from collections import deque
from enum import Enum
from functools import cache, cached_property
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from players import Player


class Direction(Enum):
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
    def reversed(self) -> D:
        return list(D)[7 - self.as_int]
    
    @cached_property
    def is_diagonal(self) -> bool:
        return 2 <= self.as_int <= 5

D = Direction

class Move:
    @cache
    def __new__(cls, x: int, y: int, t: str) -> Move:  # type: ignore[override]
        return super().__new__(cls)

    def __init__(self, x: int, y: int, t: str):
        self.c = (x, y)
        self.t = t

    def __repr__(self) -> str:
        return f"Move({self.c[0]}, {self.c[1]}, {self.t})"

    def __bool__(self) -> bool:
        return True


PASS = Move(0, 0, "P")


class Node:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        self.c = (x, y)
        self.rocked_by: Player | None = None
        self.neighbours: list[Node | None] = [None for _ in D]
        self.empty_directions: set[D] = set(D)
        self.neighbour_count: int = 0

    @property
    def neighbours_list(self) -> Iterable[Node]:
        return [n for n in self.neighbours if n is not None]

    @property
    def connected(self) -> bool:
        return self.neighbour_count > 0

    def set_neighbour(self, d: D, neighbour: Node) -> None:
        self.neighbours[d.as_int] = neighbour
        self.empty_directions.discard(d)
        self.neighbour_count += 1

    def clear_neighbour(self, d: D) -> None:
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
    def __init__(self, start: Node, end: Node, d: D):
        self.start = start
        self.d = d
        self.end = end

    @cached_property
    def ordered(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return tuple(sorted((self.start.c, self.end.c))) # type: ignore

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
