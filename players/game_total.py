from typing import Iterator, Optional

import players_ext

from game import Edge, Game, GameProtocol
from models import D, Move, Node, Stick
from players.base import Player
from players.move_utils import to_cpp_move


class GameTotal(GameProtocol):
    players: list[Player]
    current_player: int
    turn_number: int
    winner: Optional[int]
    players_scores: list[int]
    num_rocks: list[int]
    num_players: int
    moves: list[Move]
    sticks: list[Stick]
    rocks: list[Node]
    points: dict[tuple[int, int], Node]
    connected_points: set[Node]

    _DELEGATED_FIELDS = {
        "players",
        "current_player",
        "turn_number",
        "winner",
        "players_scores",
        "num_rocks",
        "num_players",
        "moves",
        "sticks",
        "rocks",
        "points",
        "connected_points",
    }

    def __init__(self, py: Game, cpp: players_ext.GameState) -> None:
        object.__setattr__(self, "py", py)
        object.__setattr__(self, "cpp", cpp)

    def __getattr__(self, name: str):
        if name in self._DELEGATED_FIELDS:
            return getattr(self.py, name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value) -> None:
        if name in ("py", "cpp"):
            object.__setattr__(self, name, value)
            return
        if name in self._DELEGATED_FIELDS:
            setattr(self.py, name, value)
            return
        object.__setattr__(self, name, value)
    
    def do_move(self, player_number: int, m: Move) -> None:
        assert isinstance(player_number, int)
        self.py.do_move(player_number, m)
        self.cpp.do_move(to_cpp_move(m), player_number)
        assert self.py.current_player == self.cpp.current_player

    def undo_move(self) -> None:
        self.py.undo_move()
        self.cpp.undo_move()
        assert self.py.current_player == self.cpp.current_player

    def get_possible_moves(self, player_number: int) -> Iterator[Move]:
        return self.py.get_possible_moves(player_number)

    def set_current_player0(self) -> None:
        self.py.set_current_player0()
        self.cpp.set_current_player0()
        assert self.py.current_player == self.cpp.current_player

    def valid_move(self, m: Move, player_number: int) -> bool:
        return self.py.valid_move(m, player_number)
    
    def wait_for_move_input(self, prompt: str) -> str:
        return self.py.wait_for_move_input(prompt)
    
    def render(self, block: bool = False, game_index: Optional[int] = None, total_games: Optional[int] = None) -> None:
        return self.py.render(block, game_index, total_games)
    
    def intersects_stick(self, start: tuple[int, int], d: D) -> bool:
        return self.py.intersects_stick(start, d)

    def coord_in_claimed_region(self, c: tuple[int, int]) -> bool:
        return self.py.coord_in_claimed_region(c)

    def _path_of_smallest_area(self, start: Node, end: Node) -> tuple[int, tuple[tuple[int, int], ...], frozenset[Edge]] | None:
        return self.py._path_of_smallest_area(start, end)
