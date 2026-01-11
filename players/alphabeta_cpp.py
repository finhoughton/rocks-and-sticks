from __future__ import annotations

from typing import TYPE_CHECKING

from game import GameProtocol
from players.game_total import GameTotal
from players.move_utils import to_py_move

try:
    import players_ext
except Exception as e:
    raise ImportError("players_ext extension not available; build with `python setup.py build_ext --inplace`") from e

from models import Move as PyMove

from .ai import AIPlayer

if TYPE_CHECKING:
    from game import GameProtocol


class AlphaBetaPlayerCPP(AIPlayer):
    def __init__(
        self,
        player_number: int,
        *,
        seed: int = 0,
        depth: int = 3,
        move_cap: int = 48,
        pass_penalty: float = 1.2,
    ) -> None:
        super().__init__(player_number, True)
        self.engine = players_ext.AlphaBetaEngine(int(seed), float(pass_penalty))
        self.depth = int(depth)
        self.move_cap = int(move_cap)

    def get_move(self, game: GameProtocol) -> PyMove:
        assert isinstance(game, GameTotal), "AlphaBetaPlayerCPP requires GameTotal wrapper"
        best_move = self.engine.choose_move(game.cpp, int(self.depth), int(self.move_cap))
        return to_py_move(best_move)

    def set_model_checkpoint(self, path: str, device: str = "cpu") -> None:
        self.engine.set_model_checkpoint(str(path), str(device))

    def clear_stats(self) -> None:
        self.engine.clear_stats()

    def get_profile_stats(self) -> dict:
        return dict(self.engine.get_profile_stats())
