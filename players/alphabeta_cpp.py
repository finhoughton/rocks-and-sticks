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
        pass_penalty: float = 3.0,
        use_heuristic: bool = False,
        time_limit_ms: int = 0,
        max_depth: int = 20,
        native_model: str = "",
        nn_ordering_depth: int = 2,
    ) -> None:
        super().__init__(player_number, not use_heuristic)
        self.engine = players_ext.AlphaBetaEngine(int(seed), float(pass_penalty))
        self.depth = int(depth)
        self.move_cap = int(move_cap)
        self.time_limit_ms = int(time_limit_ms)
        self.max_depth = int(max_depth)
        if use_heuristic:
            self.engine.set_use_heuristic(True)
        if native_model:
            if not self.engine.load_native_model(str(native_model)):
                raise RuntimeError(f"Failed to load native NN weights: {native_model}")
        if nn_ordering_depth != 2:
            self.engine.set_nn_ordering_depth(int(nn_ordering_depth))

    def get_move(self, game: GameProtocol) -> PyMove:
        assert isinstance(game, GameTotal), "AlphaBetaPlayerCPP requires GameTotal wrapper"
        if self.time_limit_ms > 0:
            best_move = self.engine.choose_move_iterative(
                game.cpp, int(self.max_depth), int(self.time_limit_ms), int(self.move_cap)
            )
        else:
            best_move = self.engine.choose_move(game.cpp, int(self.depth), int(self.move_cap))
        return to_py_move(best_move)

    def set_model_checkpoint(self, path: str, device: str = "cpu") -> None:
        self.engine.set_model_checkpoint(str(path), str(device))

    def load_native_model(self, path: str) -> None:
        """Load native C++ NN weights for GIL-free inference."""
        if not self.engine.load_native_model(str(path)):
            raise RuntimeError(f"Failed to load native NN weights: {path}")

    def set_nn_ordering_depth(self, min_depth: int) -> None:
        """Set minimum AB depth at which NN move ordering is used."""
        self.engine.set_nn_ordering_depth(int(min_depth))

    def clear_stats(self) -> None:
        self.engine.clear_stats()

    def get_profile_stats(self) -> dict:
        return dict(self.engine.get_profile_stats())
