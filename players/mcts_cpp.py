from __future__ import annotations

from typing import TYPE_CHECKING

from game import GameProtocol
from players.game_total import GameTotal
from players.move_utils import to_py_move

try:
    import mcts_ext
except Exception as e:
    raise ImportError("mcts_ext extension not available; build with `python setup.py build_ext --inplace`") from e

from models import PASS, move_key
from models import Move as PyMove

from .ai import AIPlayer

if TYPE_CHECKING:
    from game import GameProtocol as Game


def _py_move_to_cpp(mv: PyMove) -> mcts_ext.Move:
    cm = mcts_ext.Move()
    cm.x = int(mv.c[0])
    cm.y = int(mv.c[1])
    cm.t = mv.t
    return cm


def _py_game_to_cpp(game: Game) -> mcts_ext.GameState:
    gs = mcts_ext.GameState()
    player = 0
    for mv in game.moves:
        gs.do_move(_py_move_to_cpp(mv), player)
        player = (player + 1) % game.num_players
    return gs

class MCTSPlayerCPP(AIPlayer):
    # Thin wrapper around the C++ `mcts_ext.MCTSEngine`.

    def __init__(
        self,
        player_number: int,
        seed: int = 0,
        c_puct: float = 1.41421356,
        n_rollouts: int = 1000,
    ) -> None:

        # GNN evaluation is mandatory for the C++ MCTS backend.
        # Ensure `gnn.model.load_model(...)` has been called before using this.
        super().__init__(player_number, True)
        self.engine = mcts_ext.MCTSEngine(seed, c_puct)
        self.n_rollouts = int(n_rollouts)

    @property
    def _root_key(self) -> int:
        return int(self.engine.get_current_root_key())

    def get_move(self, game: GameProtocol, reuse_tree: bool = False) -> PyMove:
        assert isinstance(game, GameTotal), "MCTSPlayerCPP requires GameTotal wrapper"
        best_move = self.engine.choose_move(game.cpp, self.n_rollouts)
        py_move = to_py_move(best_move)

        # Safety net: if the C++ backend ever disagrees with Python legality (e.g.
        # due to subtle claimed-region/intersection differences), fall back to a
        # deterministic Python-legal move.
        player = game.py.players[game.py.current_player]
        if not game.py.valid_move(py_move, player.number):
            legal_moves = sorted(list(game.py.get_possible_moves(player.number)), key=move_key)
            non_pass = [m for m in legal_moves if m is not PASS]
            return non_pass[0] if non_pass else PASS

        return py_move

    def advance_root(self, move: PyMove, game: GameProtocol) -> None:
        assert isinstance(game, GameTotal), "MCTSPlayerCPP requires GameTotal wrapper"
        self.engine.advance_root(game.cpp)

    def prune_tables(self, max_states: int) -> None:
        self.engine.prune_tables(int(max_states))

    def get_root_visit_stats(self, game: GameProtocol) -> list[dict]:
        assert isinstance(game, GameTotal), "MCTSPlayerCPP requires GameTotal wrapper"
        return list(self.engine.get_root_visit_stats(game.cpp))
