from __future__ import annotations

from game import GameProtocol
from players.game_total import GameTotal
from players.move_utils import to_py_move

try:
    import players_ext
except Exception as e:
    raise ImportError("players_ext extension not available; build with `python setup.py build_ext --inplace`") from e

from models import Move as PyMove

from .ai import AIPlayer


class MCTSPlayerCPP(AIPlayer):
    # Thin wrapper around the C++ `players_ext.MCTSEngine`.

    def __init__(
        self,
        player_number: int,
        seed: int = 0,
        c_puct: float = 1.41421356,
        n_rollouts: int = 1000,
        *,
        verbose: bool = False,
    ) -> None:

        super().__init__(player_number, True)
        self.engine = players_ext.MCTSEngine(seed, c_puct)
        self.engine.set_verbose(bool(verbose))
        self.n_rollouts = int(n_rollouts)

    @property
    def _root_key(self) -> int:
        return int(self.engine.get_current_root_key())

    def get_move(self, game: GameProtocol, reuse_tree: bool = False) -> PyMove:
        assert isinstance(game, GameTotal), "MCTSPlayerCPP requires GameTotal wrapper"
        best_move = self.engine.choose_move(game.cpp, self.n_rollouts)
        py_move = to_py_move(best_move)

        player = game.players[game.current_player]
        if not game.valid_move(py_move, player.number):
            try:
                py_moves = {(m.c[0], m.c[1], m.t) for m in game.get_possible_moves(player.number)}
                cpp_moves = game.cpp.get_possible_moves(player.number)
                cpp_moves_py = set()
                for m in cpp_moves:
                    t = m.t
                    if t in ("P", "R"):
                        cpp_moves_py.add((m.x, m.y, t))
                    else:
                        pm = to_py_move(m)
                        cpp_moves_py.add((pm.c[0], pm.c[1], pm.t))

                only_cpp = list(cpp_moves_py - py_moves)[:20]
                only_py = list(py_moves - cpp_moves_py)[:20]
                raise ValueError(
                    "C++ MCTSPlayerCPP selected illegal move "
                    f"{py_move} for player {player.number}. "
                    f"state_key={int(game.cpp.state_key())} "
                    f"root_key={int(self.engine.get_current_root_key())} "
                    f"py_moves={len(py_moves)} cpp_moves={len(cpp_moves_py)} "
                    f"only_cpp_sample={only_cpp} only_py_sample={only_py}"
                )
            except Exception:
                raise ValueError(f"C++ MCTSPlayerCPP selected illegal move {py_move} for player {player.number}")
        return py_move

    def advance_root(self, move: PyMove, game: GameProtocol) -> None:
        assert isinstance(game, GameTotal), "MCTSPlayerCPP requires GameTotal wrapper"
        self.engine.advance_root(game.cpp)

    def prune_tables(self, max_states: int) -> None:
        self.engine.prune_tables(int(max_states))

    def set_exploration(
        self,
        *,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        temperature: float,
        temperature_moves: int,
    ) -> None:
        self.engine.set_exploration(
            float(dirichlet_alpha),
            float(dirichlet_epsilon),
            float(temperature),
            int(temperature_moves),
        )

    def set_model_checkpoint(self, path: str, device: str = "cpu") -> None:
        self.engine.set_model_checkpoint(str(path), str(device))

    def reset_search(self) -> None:
        self.engine.reset_search()

    def get_root_visit_stats(self, game: GameProtocol) -> list[dict]:
        assert isinstance(game, GameTotal), "MCTSPlayerCPP requires GameTotal wrapper"
        return list(self.engine.get_root_visit_stats(game.cpp))
