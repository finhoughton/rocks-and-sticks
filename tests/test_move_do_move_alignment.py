import random

import players_ext

from gnn.game_generation import randomize_start
from players.game_total import GameTotal


def test_is_move_legal_implies_do_move():
    for _ in range(200):
        py_game = __import__('game').Game()
        cpp_state = players_ext.GameState()
        total = GameTotal(py_game, cpp_state)
        move_log = []
        randomize_start(total, 50, 20, move_log=move_log)

        player_num = py_game.current_player

        cpp_moves = list(cpp_state.get_possible_moves(player_num))
        assert len(cpp_moves) > 0
        for m in cpp_moves:
            assert bool(cpp_state.is_move_legal(m, player_num))
            ok = bool(cpp_state.can_apply_move(m, player_num))
            if not ok:
                print("DEBUG: candidate move failed can_apply_move:", (m.x, m.y, m.t))
            assert ok, f"Move {m.x},{m.y},{m.t} reported legal but cannot be applied"

