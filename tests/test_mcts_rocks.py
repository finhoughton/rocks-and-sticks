import mcts_ext

import game
from models import PASS, Move


def _to_cpp_move(move):
    cm = mcts_ext.Move()
    cm.x = move.c[0]
    cm.y = move.c[1]
    map_dir = {
        "N": "N",
        "E": "E",
        "NE": "A",
        "SE": "B",
        "NW": "C",
        "SW": "D",
        "W": "W",
        "S": "S",
    }
    if move.t == "P" or move.t == "R":
        cm.t = move.t
    else:
        cm.t = map_dir.get(move.t, "N")
    return cm


def test_rock_placement_and_undo():
    py_game = game.Game()
    cpp_state = mcts_ext.GameState()

    # advance two passes so rocks become available
    py_game.do_move(0, PASS)
    cpp_p0 = mcts_ext.Move()
    cpp_p0.t = 'P'
    cpp_state.do_move(cpp_p0, 0)
    py_game.do_move(1, PASS)
    cpp_p1 = mcts_ext.Move()
    cpp_p1.t = 'P'
    cpp_state.do_move(cpp_p1, 1)

    # find a rock move from Python moves
    player_num = py_game.current_player
    py_moves = list(py_game.get_possible_moves(player_num))
    rock_moves = [m for m in py_moves if m.t == 'R']
    assert rock_moves, "No rock moves available in Python game"
    mv = rock_moves[0]

    before_py = py_game.num_rocks[player_num]
    before_cpp_moves = {(m.x, m.y, m.t) for m in cpp_state.get_possible_moves(player_num)}

    py_game.do_move(player_num, mv)
    cm = _to_cpp_move(mv)
    cpp_state.do_move(cm, player_num)

    assert py_game.rocks
    after_cpp_moves = {(m.x, m.y, m.t) for m in cpp_state.get_possible_moves(player_num)}
    assert (mv.c[0], mv.c[1], 'R') in before_cpp_moves
    assert (mv.c[0], mv.c[1], 'R') not in after_cpp_moves

    py_game.undo_move()
    cpp_state.undo_move()

    assert py_game.num_rocks[player_num] == before_py
    restored_cpp_moves = {(m.x, m.y, m.t) for m in cpp_state.get_possible_moves(player_num)}
    assert (mv.c[0], mv.c[1], 'R') in restored_cpp_moves


def test_rock_must_be_adjacent_to_anchor():
    py_game = game.Game()
    cpp_state = mcts_ext.GameState()

    # advance two passes so rocks become available
    py_game.do_move(0, PASS)
    cpp_p0 = mcts_ext.Move()
    cpp_p0.t = 'P'
    cpp_state.do_move(cpp_p0, 0)
    py_game.do_move(1, PASS)
    cpp_p1 = mcts_ext.Move()
    cpp_p1.t = 'P'
    cpp_state.do_move(cpp_p1, 1)

    player_num = py_game.current_player
    far = Move(100, 100, 'R')

    assert not py_game.valid_move(far, player_num)
    assert far not in list(py_game.get_possible_moves(player_num))

    try:
        py_game.do_move(player_num, far)
        assert False, "Expected Python Game.do_move to reject illegal rock move"
    except ValueError:
        pass

    cm = mcts_ext.Move()
    cm.x = 100
    cm.y = 100
    cm.t = 'R'
    try:
        cpp_state.do_move(cm, player_num)
        assert False, "Expected C++ GameState.do_move to reject illegal rock move"
    except Exception:
        pass
