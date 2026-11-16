import players_ext

import game
from models import PASS


def _first_move_of_type(moves, move_type: str):
    for m in moves:
        if m.t == move_type:
            return m
    return None


def _first_cpp_stick(moves):
    for m in moves:
        if m.t not in {'P', 'R'}:
            return m
    return None


def _has_cpp_rock_move(cpp, player_number: int) -> bool:
    return any(m.t == 'R' for m in cpp.get_possible_moves(player_number))


def test_pass_does_not_refill_rocks_python_and_cpp():
    py_game = game.Game()
    cpp = players_ext.GameState()

    # P0: make a stick move so game can progress and rocks become available later.
    p0_stick = next(m for m in py_game.get_possible_moves(0) if m.t not in {'P', 'R'})
    py_game.do_move(0, p0_stick)

    cm = players_ext.Move()
    cm.x = p0_stick.c[0]
    cm.y = p0_stick.c[1]
    cm.t = {'NE': 'A', 'SE': 'B', 'NW': 'C', 'SW': 'D'}.get(p0_stick.t, p0_stick.t)
    cpp.do_move(cm, 0)

    # P1: pass
    py_game.do_move(1, PASS)
    cpp_p = players_ext.Move()
    cpp_p.t = 'P'
    cpp.do_move(cpp_p, 1)

    # P0: place two rocks so stock reaches zero.
    p0 = py_game.current_player
    py_rock = _first_move_of_type(py_game.get_possible_moves(p0), 'R')
    assert py_rock is not None
    py_game.do_move(p0, py_rock)
    assert py_game.num_rocks[p0] == 1

    cpp_moves = list(cpp.get_possible_moves(p0))
    cpp_rock = _first_move_of_type(cpp_moves, 'R')
    assert cpp_rock is not None
    cpp.do_move(cpp_rock, p0)
    assert _has_cpp_rock_move(cpp, p0)

    # Return to P0 and place second rock.
    py_game.do_move(1 - p0, PASS)
    cpp.do_move(cpp_p, 1 - p0)
    py_rock2 = _first_move_of_type(py_game.get_possible_moves(p0), 'R')
    assert py_rock2 is not None
    py_game.do_move(p0, py_rock2)

    cpp_rock2 = _first_move_of_type(cpp.get_possible_moves(p0), 'R')
    assert cpp_rock2 is not None
    cpp.do_move(cpp_rock2, p0)

    assert py_game.num_rocks[p0] == 0
    assert not _has_cpp_rock_move(cpp, p0)

    # P1 pass, P0 pass, P1 pass -> P0 to move again.
    py_game.do_move(1 - p0, PASS)
    cpp.do_move(cpp_p, 1 - p0)
    py_game.do_move(p0, PASS)
    cpp.do_move(cpp_p, p0)
    py_game.do_move(1 - p0, PASS)
    cpp.do_move(cpp_p, 1 - p0)

    assert py_game.num_rocks[p0] == 0
    assert not _has_cpp_rock_move(cpp, p0)


def test_stick_refills_rocks_python_and_cpp():
    py_game = game.Game()
    cpp = players_ext.GameState()

    # P0 stick
    p0_stick = next(m for m in py_game.get_possible_moves(0) if m.t not in {'P', 'R'})
    py_game.do_move(0, p0_stick)
    cm = players_ext.Move()
    cm.x = p0_stick.c[0]
    cm.y = p0_stick.c[1]
    cm.t = {'NE': 'A', 'SE': 'B', 'NW': 'C', 'SW': 'D'}.get(p0_stick.t, p0_stick.t)
    cpp.do_move(cm, 0)

    # P1 pass
    py_game.do_move(1, PASS)
    cpp_p = players_ext.Move()
    cpp_p.t = 'P'
    cpp.do_move(cpp_p, 1)

    # P0 uses both rocks -> reaches zero.
    p0 = py_game.current_player
    py_rock = _first_move_of_type(py_game.get_possible_moves(p0), 'R')
    assert py_rock is not None
    py_game.do_move(p0, py_rock)
    cpp_rock = _first_move_of_type(cpp.get_possible_moves(p0), 'R')
    assert cpp_rock is not None
    cpp.do_move(cpp_rock, p0)
    assert py_game.num_rocks[p0] == 1
    assert _has_cpp_rock_move(cpp, p0)

    py_game.do_move(1 - p0, PASS)
    cpp.do_move(cpp_p, 1 - p0)
    py_rock2 = _first_move_of_type(py_game.get_possible_moves(p0), 'R')
    assert py_rock2 is not None
    py_game.do_move(p0, py_rock2)
    cpp_rock2 = _first_move_of_type(cpp.get_possible_moves(p0), 'R')
    assert cpp_rock2 is not None
    cpp.do_move(cpp_rock2, p0)
    assert py_game.num_rocks[p0] == 0
    assert not _has_cpp_rock_move(cpp, p0)

    # P1 pass, then P0 stick -> refill to 2 (at least one rock move should exist later).
    py_game.do_move(1 - p0, PASS)
    cpp.do_move(cpp_p, 1 - p0)

    py_stick = next(m for m in py_game.get_possible_moves(p0) if m.t not in {'P', 'R'})
    py_game.do_move(p0, py_stick)

    cpp_stick = _first_cpp_stick(cpp.get_possible_moves(p0))
    assert cpp_stick is not None
    cpp.do_move(cpp_stick, p0)

    assert py_game.num_rocks[p0] == 2

    # Give turn back to p0 and verify rock moves are available again.
    py_game.do_move(1 - p0, PASS)
    cpp.do_move(cpp_p, 1 - p0)
    assert _first_move_of_type(py_game.get_possible_moves(p0), 'R') is not None
    assert _has_cpp_rock_move(cpp, p0)