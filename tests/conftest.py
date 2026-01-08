import mcts_ext
import pytest

import game
import models
from players.move_utils import (
    CPP_TO_PY_DIR,
    PY_TO_CPP_DIR,
    to_cpp_move,
    to_py_move,
)


def py_moves_as_cpp_tuples(py_game: game.Game):
    player_num = py_game.current_player
    out = set()
    for m in py_game.get_possible_moves(player_num):
        if m.t == "P" or m.t == "R":
            out.add((m.c[0], m.c[1], m.t))
        else:
            out.add((m.c[0], m.c[1], PY_TO_CPP_DIR[m.t]))
    return out


def cpp_moves_as_py_tuples(cpp_state: mcts_ext.GameState, player_number: int):
    out = set()
    for m in cpp_state.get_possible_moves(player_number):
        if m.t == "P" or m.t == "R":
            out.add((m.x, m.y, m.t))
        else:
            out.add((m.x, m.y, CPP_TO_PY_DIR[m.t]))
    return out


def apply_move_both(py_game: game.Game, cpp_state: mcts_ext.GameState, py_move: models.Move):
    player_num = py_game.current_player
    py_game.do_move(player_num, py_move)
    cpp_state.do_move(to_cpp_move(py_move), player_num)


@pytest.fixture
def py_game():
    return game.Game()


@pytest.fixture
def cpp_state():
    return mcts_ext.GameState()


@pytest.fixture
def utils():
    return {
        "to_cpp_move": to_cpp_move,
        "to_py_move": to_py_move,
        "py_moves_as_cpp_tuples": py_moves_as_cpp_tuples,
        "cpp_moves_as_py_tuples": cpp_moves_as_py_tuples,
        "apply_move_both": apply_move_both,
    }
