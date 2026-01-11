import players_ext

from models import PASS, Move

PY_TO_CPP_DIR = {
    "N": "N",
    "E": "E",
    "NE": "A",
    "SE": "B",
    "NW": "C",
    "SW": "D",
    "W": "W",
    "S": "S",
}
CPP_TO_PY_DIR = {v: k for k, v in PY_TO_CPP_DIR.items()}


def to_cpp_move(py_move: Move) -> players_ext.Move:
    cm = players_ext.Move()
    cm.x = py_move.c[0]
    cm.y = py_move.c[1]
    t = py_move.t
    if t == "P" or t == "R":
        cm.t = t
    else:
        cm.t = PY_TO_CPP_DIR[t]
    return cm


def to_py_move(cpp_move: players_ext.Move) -> Move:
    t = cpp_move.t
    if t == "R":
        py_t = t
    elif t == "P":
        return PASS
    else:
        py_t = CPP_TO_PY_DIR[t]
    return Move(cpp_move.x, cpp_move.y, py_t)
