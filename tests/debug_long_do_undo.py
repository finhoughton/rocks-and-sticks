import random

import players_ext

import game

random.seed(2026)
py_game = game.Game()
cpp_state = players_ext.GameState()

history = []

for i in range(100):
    player = py_game.players[py_game.current_player]
    py_moves = list(py_game.get_possible_moves(player.number))
    if not py_moves:
        break
    move = random.choice(py_moves)

    before_py = len(list(py_game.get_possible_moves(player.number)))
    before_cpp = len(cpp_state.get_possible_moves(py_game.current_player))

    py_game.do_move(player.number, move)
    cm = players_ext.Move()
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
    cm.t = map_dir.get(move.t, move.t)
    cpp_state.do_move(cm, player.number)

    after_py = len(list(py_game.get_possible_moves(py_game.current_player)))
    after_cpp = len(cpp_state.get_possible_moves(py_game.current_player))

    print(f"step {i}: player={player.number} move={move} before_py={before_py} before_cpp={before_cpp} after_py={after_py} after_cpp={after_cpp}")

    history.append((before_py, before_cpp))

print("undoing")

while py_game.moves:
    player = py_game.players[(py_game.current_player - 1) % py_game.num_players]
    last_before_py, last_before_cpp = history.pop()
    py_game.undo_move()
    cpp_state.undo_move()
    restored_py = len(list(py_game.get_possible_moves(player.number)))
    restored_cpp = len(cpp_state.get_possible_moves(player.number))
    print(f"undo player={player.number} restored_py={restored_py} expected_py={last_before_py} restored_cpp={restored_cpp} expected_cpp={last_before_cpp}")

    if restored_py != last_before_py or restored_cpp != last_before_cpp:
        print("Mismatch detected, dumping state")
        print("Py moves:", list(py_game.get_possible_moves(player.number)))
        print("Cpp moves:", cpp_state.get_possible_moves(player.number))
        break
