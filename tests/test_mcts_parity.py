import random


def test_moves_include_pass(py_game, cpp_state, utils):
    # Use fixtures and helpers from conftest.py
    player_num = py_game.current_player
    pset = {(m.c[0], m.c[1], m.t) for m in py_game.get_possible_moves(player_num)}
    cset = utils["cpp_moves_as_py_tuples"](cpp_state, player_num)

    assert any(t == "P" for (_x, _y, t) in pset)
    assert any(t == "P" for (_x, _y, t) in cset)


def test_do_undo_consistency_random(py_game, cpp_state, utils):
    for _ in range(100):
        player_num = py_game.current_player
        py_moves = list(py_game.get_possible_moves(player_num))
        assert len(py_moves) > 0
        move = random.choice(py_moves)
        print(f"Chosen move: {move}")

        before_py = len(list(py_game.get_possible_moves(player_num)))
        before_cpp = len(cpp_state.get_possible_moves(player_num))

        utils["apply_move_both"](py_game, cpp_state, move)

        py_moves = list(py_game.get_possible_moves(py_game.current_player))
        after_py = len(py_moves)
        cpp_moves = cpp_state.get_possible_moves(py_game.current_player)
        print(f"CPP:  {cpp_moves}")
        print(f"PY:   {py_moves}")
        after_cpp = len(cpp_moves)

        assert isinstance(after_py, int)
        assert isinstance(after_cpp, int)

        py_game.undo_move()
        cpp_state.undo_move()

        restored_py = len(list(py_game.get_possible_moves(player_num)))
        restored_cpp = len(cpp_state.get_possible_moves(player_num))

        assert restored_py == before_py
        assert restored_cpp == before_cpp
        assert after_py == after_cpp
    