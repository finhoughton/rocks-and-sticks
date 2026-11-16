from game import Game
from players import RandomPlayer


def test_get_possible_moves_uses_player_index_for_rocks():
    # Deliberately swap player objects so list index != player.number.
    p0 = RandomPlayer(0, seed=1)
    p1 = RandomPlayer(1, seed=2)
    g = Game(players=[p1, p0])

    # Make rocks available in principle.
    g.turn_number = 1

    # Distinct rock stocks by slot index.
    g.num_rocks[0] = 0
    g.num_rocks[1] = 2

    moves_for_0 = list(g.get_possible_moves(0))
    moves_for_1 = list(g.get_possible_moves(1))

    # Player index 0 has zero rocks => no rock moves.
    assert not any(m.t == "R" for m in moves_for_0)
    # Player index 1 has rocks => at least one rock move should exist.
    assert any(m.t == "R" for m in moves_for_1)
