"""The readable evaluation derived from the rs engine's raw score."""

from __future__ import annotations

from players.rs_display import MATE, describe, final


def test_sign_convention_is_player_one_positive() -> None:
    assert describe(400, 0).pawns > 0 and describe(400, 0).p1_chance > 0.5
    assert describe(400, 1).pawns < 0 and describe(400, 1).p1_chance < 0.5
    assert describe(-400, 1).pawns > 0
    # the same raw score for either mover describes mirror-image chances (up to the small move bonus)
    assert abs(describe(300, 0).p1_chance + describe(300, 1).p1_chance - 1.0) < 0.05


def test_forced_wins_count_the_winners_own_moves() -> None:
    # mover wins with the move it is about to play
    e = describe(MATE - 1, 1)
    assert (e.winner, e.moves_to_win, e.short) == (1, 1, "-M1")
    # player 1 scores now and player 2 cannot answer: two plies, one move by the winner
    e = describe(MATE - 2, 0)
    assert (e.winner, e.moves_to_win, e.short) == (0, 1, "+M1")
    # mover loses in 3 plies: the opponent plays one move in between
    e = describe(-(MATE - 3), 0)
    assert (e.winner, e.moves_to_win) == (1, 1)
    e = describe(MATE - 7, 1)
    assert (e.winner, e.moves_to_win, e.text) == (1, 4, "Player 2 wins in 4")


def test_final() -> None:
    assert final(0).p1_chance == 1.0 and final(1).p1_chance == 0.0 and final(None).p1_chance == 0.5
