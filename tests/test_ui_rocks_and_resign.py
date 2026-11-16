"""The game window shows each player's rocks left and lets a human resign."""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from game import Game  # noqa: E402
from models import Move  # noqa: E402
from players.base import HumanPlayer, RandomPlayer, Resigned  # noqa: E402


def _rendered(game: Game) -> Game:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the Agg canvas cannot be shown
        game.render(block=False)
    return game


def test_rocks_left_are_shown_for_both_players_and_follow_the_game() -> None:
    g = Game([HumanPlayer(0), RandomPlayer(1)])
    texts = lambda: [t.get_text() for t in _rendered(g)._render_rock_texts]  # noqa: E731
    assert texts() == ["Player 1 (you) rocks  ●●", "Player 2 rocks  ●●"]
    for m in [Move(0, 0, "S"), Move(0, 1, "NE"), Move(0, -1, "R")]:
        g.do_move(g.current_player, m)
    assert texts()[0].endswith("●○")
    g.do_move(1, Move(1, 2, "R"))
    g.do_move(0, Move(0, 0, "R"))
    assert texts() == ["Player 1 (you) rocks  ○○", "Player 2 rocks  ●○"]
    g.do_move(1, Move(1, 2, "E"))  # a stick refills
    assert texts()[1].endswith("●●")


def test_resign_button_needs_two_clicks_and_typing_resign_also_works() -> None:
    g = _rendered(Game([HumanPlayer(0), RandomPlayer(1)]))
    button = g._render_resign_button
    button._observers.process("clicked", None)
    assert g._pending_input_text is None and "again" in button.label.get_text()
    button._observers.process("clicked", None)
    assert g._pending_input_text == "resign" and button.label.get_text() == "Resign"

    g.wait_for_move_input = lambda prompt: " Resign "  # type: ignore[method-assign]
    with pytest.raises(Resigned) as info:
        g.players[0].get_move(g)
    assert info.value.player == 0


def test_a_resigned_game_says_so_in_the_title() -> None:
    g = Game([HumanPlayer(0), RandomPlayer(1)])
    g.winner, g.result_note = 1, "Player 1 resigned"
    assert _rendered(g)._render_ax.get_title().endswith("Player 2 wins (Player 1 resigned)")
