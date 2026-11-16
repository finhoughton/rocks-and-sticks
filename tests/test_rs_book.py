"""Opening book keys and mirroring."""

from __future__ import annotations

from game import Game
from models import Move
from players.base import RandomPlayer
import json

from players.rs_book import SYMMETRIES, Book, mirror_move, position_key, to_move, transform_move


def _game(moves) -> Game:
    g = Game([RandomPlayer(0), RandomPlayer(1)])
    for m in moves:
        g.do_move(g.current_player, to_move(m))
    return g


def test_mirror_images_share_a_key_and_moves_are_mirrored_back(tmp_path) -> None:
    line = [(0, 1, "NE"), (1, 2, "E"), (0, 0, "R"), (2, 2, "R")]
    mirrored = [mirror_move(m) for m in line]
    a, b = _game(line), _game(mirrored)
    ka, fa = position_key(a)
    kb, fb = position_key(b)
    assert ka == kb and fa[0] != fb[0]

    book = Book(str(tmp_path / "book.json"))
    book.store(a, (1, 2, "SE"), 123, 17)
    book.save()
    book = Book(str(tmp_path / "book.json"))
    assert book.lookup(a)["move"] == (1, 2, "SE")
    assert book.lookup(b)["move"] == (-1, 2, "SW")
    for g in (a, b):
        assert g.valid_move(to_move(book.lookup(g)["move"]), g.current_player)


def test_move_order_does_not_matter_but_side_to_move_does() -> None:
    a = _game([(0, 1, "E"), (0, 0, "W"), (1, 1, "R"), (-1, 0, "R")])
    b = _game([(0, 1, "E"), (0, 0, "W"), (1, 1, "R")])
    assert position_key(a)[0] != position_key(b)[0]
    c = _game([(0, 0, "W"), (0, 1, "E")])
    d = _game([(0, 1, "E"), (0, 0, "W")])
    assert position_key(c)[0] == position_key(d)[0]


def test_symmetric_position_is_its_own_mirror() -> None:
    g = _game([])
    assert position_key(g) == (position_key(g)[0], (False, False))
    assert Move(0, 1, "N") == to_move(mirror_move((0, 1, "N")))


def test_all_four_images_share_a_key_and_moves_come_back_oriented(tmp_path) -> None:
    line = [(0, 1, "NE"), (1, 2, "E"), (0, 0, "R"), (2, 2, "R")]
    games = [_game([transform_move(m, sym) for m in line]) for sym in SYMMETRIES]
    assert len({position_key(g)[0] for g in games}) == 1
    book = Book(str(tmp_path / "book.json"))
    book.store(games[0], (1, 2, "SE"), 50, 18)
    for sym, g in zip(SYMMETRIES, games):
        assert book.lookup(g)["move"] == transform_move((1, 2, "SE"), sym)
        assert g.valid_move(to_move(book.lookup(g)["move"]), g.current_player)


def test_old_format_books_are_looked_up_with_the_left_right_mirror_only(tmp_path) -> None:
    line = [(0, 1, "NE"), (1, 2, "E")]
    g = _game(line)
    key, _ = position_key(g, SYMMETRIES[:2])
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"version": 1, "entries": {key: {"m": [1, 2, "SE"], "s": 0, "d": 9}}}))
    book = Book(str(path))
    assert book.lookup(g) is not None
    assert book.lookup(_game([transform_move(m, (True, False)) for m in line])) is not None
    assert book.lookup(_game([transform_move(m, (False, True)) for m in line])) is None
