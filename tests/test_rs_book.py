"""Opening book keys and mirroring."""

from __future__ import annotations

from game import Game
from models import Move
from players.base import RandomPlayer
import json

import pytest

from players.rs_book import IDENTITY, SYMMETRIES, Book, inverse, mirror_move, position_key, to_move, transform_move


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
    assert ka == kb and fa != fb

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
    assert position_key(g) == (position_key(g)[0], IDENTITY)
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


def test_turned_and_shifted_images_share_a_key_and_moves_come_back_oriented(tmp_path) -> None:
    # a diagonal mirror (the starting stick and the first move change places) and a shift
    for sym, a_line, b_line in [
        ((True, False, False, -1, 0), [(0, 1, "E")], [(0, 0, "W")]),
        ((False, False, False, 0, -1), [(0, 1, "N")], [(0, 0, "S")]),
    ]:
        rest = [(0, 1, "NW"), (0, 1, "R"), (0, 0, "R")]
        a = _game(a_line + rest)
        b = _game(b_line + [transform_move(m, sym) for m in rest])
        assert position_key(a)[0] == position_key(b)[0]
        book = Book(str(tmp_path / "book.json"))
        book.store(a, (-1, 2, "N"), 50, 18)
        assert book.lookup(a)["move"] == (-1, 2, "N")
        assert book.lookup(b)["move"] == transform_move((-1, 2, "N"), sym)
        for g in (a, b):
            assert g.valid_move(to_move(book.lookup(g)["move"]), g.current_player)
        assert transform_move(transform_move((3, -2, "NE"), sym), inverse(sym)) == (3, -2, "NE")


def test_old_format_books_are_looked_up_with_the_symmetries_they_were_written_with(tmp_path) -> None:
    line = [(0, 1, "NE"), (1, 2, "E")]
    g = _game(line)
    left_right, top_bottom = SYMMETRIES[1], SYMMETRIES[2]
    shifted = _game([(0, 0, "S"), (0, 1, "N")])  # the same position as 0,1N 0,2N one step lower
    for version, symmetries, knows_top_bottom in ((1, SYMMETRIES[:2], False), (2, SYMMETRIES, True)):
        path = tmp_path / f"old{version}.json"
        entries = {}
        for game in (g, _game([(0, 1, "N"), (0, 2, "N")])):
            key, sym = position_key(game, symmetries)
            entries[key] = {"m": list(transform_move((1, 2, "SE"), sym)), "s": 0, "d": 9}
        path.write_text(json.dumps({"version": version, "entries": entries}))
        book = Book(str(path))
        assert book.lookup(g)["move"] == (1, 2, "SE")
        assert book.lookup(_game([transform_move(m, left_right) for m in line]))["move"] == (-1, 2, "SW")
        assert (book.lookup(_game([transform_move(m, top_bottom) for m in line])) is not None) == knows_top_bottom
        assert book.lookup(shifted) is None
        with pytest.raises(ValueError):
            book.store(g, (1, 2, "SE"), 0, 9)


def _builder(tmp_path):
    import argparse
    import importlib.util
    import os

    pytest.importorskip("rs_engine")
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "build_book.py")
    spec = importlib.util.spec_from_file_location("build_book", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    args = argparse.Namespace(
        tt_bits=16, threads=1, book=str(tmp_path / "book.json"), work=str(tmp_path / "work.json"), fresh=True, node_minutes=0.0005,
        screen_seconds=0.01, replies=3, min_share=0.04, temperature=150.0, max_ply=12, live_mate_plies=11, backup_margin=100, verify_plies=4, verify_seconds=0.01, max_tries=3,
        variety_ply=1, variety_moves=3, variety_margin=60, lost=400,
    )
    return mod.Builder(args)


def _after(game, move):
    """The key of the position a move leads to."""
    game.do_move(game.current_player, to_move(move))
    key = position_key(game)[0]
    game.undo_move()
    return key


def test_the_builder_replaces_a_book_move_that_led_somewhere_bad(tmp_path) -> None:
    b = _builder(tmp_path)
    start = _game([])
    b.book.store(start, (0, 0, "S"), 500, 19)  # the book thinks this move is winning
    line = [(0, 0, "S"), (0, -1, "S")]         # ... but two moves later the position is bad
    b.backup(0.3, line, -400)
    entry = b.book.lookup(start)
    bad = b.refuted[position_key(start)[0]]
    assert _after(start, (0, 0, "S")) in bad and entry["score"] < 400
    # the new entry is a move not shown to be bad (its mirror image 0,1N counts as the same
    # move), or, if nothing else looked better, the least bad one at what it was shown to be worth
    led_to = _after(start, entry["move"])
    assert led_to not in bad or entry["score"] == max(bad.values())
    assert _after(start, (0, 1, "N")) == _after(start, (0, 0, "S"))
    saved = json.loads((tmp_path / "work.json").read_text())
    assert saved["refuted"] == b.refuted


def test_the_builder_leaves_a_book_move_alone_when_it_held_up(tmp_path) -> None:
    b = _builder(tmp_path)
    start = _game([])
    b.book.store(start, (0, 0, "S"), 20, 19)
    b.backup(0.3, [(0, 0, "S"), (0, -1, "S")], -30)   # within the margin
    b.backup(0.3, [(0, 0, "E"), (0, -1, "S")], -900)  # not the book's move: nothing to correct
    assert b.book.lookup(start) == {"move": (0, 0, "S"), "score": 20, "depth": 19}
    assert b.refuted == {}


def test_the_builder_raises_the_opponents_entry_when_their_move_proves_strong(tmp_path) -> None:
    b = _builder(tmp_path)
    after = _game([(0, 0, "S")])
    b.book.store(after, (0, -1, "S"), -300, 19)  # player 2's book move, thought to be poor
    b.backup(0.3, [(0, 0, "S"), (0, -1, "S")], -500)  # ... leaves player 1 in a bad position
    assert b.book.lookup(after) == {"move": (0, -1, "S"), "score": 500, "depth": 19}


def test_the_builder_plays_a_new_position_on_and_rechecks_it_if_it_falls_apart(tmp_path) -> None:
    b = _builder(tmp_path)
    start = _game([])
    b.book.store(start, (0, 0, "S"), 5000, 19)  # promises far more than playing on will show
    # the book also has the reply: playing on must follow it, as a book-using bot would
    after = _game([(0, 0, "S")])
    b.book.store(after, (0, -1, "S"), 0, 19)
    entry = b.settle(start, [], b.book.lookup(start))
    assert entry == b.book.lookup(start) and entry["score"] < 4000
    assert b.after(start, (0, 0, "S")) in b.refuted[position_key(start)[0]]
    held = {"move": entry["move"], "score": -5000, "depth": 9}  # promises little: nothing to recheck
    assert b.settle(start, [], held) == held


def test_walking_back_carries_the_result_up_the_line_without_forgetting(tmp_path) -> None:
    b = _builder(tmp_path)
    start, after = _game([]), _game([(0, 0, "S")])
    line = [([(0, 0, "S")], True), ([(0, 0, "S"), (0, -1, "S")], True)]
    two = _game(line[1][0])
    # the line came to -579 for player 1. Player 2's entry already says so and is left alone
    b.book.store(two, (0, -1, "R"), -579, 19)
    b.book.store(after, (0, -1, "S"), 600, 19)
    b.walk_back(0, line, -579, 0)
    assert b.book.lookup(after) == {"move": (0, -1, "S"), "score": 600, "depth": 19}
    # an entry that expected less of its move is raised, with no new search to contradict it
    b.book.store(after, (0, -1, "S"), 20, 19)
    b.walk_back(0, line, -579, 0)
    assert b.book.lookup(after) == {"move": (0, -1, "S"), "score": 579, "depth": 19}
    # an entry that expected more has its move marked as bad and replaced, or kept at its worth
    b.book.store(two, (0, -1, "R"), 300, 19)
    b.walk_back(0, line, -579, 0)
    assert b.refuted[position_key(two)[0]] == {b.after(two, (0, -1, "R")): -579}
    assert b.book.lookup(two)["score"] < 300
    # a position with moves shown to be bad keeps the least bad of them if nothing else is better
    b.refute(start, (0, 0, "S"), 29000)
    b.book.store(start, (0, 0, "S"), 0, 19)
    b.research(start)
    entry = b.book.lookup(start)
    assert entry["score"] == 29000 and b.after(start, entry["move"]) == b.after(start, (0, 0, "S"))


def test_the_builder_follows_several_first_moves_but_nothing_after_a_losing_one(tmp_path) -> None:
    b = _builder(tmp_path)
    b.queue = []
    b.recheck = None
    start = _game([])
    b.refute(start, (0, 0, "S"), -579)  # shown to lose
    b.expand(1.0, [])
    firsts = [q[3][0] for q in b.queue if len(q[3]) == 1]
    assert 2 <= len(firsts) <= 3 and len({b.after(start, m) for m in firsts}) == len(firsts)
    assert b.after(start, (0, 0, "S")) not in {b.after(start, m) for m in firsts}
    assert b.lost_line([(0, 1, "N"), (0, 2, "N")]) and not b.lost_line([(0, 0, "W"), (0, 1, "N")])
    before = len(b.book)
    b.expand(0.5, [(0, 1, "N"), (0, 2, "N")])  # the losing move in its mirror image
    assert len(b.book) == before


def test_the_engine_can_be_told_which_root_moves_to_skip() -> None:
    rs_engine = pytest.importorskip("rs_engine")
    eng = rs_engine.Engine(14, 1)
    first = eng.search(0, 0, 3000)["move"]
    other_end = (0, -1, "N") if first == (0, 0, "S") else first  # a stick may be named from either end
    second = eng.search(0, 0, 3000, False, [first, other_end])["move"]
    assert second != first and eng.is_legal(*second)
    assert eng.search(0, 0, 1000, False, eng.legal_moves(True))["depth"] == 0


def test_the_book_offers_every_move_it_can_judge_and_varies_among_the_close_ones(tmp_path) -> None:
    import random

    book = Book(str(tmp_path / "book.json"))
    start = _game([])
    book.store(start, (0, 0, "S"), 10, 19)
    book.store(_game([(0, 0, "E")]), (1, 0, "S"), -5, 18)    # player 2's view: so worth +5 to player 1
    book.store(_game([(0, 0, "NE")]), (1, 1, "S"), 300, 18)  # a bad first move: -300 to player 1
    options = book.choices(start)
    assert [v for _, v in options] == [10, 5, -300]  # the four mirror images of 0,0E count once
    assert options[0][0] == (0, 0, "S") and len(start.moves) == 0

    rng = random.Random(1)
    picked = {book.pick(start, 40, rng)["move"] for _ in range(200)}
    # either good move, in any of its mirror images: 0,0S or 0,1N, and the four images of 0,0E
    assert picked == {(0, 0, "S"), (0, 1, "N"), (0, 0, "E"), (0, 0, "W"), (0, 1, "E"), (0, 1, "W")}
    assert {book.pick(start, 0, rng)["move"] for _ in range(10)} == {(0, 0, "S")}
    assert book.pick(_game([(0, 0, "S"), (0, -1, "S")]), 40, rng) is None  # not in the book


def test_the_engine_varies_its_book_moves_but_frozen_versions_do_not(tmp_path) -> None:
    pytest.importorskip("rs_engine")
    from players.rs_player import BOOK_VARIETY, RSPlayer

    path = str(tmp_path / "book.json")
    book = Book(path)
    book.store(_game([]), (0, 0, "S"), 10, 19)
    book.store(_game([(0, 0, "E")]), (1, 0, "S"), -5, 18)
    book.save()
    firsts = {RSPlayer(0, max_nodes=200, book=path, seed=s).get_move(_game([])) for s in range(30)}
    assert len(firsts) > 2 and BOOK_VARIETY > 0
    plain = {RSPlayer(0, max_nodes=200, book=path, seed=s, book_variety=0).get_move(_game([])) for s in range(10)}
    assert plain == {Move(0, 0, "S")}
