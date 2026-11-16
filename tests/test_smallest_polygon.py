"""A stick scores the smallest polygon it completes.

game.py finds that polygon by walking the two faces beside the stick. Check the result against
a brute-force search over every simple cycle through the stick, and check that it does not
depend on which end the stick is placed from.
"""

from __future__ import annotations

import random

from game import Game
from models import PASS, D, Move, Node, calculate_area
from players.base import RandomPlayer


def _brute_force_area2(game: Game, start: Node, end: Node) -> int | None:
    """Smallest non-zero area (x2) over all simple paths start -> end, closed by end -> start."""
    best: int | None = None
    path = [start]
    on_path = {start}

    def walk(node: Node) -> None:
        nonlocal best
        for nbr in node.neighbours_list:
            if nbr is end:
                if len(path) >= 2:
                    a = calculate_area(path + [end])
                    if a and (best is None or a < best):
                        best = a
            elif nbr not in on_path:
                on_path.add(nbr)
                path.append(nbr)
                walk(nbr)
                path.pop()
                on_path.discard(nbr)

    walk(start)
    return best


def _candidate_pairs(game: Game):
    for p in list(game.connected_points):
        for d in D:
            if p.neighbours[d.as_int] is not None or game.intersects_stick(p.c, d):
                continue
            q = game.points.get((p.x + d.delta[0], p.y + d.delta[1]))
            if q is not None and q.connected:
                yield p, q, d


def _random_dense_game(seed: int, n_moves: int) -> Game:
    rng = random.Random(seed)
    game = Game([RandomPlayer(0), RandomPlayer(1)])
    for _ in range(n_moves):
        moves = [m for m in game.get_possible_moves(game.current_player) if m is not PASS and m.t != "R"]
        joins = []
        for m in moves:
            end = game.points.get((m.c[0] + D[m.t].delta[0], m.c[1] + D[m.t].delta[1]))
            if end is not None and end.connected:
                joins.append(m)
        compact = [m for m in moves if abs(m.c[0]) <= 2 and abs(m.c[1]) <= 2]
        r = rng.random()
        pool = joins if (joins and r < 0.45) else (compact if (compact and r < 0.9) else moves)
        game.do_move(game.current_player, rng.choice(pool))
    return game


def test_matches_brute_force_and_is_orientation_independent() -> None:
    checked = 0
    with_cycles = 0
    for seed in range(120):
        game = _random_dense_game(seed, n_moves=10 + seed % 9)
        for p, q, d in _candidate_pairs(game):
            expected = _brute_force_area2(game, p, q)
            got = game._path_of_smallest_area(p, q)
            assert (got[0] if got else None) == expected, f"seed {seed} {p.c}->{q.c}: face walk {got and got[0]} brute force {expected}"
            back = game._path_of_smallest_area(q, p)
            assert (back[0] if back else None) == expected

            # placing the stick from either end scores the same
            gains = []
            for m in (Move(p.x, p.y, d.name), Move(q.x, q.y, d.reversed.name)):
                before = game.players_scores[0]
                game.do_move(0, m)
                gains.append(game.players_scores[0] - before)
                game.undo_move()
            want = 0 if (expected is None or expected == 1) else expected
            assert gains == [want, want], f"seed {seed} {p.c}->{q.c}: gains {gains} expected {want}"
            checked += 1
            with_cycles += expected is not None and expected > 1
    assert checked > 500 and with_cycles > 100, (checked, with_cycles)


def test_polygon_vertices_form_the_scored_region() -> None:
    game = Game([RandomPlayer(0), RandomPlayer(1)])
    for m in (Move(0, 1, "E"), Move(1, 1, "S")):
        game.do_move(game.current_player, m)
    game.do_move(0, Move(1, 0, "W"))  # closes the unit square
    assert game.players_scores[0] == 2
    region = game.claimed_regions[-1]
    assert sorted(region.vertices) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_old_cpp_state_agrees_on_who_wins() -> None:
    """players_ext.GameState uses the same face walk; its winner must track game.py's."""
    import players_ext

    from players.move_utils import to_cpp_move

    for seed in range(40):
        rng = random.Random(500 + seed)
        game = Game([RandomPlayer(0), RandomPlayer(1)])
        cpp = players_ext.GameState()
        for ply in range(30):
            moves = [m for m in game.get_possible_moves(game.current_player) if m is not PASS]
            sticks = [m for m in moves if m.t != "R"]
            joins = []
            for m in sticks:
                end = game.points.get((m.c[0] + D[m.t].delta[0], m.c[1] + D[m.t].delta[1]))
                if end is not None and end.connected:
                    joins.append(m)
            r = rng.random()
            pool = joins if (joins and r < 0.5) else (sticks if r < 0.9 else moves)
            m = rng.choice(pool)
            mover = game.current_player
            game.do_move(mover, m)
            cpp.do_move(to_cpp_move(m), mover)
            assert cpp.winner == (-1 if game.winner is None else game.winner), f"seed {seed} ply {ply} after {m}"
            assert cpp.current_player == game.current_player
