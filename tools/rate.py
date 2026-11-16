"""Ratings for bots and people (see ratings.py for how they are computed).

    python3 tools/rate.py                                # the ratings table
    python3 tools/rate.py --pairs                        # ... and every head-to-head record
    python3 tools/rate.py match rs:100k ab:3 --games 20  # play a rated match between two bots
    python3 tools/rate.py record you rs:5s:4t --winner 2 # enter a result by hand

Bot specs: random, ab:<depth>, mcts:<rollouts>, rs:<budget>[:<threads>t][:nobook] with a
budget such as 5s, 500ms, 100k or 1M (see players/specs.py). Any other name is a person.
Your games against bots in main.py are recorded automatically under the name "you"
(change it with --name).

Prefer node budgets (rs:100k) to time budgets (rs:1s) for bot-vs-bot rating: timed bots get
weaker when something else is using the CPU, which would bias their ratings.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ratings  # noqa: E402
from players.specs import canonical, is_bot_spec  # noqa: E402


def print_table(path: str, pairs: bool = False, only: set[str] | None = None) -> None:
    games = ratings.load_games(path)
    if not games:
        print("no rated games yet")
        return
    fit = ratings.fit(games)
    rows = [r for r in fit.ratings if only is None or r.name in only]
    width = max(10, max(len(r.name) for r in rows))
    print(f"{'':3s} {'player':{width}s} {'rating':>7s} {'±95%':>6s} {'games':>6s} {'W-L':>8s} {'score':>6s}  description")
    rank = 0
    for r in rows:
        rank += 1
        mark = "" if r.linked else " *"
        ci = "ref" if r.ci95 < 1e-9 and r.linked else f"{r.ci95:.0f}"
        wl = f"{r.wins}-{r.losses}" + (f"-{r.unfinished}" if r.unfinished else "")
        print(f"{rank:3d} {r.name:{width}s} {r.elo:7.0f} {ci:>6s} {r.games:6d} {wl:>8s} {100 * r.score:5.0f}%  {r.description}{mark}")
    print(f"\n{fit.n_games} games. Moving first is worth {fit.advantage:+.0f} ± {fit.advantage_ci95:.0f} points.")
    if fit.anchor:
        print(f"Scale: {fit.anchor} is fixed at {ratings.ANCHOR_ELO:.0f}; 200 points apart = about 76% for the stronger side.")
    else:
        print(f"Scale: {ratings.ANCHOR} has no games yet, so ratings are relative to the average of each group.")
    if any(not r.linked for r in rows):
        print("* not linked to the reference player by any chain of games: only comparable within its own group.")
    if pairs:
        print("\nHead to head (wins-losses[-unfinished] for the first player named):")
        rec: dict[tuple[str, str], list[int]] = {}
        for g in games:
            a = ratings.rating_key(g["p1"], g.get("v1"))
            b = ratings.rating_key(g["p2"], g.get("v2"))
            w = g.get("winner")
            if a > b:
                a, b, w = b, a, (None if w is None else 1 - w)
            t = rec.setdefault((a, b), [0, 0, 0])
            t[0 if w == 0 else 1 if w == 1 else 2] += 1
        for (a, b), t in sorted(rec.items()):
            print(f"  {a:>{width}s} vs {b:{width}s} {t[0]}-{t[1]}" + (f"-{t[2]}" if t[2] else ""))


def _play_one(spec_p1: str, spec_p2: str, seed: int, max_plies: int) -> tuple[int | None, int, str, list[int]]:
    """Play one game between two bots; returns (winner, plies, the moves played, and how many
    moves each side took from its opening book)."""
    import players_ext

    from game import Game
    from players.game_total import GameTotal
    from players.specs import make_bot

    p1 = make_bot(spec_p1, 0, seed=seed)
    p2 = make_bot(spec_p2, 1, seed=seed + 1)
    game = GameTotal(Game(players=[p1, p2]), players_ext.GameState())
    plies = 0
    moves = []
    book = [0, 0]
    while game.winner is None and plies < max_plies:
        cp = game.current_player
        mv = game.players[cp].get_move(game)
        book[cp] += bool(getattr(game.players[cp], "last_info", {}).get("book"))
        moves.append(repr(mv))
        game.do_move(cp, mv)
        plies += 1
    return game.winner, plies, " ".join(moves), book


def cmd_match(args) -> None:
    a, b = canonical(args.a), canonical(args.b)
    if a == b:
        raise SystemExit("a bot cannot play itself in a rated match")
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(3), "little") * 1000
    timed = [s for s in (a, b) if s.startswith("rs:") and s.split(":")[1].endswith("s")]
    if timed and args.jobs > 1:
        print(f"note: {', '.join(timed)} is timed; running games in parallel makes it weaker. Consider --jobs 1.")
    print(f"rated match: {a} vs {b}, {args.games} games, {args.jobs} at a time")
    score = 0.0
    done = 0
    seen = set()
    book_moves = {a: 0, b: 0}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {}
        for g in range(args.games):
            a_first = g % 2 == 0  # alternate who moves first
            p1, p2 = (a, b) if a_first else (b, a)
            futures[ex.submit(_play_one, p1, p2, seed + 10 * g, args.max_plies)] = (p1, p2, a_first)
        for fut in as_completed(futures):
            p1, p2, a_first = futures[fut]
            winner, plies, moves, book = fut.result()
            seen.add(moves)
            book_moves[p1] += book[0]
            book_moves[p2] += book[1]
            ratings.record_game(p1, p2, winner, plies=plies, source="match", path=args.file)
            res = 0.5 if winner is None else float((winner == 0) == a_first)
            score += res
            done += 1
            word = {1.0: "win", 0.0: "loss", 0.5: "unfinished"}[res]
            print(f"  game {done:3d}: {a} ({'first' if a_first else 'second'}) {word} in {plies} plies   [{a} {score:g}/{done}]", flush=True)
    print(f"{a} scored {score:g}/{args.games} against {b} ({(time.time() - t0) / 60:.1f} min)")
    # repeated games add no information, so say how many were different
    print(f"{len(seen)} different games", end="")
    used = [f"{s} {n / args.games:.1f}" for s, n in book_moves.items() if n]
    print(f"; book moves per game: {', '.join(used)}" if used else "", end="\n\n")
    print_table(args.file)


def cmd_record(args) -> None:
    w = {"1": 0, "2": 1, "none": None}[args.winner]
    rec = ratings.record_game(args.p1, args.p2, w, source="manual", note=args.note, path=args.file)
    print(f"recorded: {rec['p1']} vs {rec['p2']}, winner: {'unfinished' if w is None else rec['p1' if w == 0 else 'p2']}\n")
    print_table(args.file)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default=None, help="results file (default: ratings.jsonl, or $RS_RATINGS_FILE)")
    ap.add_argument("--pairs", action="store_true", help="also list every head-to-head record")
    sub = ap.add_subparsers(dest="cmd")

    m = sub.add_parser("match", help="play a rated match between two bots")
    m.add_argument("a")
    m.add_argument("b")
    m.add_argument("--games", type=int, default=20, help="games to play; sides alternate (default 20)")
    m.add_argument("--jobs", type=int, default=2, help="games played at once, one process each (default 2)")
    m.add_argument("--max-plies", type=int, default=160, help="unfinished after this many moves counts as half a win each")
    m.add_argument("--seed", type=int, default=None)

    r = sub.add_parser("record", help="enter a finished game by hand")
    r.add_argument("p1", help="who moved first: a bot spec or a person's name")
    r.add_argument("p2", help="who moved second")
    r.add_argument("--winner", required=True, choices=["1", "2", "none"], help="1 = first player won, 2 = second, none = unfinished")
    r.add_argument("--note", default=None)

    args = ap.parse_args()
    if args.cmd == "match":
        cmd_match(args)
    elif args.cmd == "record":
        for name in (args.p1, args.p2):
            if not is_bot_spec(name) and ":" in name:
                raise SystemExit(f"{name!r} looks like a bot spec but is not one (see --help)")
        cmd_record(args)
    else:
        print_table(args.file, pairs=args.pairs)


if __name__ == "__main__":
    main()
