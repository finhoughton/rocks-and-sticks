"""Play, watch or replay Rocks and Sticks.

    python3 main.py                              # you against the engine, you move first
    python3 main.py rs:100k                      # you against a particular bot
    python3 main.py rs:5s:4t:ponder human        # the engine moves first
    python3 main.py human human                  # two people at one screen
    python3 main.py rs:2s:4t rs:2s:4t --games 3  # watch two bots
    python3 main.py rs:100k ab:3 --games 20 --no-display --jobs 4   # a quick match, no window
    python3 main.py --replay saved_games_human/game_00040.json --analyse 2

The two arguments are Player 1 and Player 2. Each is a bot (rs:5s:4t, rs:100k, ab:3,
mcts:3000, random, ...: see players/specs.py) or a person: "human", or any other name. With
no arguments you play rs:5s:4t:ponder.

In the window, type a move into the box: "X Y D" for a stick from X,Y in direction N, NE, E,
SE, S, SW, W or NW; "X Y R" for a rock; "P" to pass. Type "resign" or click Resign twice to
give up.

Every finished game between two differently named players is added to the ratings
(python3 tools/rate.py shows the table) unless --no-rate is given. Games played in a window
are saved (yours to saved_games_human/, bot games to saved_games_bots/) and can be replayed
with --replay.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Iterator

import ratings
from game import Game
from models import PASS, Move
from players.base import HumanPlayer, Player, RandomPlayer, Resigned
from players.specs import canonical, is_bot_spec, make_bot, needs_cpp_state

DEFAULT_OPPONENT = "rs:5s:4t:ponder"
# where games played in a window are saved: with a person in them, and between bots
SAVE_DIRS = ("saved_games_human", "saved_games_bots")


def player_id(token: str, name: str) -> str:
    """The rating name of a player argument: a bot's canonical spec, or a person's name."""
    if is_bot_spec(token):
        return canonical(token)
    if ":" in token:
        canonical(token)  # looks like a bot spec but is not one: raise its error
    return name if token == "human" else token


def save_game(game, ids: list[str], evals: list, resigned: int | None, save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    taken = [int(os.path.basename(p)[5:10]) for p in glob.glob(os.path.join(save_dir, "game_*.json"))]
    path = os.path.join(save_dir, f"game_{max(taken, default=-1) + 1:05d}.json")
    humans = [i for i, pid in enumerate(ids) if not is_bot_spec(pid)]
    payload = {
        "winner": game.winner,
        "moves": [{"x": m.c[0], "y": m.c[1], "t": m.t} for m in game.moves],
        # one entry per move: the engine's search result for its own moves, None otherwise
        "evals": evals,
        "meta": {
            "players": ids,
            "human_player": humans[0] if len(humans) == 1 else None,
            "bot_spec": next((pid for pid in ids if is_bot_spec(pid)), None),
            "resigned": resigned,
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def play_one(ids: list[str], seed: int, opts: dict) -> dict:
    """Play one game between Player 1 = ids[0] and Player 2 = ids[1]; returns its result."""
    display, show_eval = opts["display"], opts["show_eval"]
    humans = [not is_bot_spec(pid) for pid in ids]
    players: list[Player] = []
    for n, pid in enumerate(ids):
        if humans[n]:
            players.append(HumanPlayer(n))
        else:
            bot = make_bot(pid, n, seed=seed + n, verbose=display)
            if opts["rs_weights"] and pid.startswith("rs:"):
                bot.engine.load_weights(opts["rs_weights"])  # type: ignore[attr-defined]
            players.append(bot)
    game = Game(players)
    if any(needs_cpp_state(pid) for pid in ids):
        import players_ext

        from players.game_total import GameTotal

        game = GameTotal(game, players_ext.GameState())  # type: ignore
    if opts["random_start"]:
        from gnn.game_generation import randomize_start

        random.seed(seed)
        randomize_start(game, max_sticks=5)

    def show(ev=None) -> None:
        if not display:
            return
        import matplotlib.pyplot as plt

        game.render(block=False, game_index=opts["index"], total_games=opts["total"])
        if show_eval:
            from players.rs_display import draw_eval_bar

            draw_eval_bar(plt.gcf(), ev)
        if not any(humans):
            plt.pause(opts["delay"])

    evals: list = []
    book = [0, 0]
    resigned = None
    show()
    while game.winner is None and len(game.moves) < opts["max_moves"]:
        cp = game.current_player
        t0 = time.perf_counter()
        try:
            m = players[cp].get_move(game)
        except Resigned:
            # the other player wins; the game is saved and rated like any other loss
            resigned = cp
            game.winner = 1 - cp
            getattr(game, "py", game).result_note = f"Player {cp + 1} resigned"
            break
        info = None if humans[cp] else getattr(players[cp], "last_info", None)
        evals.append({"score": int(info["score"]), "depth": int(info["depth"]), "mover": cp} if info else None)
        book[cp] += bool(info and info.get("book"))
        if display:
            took = "" if humans[cp] else f"  ({time.perf_counter() - t0:.1f}s)"
            print(f"{len(game.moves) + 1:3d}. Player {cp + 1} ({ids[cp]}): {'pass' if m is PASS else f'{m.c[0]} {m.c[1]} {m.t}'}{took}")
        game.do_move(cp, m)
        ev = None
        if show_eval and display:
            from players.rs_display import describe, final

            ev = final(game.winner) if game.winner is not None else (describe(info["score"], cp) if info else None)
        show(ev)
    for p in players:
        getattr(p, "stop_pondering", lambda: None)()

    result = {"winner": game.winner, "plies": len(game.moves), "moves": " ".join(repr(m) for m in game.moves), "book": book, "path": None}
    if display:
        if game.winner is None:
            print(f"No winner after {len(game.moves)} moves")
        else:
            how = "by resignation" if resigned is not None else f"with area {game.players_scores[game.winner] / 2:g}"
            print(f"Player {game.winner + 1} ({ids[game.winner]}) wins {how}")
    save_dir = opts["save_dir"] or (SAVE_DIRS[0 if any(humans) else 1] if display else None)
    if save_dir:
        # saved before the final position is shown: closing the terminal must not lose the game
        result["path"] = save_game(game, ids, evals, resigned, save_dir)
        if display:
            print(f"saved {result['path']}")
    if resigned is not None:
        show()
    return result


def _results(calls: list[tuple], jobs: int) -> Iterator[dict]:
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            yield from ex.map(play_one, *zip(*calls))
    else:
        for call in calls:
            yield play_one(*call)


def run_games(args) -> None:
    try:
        ids = [player_id(t, args.name) for t in args.players]
    except ValueError as e:
        raise SystemExit(str(e))
    bots_only = all(is_bot_spec(pid) for pid in ids)
    display = not args.no_display
    if not bots_only and not display:
        raise SystemExit("a person needs the window: drop --no-display")
    rated = not args.no_rate and ids[0] != ids[1] and not args.random_start and not args.rs_weights
    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(3), "little") * 1000
    many = args.games > 1
    opts = {
        "display": display, "show_eval": args.show_eval or bots_only, "delay": args.delay, "max_moves": args.max_moves,
        "random_start": args.random_start, "rs_weights": args.rs_weights, "save_dir": args.save_dir, "total": args.games if many else None,
    }
    # two different bots swap sides every game; a person keeps the side they asked for
    swap = bots_only and ids[0] != ids[1]
    order = [ids[::-1] if (swap and g % 2) else ids for g in range(args.games)]
    # results are reported for Player 1 as named on the command line (for a bot against
    # itself, simply for whoever moved first)
    who = ids[0] if ids[0] != ids[1] else "Player 1"
    jobs = 1 if display else max(1, args.jobs)
    if jobs > 1 and any(pid.startswith("rs:") and pid.split(":")[1].endswith("s") for pid in ids):
        print("note: a timed bot gets weaker when games run in parallel; consider --jobs 1 or a node budget such as rs:100k")
    calls = [(order[g], seed + 10 * g, {**opts, "index": g if many else None}) for g in range(args.games)]

    score, seen, book_moves = 0.0, set(), {pid: 0 for pid in ids}
    for g, r in enumerate(_results(calls, jobs)):
        p1, p2 = order[g]
        if rated:
            ratings.record_game(p1, p2, r["winner"], plies=r["plies"], source="match" if bots_only else "human", note=r["path"])
        first = order[g] is ids
        won = 0.5 if r["winner"] is None else float((r["winner"] == 0) == first)
        score += won
        seen.add(r["moves"])
        book_moves[p1] += r["book"][0]
        book_moves[p2] += r["book"][1]
        if many or not display:
            word = {1.0: "win", 0.0: "loss", 0.5: "unfinished"}[won]
            print(f"  game {g + 1:3d}: {who} ({'first' if first else 'second'}) {word} in {r['plies']} moves   [{who} {score:g}/{g + 1}]", flush=True)
        if display and g + 1 < args.games:
            import matplotlib.pyplot as plt

            plt.pause(max(3.0, 3 * args.delay))  # leave the final position up for a moment
            plt.close("all")

    if many:
        print(f"{who} scored {score:g}/{args.games}" + (f" against {ids[1]}" if ids[0] != ids[1] else ""))
        # repeated games add no information, so say how many were different
        used = [f"{pid} {n / args.games / (2 if ids[0] == ids[1] else 1):.1f}" for pid, n in book_moves.items() if n]
        print(f"{len(seen)} different games" + (f"; book moves per game: {', '.join(used)}" if used else ""))
    if rated:
        for r in ratings.fit(ratings.load_games()).ratings:
            if r.name in ids:
                print(f"  rating  {r.name:16s} {r.elo:6.0f} ± {r.ci95:.0f}   ({r.wins}-{r.losses} in {r.games} rated games)")
        print("  (full table: python3 tools/rate.py)")
    if display:
        import matplotlib.pyplot as plt

        print("Close the window to exit.")
        plt.show()


def replay(path: str, delay: float, analyse: float, threads: int) -> None:
    """Show a saved game move by move, with the engine's evaluations if it has them (or, with
    analyse > 0, with a fresh evaluation of every position after that many seconds' thought)."""
    import matplotlib.pyplot as plt

    from players.rs_display import describe, draw_eval_bar, final

    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    saved = payload.get("evals") or []
    show_eval = analyse > 0 or any(saved)
    analyst = None
    if analyse > 0:
        from players.rs_player import RSPlayer

        analyst = RSPlayer(0, time_ms=int(analyse * 1000), threads=threads, tt_bits=22)
    game = Game([RandomPlayer(0), RandomPlayer(1)])
    game.render(block=False)
    for i, mv in enumerate(payload["moves"]):
        m = PASS if mv["t"] == "P" else Move(int(mv["x"]), int(mv["y"]), str(mv["t"]))
        mover = int(mv["p"]) if mv.get("p") is not None else game.current_player
        game.do_move(mover, m)
        ev = None
        if show_eval:
            if game.winner is not None:
                ev = final(game.winner)
            elif analyst is not None:
                info = analyst.analyse(game)
                ev = describe(info["score"], info["mover"])
            elif i < len(saved) and saved[i]:
                # recorded by the engine just before it played this move
                ev = describe(int(saved[i]["score"]), int(saved[i]["mover"]))
        print(f"{i + 1:3d}. Player {mover + 1}: {m}" + (f"   {ev.text}" if ev is not None else ""))
        game.render(block=False)
        if show_eval:
            draw_eval_bar(plt.gcf(), ev)
        plt.pause(delay)
    resigned = (payload.get("meta") or {}).get("resigned")
    if game.winner is None and payload.get("winner") is not None and resigned is not None:
        game.winner, game.result_note = payload["winner"], f"Player {resigned + 1} resigned"
        game.render(block=False)
    print(f"Game over. Winner: {'none' if payload.get('winner') is None else 'Player ' + str(payload['winner'] + 1)}")
    plt.show()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("players", nargs="*", help='Player 1 and Player 2: a bot spec, "human", or a person\'s name')
    ap.add_argument("--games", type=int, default=1, help="games to play; bots swap sides each game")
    ap.add_argument("--no-display", action="store_true", help="no window (bots only): just play and report")
    ap.add_argument("--jobs", type=int, default=1, help="with --no-display: games played at once, one process each")
    ap.add_argument("--delay", type=float, default=1.0, help="pause after each move when watching bots or a replay, in seconds")
    ap.add_argument("--show-eval", action="store_true", help="draw the engine's evaluation bar while you play (always on when watching bots)")
    ap.add_argument("--name", default="you", help='the rating name of "human" (default: you)')
    ap.add_argument("--no-rate", action="store_true", help="do not add these games to the ratings")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-moves", type=int, default=160, help="a game with no winner after this many moves is unfinished")
    ap.add_argument("--random-start", action="store_true", help="start from a random position (unrated)")
    ap.add_argument("--rs-weights", default=None, help="evaluation file for rs bots instead of their own (unrated)")
    ap.add_argument("--save-dir", default=None, help=f"where to save games (default: {SAVE_DIRS[0]}/ or {SAVE_DIRS[1]}/ for games in a window, nowhere with --no-display)")
    ap.add_argument("--replay", metavar="FILE", help="replay a saved game instead of playing")
    ap.add_argument("--analyse", type=float, default=0.0, help="with --replay: have the engine evaluate each position for this many seconds")
    ap.add_argument("--threads", type=int, default=4, help="search threads for --analyse")
    args = ap.parse_args(argv)

    if args.replay:
        replay(args.replay, args.delay, args.analyse, args.threads)
        return
    if len(args.players) > 2:
        ap.error("give at most two players")
    if len(args.players) < 2:  # nobody named: you against the engine; one name: you against them
        args.players = ["human"] + (args.players or [DEFAULT_OPPONENT])
    run_games(args)


if __name__ == "__main__":
    main()
