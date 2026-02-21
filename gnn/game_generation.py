import json
import math
import os
import random
from glob import glob
from typing import Callable, Iterable

from constants import HALF_AREA_COUNTS
from game import Game, GameProtocol
from gnn.encode import EncodedGraph, encode_game_to_graph
from models import D, Move, calculate_end
from players import Player


def _has_near_area_threat(game: GameProtocol) -> bool:
    """Return True if any single legal stick move for any player would score area.

    This catches randomized positions that are one stick away from an area closure,
    which tend to produce near-forced tactical wins that the value model can't
    evaluate correctly.  Measured with ``scripts/diag_quick_wins.py``.

    Note: In practice, the existing per-stick ``_path_of_smallest_area`` check
    in ``randomize_start`` already prevents all depth-1 area closures (measured:
    0/1000 rejections). This function is kept for correctness but the main
    filter is :func:`probe_quick_win`, which uses a short MCTS playout to catch
    deeper forced wins (4-6 moves out).
    """
    for player in range(game.num_players):
        for move in game.get_possible_moves(player):
            if move.t not in D.__members__:
                continue  # only stick moves can create area
            scores_before = list(game.players_scores)
            game.do_move(player, move)
            scored = game.players_scores[player] > scores_before[player]
            game.undo_move()
            if scored:
                return True
    return False


def probe_quick_win(
    game: GameProtocol,
    get_move_fn: Callable[[GameProtocol, int], Move],
    max_probe_moves: int = 6,
) -> bool:
    """Play a short probe game and return True if either side wins.

    After the probe, all moves are undone so the game state is restored.
    ``get_move_fn(game, player)`` should return a move for the given player
    in the current game state (e.g. a low-rollout MCTS call).
    """
    moves_made = 0
    winner_found = False
    for _ in range(max_probe_moves):
        if game.winner is not None:
            winner_found = True
            break
        cp = game.current_player
        mv = get_move_fn(game, cp)
        game.do_move(cp, mv)
        moves_made += 1
    if game.winner is not None:
        winner_found = True
    # Undo all probe moves to restore the original position.
    for _ in range(moves_made):
        game.undo_move()
    return winner_found


def randomize_start(
    game: GameProtocol,
    max_sticks: int = 5,
    max_rocks: int = 3, # per player
    move_log: list[Move] | None = None,
    mover_log: list[int] | None = None,
    filter_near_area: bool = False,
    max_filter_retries: int = 50,
    probe_fn: Callable[[GameProtocol, int], Move] | None = None,
    probe_max_moves: int = 6,
) -> None:
    """Randomize the starting position of the game.

    When *filter_near_area* is True **and** *probe_fn* is provided, after each
    randomization attempt a short probe game of up to *probe_max_moves* half-moves
    is played using *probe_fn* to obtain moves.  If either side wins in the probe,
    the position is discarded and a new randomization is attempted (up to
    *max_filter_retries* times).

    ``probe_fn(game, player) -> Move`` should return a move for *player* in the
    current game state.  A low-rollout MCTS call is typical.
    """
    # not true randomization; biased towards interesting / good for training positions
    # additionally, we make sure there does not exist an immediate win for either player

    use_filter = bool(filter_near_area) and probe_fn is not None
    max_outer = (int(max_filter_retries) + 1) if use_filter else 1

    for _filter_attempt in range(max_outer):
        # Track moves placed in this attempt so we can undo on rejection.
        _moves: list[Move] = []
        _movers: list[int] = []

        weights = [math.exp(-0.33 * ((k - (max_sticks + 0.5) / 2)) ** 4) for k in range(1, max_sticks + 1)]
        target_sticks = random.choices(range(1, max_sticks + 1), weights=weights, k=1)[0]

        attempts = 0
        while target_sticks > 0 and attempts < 20:
            attempts += 1
            moves = [m for m in game.get_possible_moves(game.current_player) if m.t in D.__members__]
            if not moves:
                break
            mv = random.choice(moves)
            mover = game.current_player
            game.do_move(mover, mv)
            failed = False
            for p in game.connected_points:
                for d in D:
                    q = game.points.get(calculate_end(p.c, d), None)
                    if q is not None and q.connected:
                        path = game._path_of_smallest_area(p, q)
                        if path is not None and (HALF_AREA_COUNTS or path[0] > 1):
                            failed = True
                            break
                else:
                    continue
                break
            if failed:
                game.undo_move()
                continue
            target_sticks -= 1
            _moves.append(mv)
            _movers.append(mover)

        alpha = 0.7
        r_weights = [math.exp(alpha * k) for k in range(0, max_rocks)]
        rocks_each = random.choices(range(0, max_rocks), weights=r_weights, k=1)[0]
        for p in game.players:
            for _ in range(rocks_each + (random.random() > 0.3)):
                rock_moves = [m for m in game.get_possible_moves(p.number) if m.t == "R"]
                if not rock_moves:
                    break
                rock_weights = [12.0 if ((node := game.points.get(m.c)) is not None and node.connected) else 1.0 for m in rock_moves]
                mv = random.choices(rock_moves, weights=rock_weights, k=1)[0]
                game.do_move(p.number, mv)
                _moves.append(mv)
                _movers.append(p.number)

        # --- filter check: probe for quick forced wins ---
        if use_filter and probe_fn is not None and _filter_attempt < max_outer - 1:
            # Temporarily set player 0 so the probe runs from the correct state.
            game.set_current_player0()
            rejected = probe_quick_win(game, probe_fn, max_probe_moves=probe_max_moves)
            if rejected:
                # Undo set_current_player0 side-effects: this only resets
                # current_player / turn tracking, which we'll redo on the next
                # successful attempt via set_current_player0 at the end.
                # However, we need to undo the placed moves to clear the board.
                # set_current_player0 is idempotent so just undo the moves.
                for _ in range(len(_moves)):
                    game.undo_move()
                continue

        # Passed filter (or last attempt, or filtering disabled) — commit.
        if move_log is not None:
            move_log.extend(_moves)
        if mover_log is not None:
            mover_log.extend(_movers)
        game.set_current_player0()
        return

def play_self_play_game(
    player_factories: Iterable[Callable[[int], Player]],
    max_moves: int = 256
) -> tuple[list[EncodedGraph], list[Move], int | None]:

    players = [factory(i) for i, factory in enumerate(player_factories)]
    game = Game(players)
    trajectory: list[EncodedGraph] = []
    move_log: list[Move] = []
    randomize_start(game, move_log=move_log)

    while game.winner is None and len(game.moves) < max_moves:
        player = game.players[game.current_player]
        trajectory.append(encode_game_to_graph(game))
        mv = player.get_move(game)
        move_log.append(mv)
        game.do_move(player.number, mv)
    return trajectory, move_log, game.winner


def generate_self_play_games(
    num_games: int,
    player_factories: Iterable[Callable[[int], Player]],
    save_games_dir: str,
    max_moves: int = 256,
    swap_roles: bool = True
) -> None:

    pf = list(player_factories)
    start_index = 0
    os.makedirs(save_games_dir, exist_ok=True)
    existing = [p for p in glob(os.path.join(save_games_dir, "game_*.json"))]
    if existing:
        def _idx(p: str) -> int:
            stem = os.path.basename(p)
            try:
                return int(stem.split("_")[1].split(".")[0])
            except Exception:
                return -1
        start_index = max(map(_idx, existing)) + 1
    for i in range(num_games):
        print(f"Generating game {i+1}/{num_games}...")
        this_pf = pf if (not swap_roles or (i % 2 == 0)) else list(reversed(pf))
        _, moves, winner = play_self_play_game(this_pf, max_moves=max_moves)
        game_path = os.path.join(save_games_dir, f"game_{start_index + i:05d}.json")
        payload: dict[str, object] = {
            "winner": winner,
            "moves": [
                {"x": m.c[0], "y": m.c[1], "t": m.t}
                for m in moves
            ],
            "max_moves_reached": (len(moves) >= max_moves and winner is None),
        }
        with open(game_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
