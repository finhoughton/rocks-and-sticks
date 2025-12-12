from __future__ import annotations

import copy
import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, DefaultDict, Generator, Iterator, List, cast

from constants import AI_DEPTH, N_ROCKS
from models import PASS, Move, move_sort_key

if TYPE_CHECKING:
    from game import Game
    from models import Node


@contextmanager
def applied_move(game: Game, player: Player, move: Move) -> Generator[None, None, None]:
    """Apply a move and reliably undo it.

    Keeps search code readable while still guaranteeing `undo_move()` runs
    even on early returns/breaks/exceptions.
    """

    game.do_move(player, move)
    try:
        yield
    finally:
        game.undo_move(player)


@contextmanager
def rollback_to(game: Game) -> Generator[None, None, None]:
    """Rollback `game` to the move count at context entry."""

    start_len = len(game.moves)
    try:
        yield
    finally:
        # `undo_move` doesn't actually depend on which Player is passed.
        p0 = game.players[0]
        while len(game.moves) > start_len:
            game.undo_move(p0)


class Player(ABC):
    def __init__(self, player_number: int):
        self.number = player_number

    @abstractmethod
    def get_move(self, game: Game) -> Move:
        raise NotImplementedError

    def can_place(self, point: Node) -> bool:
        # Identity comparison avoids calling Player.__eq__ in hot paths.
        return point.rocked_by is None or point.rocked_by is self

    def __hash__(self) -> int:
        return self.number

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Player):
            return False
        return self.number == value.number

    def __deepcopy__(self, memo: dict[int, object]) -> Player:
        # Players are effectively immutable identifiers.
        # Crucially, this prevents deepcopy(Game) from cloning MCTSPlayer's
        # internal search dictionaries, which can be huge.
        return self


class HumanPlayer(Player):
    def get_move(self, game: Game) -> Move:
        inp = input(f"Player {self.number + 1}, enter move: ")
        if inp == "P":
            return PASS
        try:
            x_s, y_s, t = inp.split()
            x = int(x_s)
            y = int(y_s)

            m = Move(x, y, t)

            if not game.valid_move(m, self):
                raise ValueError("Invalid move")

        except Exception as e:
            print(f"{str(e)}, try again. got input: {inp}")
            return self.get_move(game)
        return m


class RandomPlayer(Player):
    def __init__(self, player_number: int, seed: int | None = None):
        super().__init__(player_number)
        self._rng = random.Random(seed)

    def get_move(self, game: Game) -> Move:
        moves = list(game.get_possible_moves(self))
        # Keep behaviour deterministic under a fixed seed.
        moves.sort(key=lambda m: (m.t, m.c[0], m.c[1]))
        return self._rng.choice(moves)


class AIPlayer(Player):
    def __init__(self, player_number: int):
        super().__init__(player_number)
        # Per-`get_move` caches (cleared at the start of each search).
        self._eval_cache: dict[tuple[object, ...], float] = {}
        self._pot_cache: dict[tuple[tuple[object, ...], int], float] = {}
        self._sticks_opp_cache: dict[tuple[tuple[object, ...], int], float] = {}

    def _clear_search_caches(self) -> None:
        self._eval_cache.clear()
        self._pot_cache.clear()
        self._sticks_opp_cache.clear()

    def _potential_area_cached(
        self, game: Game, player_idx: int, state_key: tuple[object, ...] | None = None
    ) -> float:
        if state_key is None:
            state_key = _game_key(game)
        k = (state_key, player_idx)
        v = self._pot_cache.get(k)
        if v is not None:
            return v
        v = AIPlayer.calculate_potential_area(game, game.players[player_idx])
        self._pot_cache[k] = v
        return v

    def _stick_opportunities_cached(
        self, game: Game, player_idx: int, state_key: tuple[object, ...] | None = None
    ) -> float:
        if state_key is None:
            state_key = _game_key(game)
        k = (state_key, player_idx)
        v = self._sticks_opp_cache.get(k)
        if v is not None:
            return v
        v = AIPlayer.count_stick_opportunities(game, game.players[player_idx])
        self._sticks_opp_cache[k] = v
        return v

    w1 = 2.0
    w2 = 1.0
    w3 = 1.0
    w4 = 2.0
    w5 = 5.0

    def evaluate_position_heuristic(self, game: Game) -> float:
        k = _game_key(game)
        cached = self._eval_cache.get(k)
        if cached is not None:
            return cached

        if game.winner is not None:
            if game.winner == self.number:
                v = float("inf")
            else:
                v = float("-inf")
            self._eval_cache[k] = v
            return v

        opp = game.players[1 - self.number]

        potential_area = self._potential_area_cached(game, self.number)
        blocking_opponent = self.calculate_blocking_opponent(game, self)
        stick_opportunities = self._stick_opportunities_cached(game, self.number)
        opponent_progress = self._potential_area_cached(game, opp.number)

        v = (
            self.w1 * potential_area
            + self.w2 * blocking_opponent
            + self.w3 * stick_opportunities
            - self.w4 * opponent_progress
            - self.w5 * game.num_rocks[self.number]
        )

        self._eval_cache[k] = v
        return v

    @staticmethod
    def _rock_is_search_worthy(game: Game, c: tuple[int, int]) -> bool:
        """
        Keep rock moves that are either:
        - on a stick-connected node (part of the current structure), OR
        - adjacent to at least two existing rocks (forming/contesting clusters).
        """

        point = game.points.get(c)
        if point is None:
            return False
        if point.connected:
            return True

        rock_coords = {r.c for r in game.rocks}
        x, y = c
        adjacent_rocks = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in rock_coords:
                    adjacent_rocks += 1
                    if adjacent_rocks >= 2:
                        return True
        return False

    @classmethod
    def iter_search_moves(cls, game: Game, player: Player) -> Iterator[Move]:
        for m in game.get_possible_moves(player):
            if m is PASS:
                yield m
                continue
            if m.t == "R" and not cls._rock_is_search_worthy(game, m.c):
                continue
            yield m

    @classmethod
    def calculate_potential_area(cls, game: Game, player: Player) -> float:
        potential_area = 0
        points = game.points
        can_place = player.can_place
        for point in game.connected_points:
            if not can_place(point):
                continue
            x, y = point.c
            for d in point.empty_directions:
                dx, dy = d.delta
                end = points.get((x + dx, y + dy))
                if end is not None and can_place(end):
                    potential_area += 1
        return potential_area

    @classmethod
    def calculate_blocking_opponent(cls, game: Game, player: Player) -> float:
        return 0

    @classmethod
    def count_stick_opportunities(cls, game: Game, player: Player) -> float:
        total = 0
        for point in game.connected_points:
            if not player.can_place(point):
                continue
            total += 8 - point.neighbour_count

        return float(total)


class AlphaBetaPlayer(AIPlayer):
    def get_move(self, game: Game) -> Move:
        if game.num_players != 2:
            raise ValueError("Alpha-beta AI is only implemented for 2 players")
        self._clear_search_caches()
        move, _ = self.alpha_beta(game, AI_DEPTH, float("-inf"), float("inf"), True)
        return move

    def alpha_beta(
        self, game: Game, depth: int, a: float, b: float, maximising: bool
    ) -> tuple[Move, float]:
        if depth == 0 or game.winner is not None:
            return (PASS, self.evaluate_position_heuristic(game))

        best_move = PASS

        if maximising:
            value = float("-inf")
            for move in AIPlayer.iter_search_moves(game, self):
                game.do_move(self, move)
                _, v2 = self.alpha_beta(game, depth - 1, a, b, False)
                game.undo_move(self)

                if v2 > value:
                    value = v2
                    best_move = move
                if value > b:
                    break
                a = max(a, value)
            return (best_move, value)
        value = float("inf")
        opp = game.players[1 - self.number]
        for move in AIPlayer.iter_search_moves(game, opp):
            game.do_move(opp, move)
            _, v2 = self.alpha_beta(game, depth - 1, a, b, True)
            game.undo_move(opp)
            if v2 < value:
                value = v2
                best_move = move
            if value < a:
                break
            b = min(b, value)
        return (best_move, value)


# https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
class MCTSPlayer(AIPlayer):
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(
        self,
        player_number: int,
        exploration_weight: float = 1.0,
        seed: int | None = None,
        n_rollouts: int = 1000,
        max_sim_depth: int = 60,
        tactical_root_limit: int = 20,
        tactical_branch_limit: int = 12,
        progressive_widening_c: float = 1.6,
        progressive_widening_alpha: float = 0.55,
        prior_eval_cap: int = 48,
        rave_k: float = 250.0,
        rock_prior_bonus_connected: float = 0.55,
        rock_prior_bonus_disconnected: float = 0.06,
        rock_rollout_bonus_connected: float = 0.14,
        rock_rollout_bonus_disconnected: float = 0.02,
    ):
        # Transposition-table MCTS:
        # - Stats are stored per state key, and per (state, move) edge.
        # - This merges equivalent positions reached via different paths.
        self.Ns: DefaultDict[tuple[object, ...], int] = defaultdict(int)
        self.Nsa: DefaultDict[
            tuple[tuple[object, ...], tuple[str, tuple[int, int]]], int
        ] = defaultdict(int)
        self.Wsa: DefaultDict[
            tuple[tuple[object, ...], tuple[str, tuple[int, int]]], float
        ] = defaultdict(float)
        self.Psa: dict[
            tuple[tuple[object, ...], tuple[str, tuple[int, int]]], float
        ] = {}
        self._legal_moves: dict[tuple[object, ...], list[Move]] = {}
        self._expanded_count: DefaultDict[tuple[object, ...], int] = defaultdict(int)

        # Root-only RAVE/AMAF stats.
        self.N_amaf: DefaultDict[
            tuple[tuple[object, ...], tuple[str, tuple[int, int]]], int
        ] = defaultdict(int)
        self.W_amaf: DefaultDict[
            tuple[tuple[object, ...], tuple[str, tuple[int, int]]], float
        ] = defaultdict(float)
        self.rave_k = rave_k
        self._root_key: tuple[object, ...] | None = None

        # Interpreting exploration_weight as the PUCT exploration constant.
        self.c_puct: float = exploration_weight
        self._rng = random.Random(seed)
        self.n_rollouts = n_rollouts
        self.max_sim_depth = max_sim_depth
        self.tactical_root_limit = tactical_root_limit
        self.tactical_branch_limit = tactical_branch_limit
        self.progressive_widening_c = progressive_widening_c
        self.progressive_widening_alpha = progressive_widening_alpha
        self.prior_eval_cap = prior_eval_cap
        self.rock_prior_bonus_connected = rock_prior_bonus_connected
        self.rock_prior_bonus_disconnected = rock_prior_bonus_disconnected
        self.rock_rollout_bonus_connected = rock_rollout_bonus_connected
        self.rock_rollout_bonus_disconnected = rock_rollout_bonus_disconnected
        super().__init__(player_number)

    def _rock_bonus_for_cell(
        self,
        game: Game,
        c: tuple[int, int],
        *,
        connected_bonus: float,
        disconnected_bonus: float,
    ) -> float:
        p = game.points.get(c)
        if p is not None and p.connected:
            return connected_bonus
        return disconnected_bonus

    @staticmethod
    def _move_key(m: Move) -> tuple[str, tuple[int, int]]:
        return (m.t, m.c)

    @staticmethod
    def _stable_move_tiebreak(m: Move) -> tuple[str, int, int]:
        return move_sort_key(m)

    def _heuristic_probability(self, game: Game, perspective_player: int) -> float:
        """Return a soft win-probability estimate in [0, 1] for `perspective_player`."""
        if game.winner is not None:
            return 1.0 if game.winner == perspective_player else 0.0
        if game.num_players != 2:
            return 0.5

        opp = 1 - perspective_player
        state_key = _game_key(game)
        score_diff = game.players_scores[perspective_player] - game.players_scores[opp]
        # generally, placing rocks is good. they are powerful tempo/denial moves.
        # This term is intentionally NOT a diff vs the opponent.
        rocks_used_me = N_ROCKS - game.num_rocks[perspective_player]
        pot_diff = self._potential_area_cached(
            game, perspective_player, state_key
        ) - self._potential_area_cached(game, opp, state_key)
        sticks_diff = self._stick_opportunities_cached(
            game, perspective_player, state_key
        ) - self._stick_opportunities_cached(game, opp, state_key)

        # Tuned to provide a useful gradient without dominating on any single term.
        value = (
            2.0 * score_diff
            + 0.35 * pot_diff
            + 0.10 * sticks_diff
            + 0.55 * rocks_used_me
        )
        return 0.5 + 0.5 * math.tanh(value / 6.0)

    def _rollout_pick_move(self, game: Game) -> Move:
        """Pick a rollout move: mostly greedy by heuristic, occasionally random."""
        player = game.players[game.current_player]
        moves = list(AIPlayer.iter_search_moves(game, player))
        moves.sort(key=move_sort_key)
        if len(moves) <= 1:
            return moves[0]

        # Exploration inside rollout.
        if self._rng.random() < 0.20:
            return self._rng.choice(moves)

        # Evaluate a subset to keep rollouts fast.
        if len(moves) > 12:
            candidates = self._rng.sample(moves, 12)
            candidates.sort(key=move_sort_key)
        else:
            candidates = moves

        best_move = candidates[0]
        best_score = float("-inf")
        mover_idx = player.number
        for m in candidates:
            with applied_move(game, player, m):
                s = self._heuristic_probability(game, mover_idx)
            if m.t == "R":
                s += self._rock_bonus_for_cell(
                    game,
                    m.c,
                    connected_bonus=self.rock_rollout_bonus_connected,
                    disconnected_bonus=self.rock_rollout_bonus_disconnected,
                )
            if s > best_score:
                best_score = s
                best_move = m
        return best_move

    def get_move(self, game: Game) -> Move:
        self._clear_search_caches()
        self.Ns.clear()
        self.Nsa.clear()
        self.Wsa.clear()
        self.Psa.clear()
        self._legal_moves.clear()
        self._expanded_count.clear()
        self.N_amaf.clear()
        self.W_amaf.clear()
        self._root_key = None

        working_game = copy.deepcopy(game)
        root_key = _game_key(working_game)
        self._root_key = root_key

        # Tactical fast-path:
        # - take any immediate winning move
        # - never play a move that immediately makes the opponent the winner
        player = working_game.players[working_game.current_player]
        root_moves = sorted(
            AIPlayer.iter_search_moves(working_game, player), key=move_sort_key
        )

        safe_root_moves: list[Move] = []
        for move in root_moves:
            with applied_move(working_game, player, move):
                if working_game.winner == self.number:
                    return move
                if working_game.winner is None:
                    safe_root_moves.append(move)

        def allows_forced_loss_next_round(my_move: Move) -> bool:
            # True if opponent has a reply such that every response loses.

            def top_k_by_heuristic(
                p: Player, moves: list[Move], k: int, perspective_player: int
            ) -> list[Move]:
                if k <= 0 or len(moves) <= k:
                    return moves
                scored: list[tuple[float, Move]] = []
                for mv in moves:
                    with applied_move(working_game, p, mv):
                        s = self._heuristic_probability(
                            working_game, perspective_player
                        )
                    scored.append((s, mv))
                scored.sort(key=lambda sm: (-sm[0], move_sort_key(sm[1])))
                return [mv for _, mv in scored[:k]]

            with applied_move(working_game, player, my_move):
                # If this move already ends the game (should have been filtered),
                # treat it as forced loss.
                if (
                    working_game.winner is not None
                    and working_game.winner != self.number
                ):
                    return True

                opp = working_game.players[working_game.current_player]
                opp_moves = list(AIPlayer.iter_search_moves(working_game, opp))
                opp_moves.sort(key=move_sort_key)

                # Keep this tactical check bounded.
                opp_moves = top_k_by_heuristic(
                    opp,
                    opp_moves,
                    self.tactical_branch_limit,
                    perspective_player=opp.number,
                )

                for opp_move in opp_moves:
                    with applied_move(working_game, opp, opp_move):
                        # If opponent instantly wins here, it's forced.
                        if working_game.winner == opp.number:
                            return True

                        me = working_game.players[working_game.current_player]
                        responses = list(AIPlayer.iter_search_moves(working_game, me))
                        responses.sort(key=move_sort_key)

                        responses = top_k_by_heuristic(
                            me,
                            responses,
                            self.tactical_branch_limit,
                            perspective_player=me.number,
                        )

                        has_defense = False
                        for response in responses:
                            with applied_move(working_game, me, response):
                                # After our response (end-of-round), if opponent
                                # is NOT winner, then we have a defense.
                                if working_game.winner != opp.number:
                                    has_defense = True
                                    break

                        if not has_defense:
                            return True

                return False

        # Prefer moves that do NOT allow a forced loss next round.
        allowed_root_moves = safe_root_moves
        if safe_root_moves:
            # Only check the first N moves to keep runtime sane.
            to_check = safe_root_moves[: self.tactical_root_limit]
            non_forced_loss = [
                m for m in to_check if not allows_forced_loss_next_round(m)
            ]
            if non_forced_loss:
                allowed_root_moves = non_forced_loss

        for _ in range(self.n_rollouts):
            self.do_rollout(root_key, working_game)

        best_move = self.choose(
            root_key, working_game, allowed_root_moves=allowed_root_moves
        )

        # Final tactical safety pass: avoid "one-move blunders" where our move
        # immediately hands the opponent the win at end-of-round, or where the
        # opponent has a forcing line next round.
        def visits(m: Move) -> int:
            return self.Nsa[(root_key, self._move_key(m))]

        ranked_moves = (
            list(allowed_root_moves) if allowed_root_moves else list(root_moves)
        )
        ranked_moves.sort(key=lambda m: (-visits(m), move_sort_key(m)))

        safety_limit = min(len(ranked_moves), max(self.tactical_root_limit, 40))
        for m in ranked_moves[:safety_limit]:
            with applied_move(working_game, player, m):
                if working_game.winner == self.number:
                    return m
                if (
                    working_game.winner is not None
                    and working_game.winner != self.number
                ):
                    continue

            if allows_forced_loss_next_round(m):
                continue
            return m

        return best_move

    def choose(
        self,
        root_key: tuple[object, ...],
        game: Game,
        allowed_root_moves: list[Move] | None = None,
    ) -> Move:
        "Choose the best move from the root using search statistics."
        if root_key[0] is not None:
            return PASS

        self._ensure_state_initialized(root_key, game)
        legal = self._legal_moves.get(root_key, [])
        if not legal:
            return PASS

        candidates = legal
        if allowed_root_moves is not None and allowed_root_moves:
            allowed = {(m.t, m.c) for m in allowed_root_moves}
            filtered = [m for m in candidates if (m.t, m.c) in allowed]
            if filtered:
                candidates = filtered

        # Take an immediate win if available (even if never visited).
        player = game.players[game.current_player]
        for m in candidates:
            with applied_move(game, player, m):
                if game.winner == self.number:
                    return m

        # Robust choice: pick the most visited move (stable tie-break).
        def visits(m: Move) -> int:
            return self.Nsa[(root_key, self._move_key(m))]

        max_visits = max((visits(m) for m in candidates), default=0)
        best = [m for m in candidates if visits(m) == max_visits]
        return min(best, key=move_sort_key)

    def do_rollout(self, root_key: tuple[object, ...], game: Game) -> None:
        "Make the tree one layer better. (Train for one iteration.)"
        with rollback_to(game):
            path_edges, _n_applied = self._select(root_key, game)
            reward, _n_sim_applied, sim_moves, sim_start_player = self._simulate(game)

            # Root-only AMAF/RAVE update: any move played by the root player
            # later in the rollout is treated as if it were played first.
            root_player_idx = cast(int, root_key[2])
            root_played: set[tuple[str, tuple[int, int]]] = set()
            for state_key, move in path_edges:
                mover = cast(int, state_key[2])
                if mover == root_player_idx:
                    root_played.add(self._move_key(move))
            for mover, mk in sim_moves:
                if mover == root_player_idx:
                    root_played.add(mk)

            # Map rollout reward (from the leaf-to-move perspective) back to the root.
            # for 2 players this is just a parity flip.
            root_player_idx = cast(int, root_key[2])
            root_reward = (
                reward if sim_start_player == root_player_idx else (1.0 - reward)
            )
            for mk in root_played:
                edge = (root_key, mk)
                self.N_amaf[edge] += 1
                self.W_amaf[edge] += root_reward
            self._backpropagate(path_edges, reward)

    def _select(
        self, root_key: tuple[object, ...], game: Game
    ) -> tuple[List[tuple[tuple[object, ...], Move]], int]:
        "Select a path using PUCT + progressive widening, mutating `game` along the path."
        path_edges: List[tuple[tuple[object, ...], Move]] = []
        n_applied = 0
        state_key = root_key

        while True:
            if state_key[0] is not None:
                return path_edges, n_applied

            self._ensure_state_initialized(state_key, game)
            self._ensure_progressive_widening(state_key)

            if not self._legal_moves.get(state_key, []):
                return path_edges, n_applied

            move = self._puct_select_move(state_key)

            player = game.players[game.current_player]
            game.do_move(player, move)
            n_applied += 1
            path_edges.append((state_key, move))

            # Stop after taking a previously-unvisited edge.
            if self.Nsa[(state_key, self._move_key(move))] == 0:
                return path_edges, n_applied

            state_key = _game_key(game)

    def _ensure_state_initialized(
        self, state_key: tuple[object, ...], game: Game
    ) -> None:
        if state_key in self._legal_moves:
            return
        if state_key[0] is not None:
            self._legal_moves[state_key] = []
            self._expanded_count[state_key] = 0
            return

        player = game.players[game.current_player]
        moves = sorted(self.iter_search_moves(game, player), key=move_sort_key)

        # Compute heuristic priors for a capped subset to keep expansion cheap.
        mover_idx = player.number
        priors: list[tuple[Move, float]] = []

        eval_moves = moves
        if self.prior_eval_cap > 0 and len(moves) > self.prior_eval_cap:
            # Deterministic: evaluate first N moves in sorted order, but also
            # ensure we evaluate some rock moves so they can earn meaningful priors.
            eval_moves = moves[: self.prior_eval_cap]
            extra_rocks: list[Move] = []
            for m in moves[self.prior_eval_cap :]:
                if m.t == "R":
                    extra_rocks.append(m)
                    if len(extra_rocks) >= 12:
                        break
            if extra_rocks:
                eval_moves = [*eval_moves, *extra_rocks]

        for m in eval_moves:
            with applied_move(game, player, m):
                p = self._heuristic_probability(game, mover_idx)
            if m.t == "R":
                p = min(
                    0.999,
                    p
                    + self._rock_bonus_for_cell(
                        game,
                        m.c,
                        connected_bonus=self.rock_prior_bonus_connected,
                        disconnected_bonus=self.rock_prior_bonus_disconnected,
                    ),
                )
            priors.append((m, float(p)))

        # For non-evaluated moves, assign a small prior so they still become
        # available via progressive widening.
        if len(eval_moves) < len(moves):
            min_p = min((p for _, p in priors), default=0.01)
            floor_p = max(0.005, 0.25 * min_p)
            for m in moves[len(eval_moves) :]:
                p = floor_p
                if m.t == "R":
                    p = min(
                        0.999,
                        p
                        + self._rock_bonus_for_cell(
                            game,
                            m.c,
                            connected_bonus=self.rock_prior_bonus_connected,
                            disconnected_bonus=self.rock_prior_bonus_disconnected,
                        ),
                    )
                priors.append((m, p))

        # PASS is always legal but is rarely correct early; give it a very small
        # prior so it won't dominate progressive widening/PUCT unless search
        # evidence makes it clearly best.
        if len(moves) > 1:
            min_p = min((p for _, p in priors), default=0.01)
            pass_p = max(1e-6, 0.05 * min_p)
            priors = [(m, (pass_p if m is PASS else p)) for (m, p) in priors]

        # Normalize to a proper distribution.
        total = sum(p for _, p in priors)
        if total <= 0:
            total = float(len(priors))
            priors = [(m, 1.0) for (m, _) in priors]

        for m, p in priors:
            self.Psa[(state_key, self._move_key(m))] = p / total

        # Store legal moves in descending prior order (stable tie-break), but
        # keep PASS last when there are other moves.
        priors.sort(key=lambda mp: (-mp[1], move_sort_key(mp[0])))
        if len(priors) > 1:
            non_pass = [mp for mp in priors if mp[0] is not PASS]
            only_pass = [mp for mp in priors if mp[0] is PASS]
            priors = [*non_pass, *only_pass]
        self._legal_moves[state_key] = [m for (m, _) in priors]
        self._expanded_count[state_key] = 0

    def _ensure_progressive_widening(self, state_key: tuple[object, ...]) -> None:
        # Decide how many moves to consider from this state.
        legal = self._legal_moves.get(state_key)
        if not legal:
            return
        ns = self.Ns[state_key]
        target = int(
            self.progressive_widening_c * ((ns + 1) ** self.progressive_widening_alpha)
        )
        # Ensure the root considers multiple moves before it ever seriously
        # considers passing.
        min_k = (
            6
            if (
                self._root_key is not None
                and state_key == self._root_key
                and len(legal) > 1
            )
            else 1
        )
        target = max(min_k, min(target, len(legal)))
        if self._expanded_count[state_key] < target:
            self._expanded_count[state_key] = target

    def _expanded_moves(self, state_key: tuple[object, ...]) -> list[Move]:
        legal = self._legal_moves.get(state_key, [])
        k = self._expanded_count[state_key]
        return legal[:k]

    def _puct_select_move(self, state_key: tuple[object, ...]) -> Move:
        expanded = self._expanded_moves(state_key)
        if not expanded:
            return PASS

        ns = self.Ns[state_key]
        sqrt_ns = math.sqrt(ns + 1e-9)

        def score(m: Move) -> float:
            edge = (state_key, self._move_key(m))
            nsa = self.Nsa[edge]
            if nsa > 0:
                q_ucb = self.Wsa[edge] / nsa
            else:
                q_ucb = 0.5  # neutral prior value for unseen edges

            q = q_ucb
            if (
                self._root_key is not None
                and state_key == self._root_key
                and self.rave_k > 0
            ):
                n_amaf = self.N_amaf[edge]
                if n_amaf > 0:
                    q_amaf = self.W_amaf[edge] / n_amaf
                    beta = self.rave_k / (self.rave_k + float(nsa))
                    q = (1.0 - beta) * q_ucb + beta * q_amaf
            p = self.Psa.get(
                edge, 1.0 / max(1, len(self._legal_moves.get(state_key, [])))
            )
            u = self.c_puct * p * (sqrt_ns / (1.0 + nsa))
            return q + u

        best_score = max((score(m) for m in expanded), default=float("-inf"))
        best_moves = [m for m in expanded if score(m) == best_score]
        return min(best_moves, key=move_sort_key)

    def _simulate(
        self, game: Game
    ) -> tuple[float, int, list[tuple[int, tuple[str, tuple[int, int]]]], int]:
        "Returns (reward, moves_applied, sim_moves, sim_start_player) from current `game` state."
        n_applied = 0
        sim_moves: list[tuple[int, tuple[str, tuple[int, int]]]] = []
        # Reward is measured from the perspective of the player-to-move at the
        # *start* of this simulation (i.e., at the leaf).
        sim_start_player = game.current_player

        while n_applied < self.max_sim_depth:
            if game.winner is not None:
                reward = 1.0 if game.winner == sim_start_player else 0.0
                return reward, n_applied, sim_moves, sim_start_player

            mover_idx = game.current_player
            player = game.players[mover_idx]
            move = self._rollout_pick_move(game)
            game.do_move(player, move)
            sim_moves.append((mover_idx, self._move_key(move)))
            n_applied += 1

        # Depth limit reached: use a heuristic evaluation as a soft reward.
        # This prevents MCTS from behaving randomly when terminal wins are rare.
        reward = self._heuristic_probability(game, sim_start_player)
        return reward, n_applied, sim_moves, sim_start_player

    def _backpropagate(
        self, path_edges: List[tuple[tuple[object, ...], Move]], reward: float
    ) -> None:
        "Backpropagate along edges; values are from the current-player perspective at each state."
        for state_key, move in reversed(path_edges):
            self.Ns[state_key] += 1
            edge = (state_key, self._move_key(move))
            self.Nsa[edge] += 1
            self.Wsa[edge] += reward
            reward = 1.0 - reward


def _game_key(game: Game) -> tuple[object, ...]:
    # Canonical, hashable description of game state for MCTS nodes.
    sticks = tuple(sorted(s.ordered for s in game.sticks))
    rocks = tuple(sorted(r.c for r in game.rocks))
    return (
        game.winner,
        game.turn_number,
        game.current_player,
        tuple(game.players_scores),
        tuple(game.num_rocks),
        rocks,
        sticks,
    )
