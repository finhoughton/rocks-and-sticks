from __future__ import annotations

import copy
import math
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, DefaultDict, Iterator, List, cast

from constants import ALPHABETA_DEPTH, HALF_AREA_COUNTS
from models import PASS, D, Move, MoveKey, calculate_area, calculate_end, move_sort_key

if TYPE_CHECKING:
    from game import Game
    from models import Node

StateKey = tuple[object, ...]

def _game_key(game: Game) -> StateKey:
    # Canonical, hashable description of game state for MCTS
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

@contextmanager
def applied_move(game: Game, player: Player, move: Move):
    game.do_move(player, move)
    yield
    game.undo_move()


@contextmanager
def rollback_to(game: Game):
    start_len = len(game.moves)
    yield
    while len(game.moves) > start_len:
        game.undo_move()

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
        # Players are effectively immutable identifiers
        return self

class HumanPlayer(Player):
    def get_move(self, game: Game) -> Move:
        inp = game.wait_for_move_input(f"Player {self.number + 1}, enter move: ")
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
        moves = sorted((m for m in game.get_possible_moves(self) if m is not PASS), key=move_sort_key)
        return self._rng.choice(moves)

class RockBiasedRandomPlayer(RandomPlayer):
    # Random player with a bias toward rock moves to diversify play

    def get_move(self, game: Game) -> Move:
        moves = sorted((m for m in game.get_possible_moves(self) if m is not PASS), key=move_sort_key)
        rock_moves = [m for m in moves if m.t == "R"]
        if rock_moves and self._rng.random() < 0.6:
            return self._rng.choice(rock_moves)
        return self._rng.choice(moves)

@dataclass(frozen=True)
class TacticalStats:
    max_immediate_gain: float
    scoring_move_count: int
    legal_stick_move_count: int
    top3_gain_sum: float
    bad_closure_count: int
    best_reply_gain: float
    winning_move_count: int
    potential_area: float
    stick_opportunities: float
    rock_value: float
    blocking_power: float

class AIPlayer(Player):
    # Per-`get_move` caches (cleared at the start of each search).
    # Cache heuristic evaluations per (state, perspective player).
    _eval_cache: dict[tuple[StateKey, int, bool], float] = {}
    # Tactical caches: (state_key, player_idx, include_reply) -> TacticalStats
    _tactical_cache: dict[tuple[StateKey, int, bool], TacticalStats] = {}
    _pot_cache: dict[tuple[StateKey, int], float] = {}
    _sticks_opp_cache: dict[tuple[StateKey, int], float] = {}
    _rock_value_cache: dict[tuple[StateKey, int], float] = {}
    _closure_area_cache: dict[tuple[StateKey, MoveKey], int | None] = {}

    use_gnn_eval: bool = False

    @abstractmethod
    def get_move(self, game: Game) -> Move:
        raise NotImplementedError("AIPlayer is an abstract base class")

    def _clear_search_caches(self) -> None:
        self._eval_cache.clear()
        self._pot_cache.clear()
        self._sticks_opp_cache.clear()
        self._rock_value_cache.clear()
        self._tactical_cache.clear()
        self._closure_area_cache.clear()

    @classmethod
    def _eval_game_probability(cls, game: Game, perspective_player: Player) -> float:
        value = cls._evaluate_position(game, perspective_player)
        if math.isinf(value):
            return 1.0 if value > 0 else 0.0
        return 0.5 + 0.5 * math.tanh(value / 6.0)

    @classmethod
    def _evaluate_position(cls, game: Game, player: Player) -> float:
        state_key = _game_key(game)
        cache_key = (state_key, player.number, cls.use_gnn_eval)
        cached = cls._eval_cache.get(cache_key)
        if cached is not None:
            return cached

        if cls.use_gnn_eval:
            v = cls._evaluate_with_gnn(game, player)
        else:
            v = cls._evaluate_position_handcrafted(game, player, state_key)

        cls._eval_cache[cache_key] = v
        return v

    @classmethod
    def _evaluate_with_gnn(cls, game: Game, player: Player) -> float:
        from gnn.model import evaluate_game
        prob = float(evaluate_game(game))
        if player.number != game.current_player:
            prob = 1.0 - prob

        prob = min(max(prob, 1e-4), 1.0 - 1e-4)
        logit = math.log(prob / (1.0 - prob))
        logit /= 2.0
        prob = 1.0 / (1.0 + math.exp(-logit))

        return 6.0 * math.atanh((prob - 0.5) * 2.0)

    @classmethod
    def _evaluate_position_handcrafted(cls, game: Game, player: Player, state_key: StateKey | None = None) -> float:
        if state_key is None:
            state_key = _game_key(game)
        cache_key = (state_key, player.number, cls.use_gnn_eval)

        opp_idx = 1 - player.number

        if game.winner == player.number:
            v = float("inf")
        elif game.winner == opp_idx:
            v = float("-inf")
        else:
            my_ts = cls._tactical_stats(game, player, state_key)
            opp_ts = cls._tactical_stats(game, game.players[opp_idx], state_key)

            turn_me = game.current_player == player.number
            (w_me, w_opp) = (1.0, 0.4) if turn_me else (0.4, 1.0)

            v = (
                1.2 * my_ts.blocking_power
                + 0.7 * my_ts.stick_opportunities
                + 1.3 * my_ts.potential_area
                - 2.0 * opp_ts.potential_area
                + 1.8 * (my_ts.rock_value - opp_ts.rock_value)
                + 1.2 * (my_ts.max_immediate_gain - opp_ts.max_immediate_gain)
                + 0.3 * (my_ts.top3_gain_sum - opp_ts.top3_gain_sum)
                + 0.3 * float(my_ts.scoring_move_count - opp_ts.scoring_move_count)
                + 0.1 * float(my_ts.legal_stick_move_count - opp_ts.legal_stick_move_count)
                - 0.5 * w_me * my_ts.best_reply_gain
                + 0.5 * w_opp * opp_ts.best_reply_gain
                - 0.1 * float(my_ts.bad_closure_count - opp_ts.bad_closure_count)
            )

        cls._eval_cache[cache_key] = v
        return v
    
    @classmethod
    def _tactical_stats(
        cls,
        game: Game,
        player: Player,
        state_key: StateKey | None = None,
        eval_cap: int = 32,
        reply_lines: int = 4,
        include_reply: bool = True,
    ) -> TacticalStats:
        """Cheap tactical summary for `player_idx` (cached per state).

        Includes:
        - immediate threats (max gain, number of scoring moves)
        - threat distribution (sum of top 3 gains)
        - "bad" cycle closures (area==1 when HALF_AREA_COUNTS is False)
        - shallow risk: opponent's best immediate scoring reply after our best few moves
        """

        if state_key is None:
            state_key = _game_key(game)

        cache_key = (state_key, player.number, include_reply)
        cached = cls._tactical_cache.get(cache_key)
        if cached is not None:
            return cached

        player = game.players[player.number]
        # Position-level heuristics that do not depend on sampled stick moves.
        potential_area = cls._calculate_potential_area(game, player, state_key)
        stick_opportunities = cls._count_stick_opportunities(game, player, state_key)
        rock_value = cls._estimate_rock_opportunity_value(game, player, state_key)
        blocking_power = cls._calculate_blocking_opponent(game, player)
        all_stick_moves = list(cls.search_moves_sticks(game, player))
        stick_moves_count = len(all_stick_moves)
        if stick_moves_count == 0:
            out = TacticalStats(
                max_immediate_gain=0.0,
                scoring_move_count=0,
                legal_stick_move_count=0,
                top3_gain_sum=0.0,
                bad_closure_count=0,
                best_reply_gain=0.0,
                winning_move_count=0,
                potential_area=float(potential_area),
                stick_opportunities=float(stick_opportunities),
                rock_value=float(rock_value),
                blocking_power=float(blocking_power),
            )
            cls._tactical_cache[cache_key] = out
            return out

        before = game.players_scores[player.number]
        max_gain = 0.0
        scoring_count = 0
        winning_count = 0
        bad_closure_count = 0
        gains: list[float] = []

        cap = min(eval_cap, stick_moves_count)
        stick_sample = sorted(all_stick_moves, key=move_sort_key)[:cap] # heapq.smallest is slower for small N

        closure_area_by_key: dict[MoveKey, int | None] = {}
        for mv in stick_sample:
            closure_area_by_key[move_sort_key(mv)] = cls._closure_area(game, state_key, mv)

        for mv in stick_sample:
            with applied_move(game, player, mv):
                # If this move wins outright, it is the strongest possible threat.
                if game.winner == player.number:
                    max_gain = max(max_gain, 999.0)
                    scoring_count += 1
                    winning_count += 1
                    gains.append(999.0)
                    continue
                gain = float(game.players_scores[player.number] - before)
            if gain > 0:
                scoring_count += 1
                if gain > max_gain:
                    max_gain = gain
                gains.append(gain)
            else:
                area = closure_area_by_key.get(move_sort_key(mv))
                if area == 1 and not HALF_AREA_COUNTS:
                    bad_closure_count += 1

        gains.sort(reverse=True)
        top3_sum = float(sum(gains[:3]))

        best_reply_gain = 0.0
        if include_reply and reply_lines > 0 and cap > 0:
            opp_idx = 1 - player.number

            def approx_gain(mv: Move) -> float:
                area = closure_area_by_key.get(move_sort_key(mv))
                if area is None:
                    return 0.0
                return cls._scored_gain_from_area(area)

            ranked = sorted(
                stick_sample,
                key=lambda mv: (-approx_gain(mv), move_sort_key(mv)),
            )
            for mv in ranked[:reply_lines]:
                with applied_move(game, player, mv):
                    if game.winner == player.number:
                        continue
                    opp_ts = cls._tactical_stats(
                        game,
                        game.players[opp_idx],
                        state_key=_game_key(game),
                        eval_cap=max(12, eval_cap // 2),
                        reply_lines=0,
                        include_reply=False,
                    )
                    best_reply_gain = max(best_reply_gain, opp_ts.max_immediate_gain)

        out = TacticalStats(
            max_immediate_gain=float(max_gain),
            scoring_move_count=int(scoring_count),
            legal_stick_move_count=int(stick_moves_count),
            top3_gain_sum=float(top3_sum),
            bad_closure_count=int(bad_closure_count),
            best_reply_gain=float(best_reply_gain),
            winning_move_count=int(winning_count),
            potential_area=float(potential_area),
            stick_opportunities=float(stick_opportunities),
            rock_value=float(rock_value),
            blocking_power=float(blocking_power),
        )
        cls._tactical_cache[cache_key] = out
        return out

    @classmethod
    def _closure_area(cls, game: Game, state_key: StateKey, mv: Move) -> int | None:
        key = (state_key, move_sort_key(mv))
        cached = cls._closure_area_cache.get(key)
        if cached is not None or key in cls._closure_area_cache:
            return cached

        start = game.points.get(mv.c)

        if start is None:
            area = None
        elif (end := start.neighbours[D[mv.t].as_int]) is None:
            area = None
        elif not (path := start.find_path(end, set())):
            area = None
        else:
            area = calculate_area(path)

        cls._closure_area_cache[key] = area
        return area
  
    @classmethod
    def _estimate_rock_opportunity_value(cls, game: Game, player: Player, state_key: StateKey | None = None, rock_eval_cap: int = 24) -> float:
        """Value remaining rocks by how impactful the *best* available rock is."""

        if state_key is None:
            state_key = _game_key(game)
        k = (state_key, player_idx := player.number)
        cached = cls._rock_value_cache.get(k)
        if cached is not None:
            return cached

        rocks_remaining = game.num_rocks[player_idx]
        if rocks_remaining <= 0:
            cls._rock_value_cache[k] = 0.0
            return 0.0

        me = game.players[player_idx]
        opp = game.players[1 - player_idx]

        rock_moves = sorted([
            m
            for m in game.get_possible_moves(me)
            if m is not PASS and m.t == "R" and cls._rock_is_search_worthy(game, m.c)
        ], key=move_sort_key)

        if not rock_moves:
            cls._rock_value_cache[k] = 0.0
            return 0.0

        def rock_impact(c: tuple[int, int]) -> float:
            p = game.points.get(c)
            if p is None or not p.connected:
                return 0.0
            # Only matters if the opponent could otherwise start from here.
            if not opp.can_place(p):
                return 0.0

            impact = 0.0
            for d in p.empty_directions:
                if game.intersects_stick(p.c, d):
                    continue
                impact += 1.0
                area = cls._closure_area(game, state_key, Move(p.x, p.y, d.name))
                if area is not None:
                    impact += 0.30 * cls._scored_gain_from_area(area)
            return impact

        best_impact = 0.0
        for m in rock_moves[:rock_eval_cap]:
            best_impact = max(best_impact, rock_impact(m.c))

        opportunity = min(3.0, best_impact / 4.0)
        value = float(rocks_remaining) * opportunity
        cls._rock_value_cache[k] = value
        return value
    
    @classmethod
    def _calculate_potential_area(cls, game: Game, player: Player, state_key: StateKey | None = None) -> float:
        if state_key is None:
            state_key = _game_key(game)
        k = (state_key, player.number)
        v = cls._pot_cache.get(k)
        if v is not None:
            return v
        potential_area = 0
        points = game.points
        for point in game.connected_points:
            if not player.can_place(point):
                continue
            x, y = point.c
            for d in point.empty_directions:
                dx, dy = d.delta
                end = points.get((x + dx, y + dy))
                if end is not None and player.can_place(end):
                    potential_area += 1
        cls._pot_cache[k] = float(potential_area)
        return float(potential_area)
    
    @classmethod
    def _count_stick_opportunities(cls, game: Game, player: Player, state_key: StateKey | None = None) -> float:
        if state_key is None:
            state_key = _game_key(game)

        k = (state_key, player.number)
        v = cls._sticks_opp_cache.get(k)

        if v is not None:
            return v

        return float(sum(1 for _ in cls.search_moves_sticks(game, player)))
    
    @classmethod
    def _calculate_blocking_opponent(cls, game: Game, player: Player) -> float:
        # Measure how many legal stick moves are denied due to `player`'s rocks.
        intersects = game.intersects_stick
        blocked = 0.0
        state_key = _game_key(game)

        # `game.rocks` is small (N_ROCKS), so iterating it is cheap.
        for rock in game.rocks:
            p = game.points.get(rock.c)
            if p is None:
                continue
            if p.rocked_by is not player:
                continue
            # Only connected vertices can be used as stick starts.
            if not p.connected:
                continue
            if game.coord_in_claimed_region(p.c):
                continue
            for d in p.empty_directions:
                if not intersects(p.c, d):
                    end_c = calculate_end(p.c, d)
                    if game.coord_in_claimed_region(end_c):
                        continue
                    blocked += 1.0
                    area = cls._closure_area(
                        game, state_key, Move(p.x, p.y, d.name)
                    )
                    if area is not None:
                        blocked += 0.25 * cls._scored_gain_from_area(area)

        return float(blocked)

    @classmethod
    def search_moves_all(cls, game: Game, player: Player) -> Iterator[Move]:
        """all valid move except rocks that are not search worthy"""
        for m in game.get_possible_moves(player):
            if m.t == "R" and not cls._rock_is_search_worthy(game, m.c):
                continue
            yield m

    @staticmethod
    def search_moves_sticks(game: Game, player: Player) -> Iterator[Move]:
        """all valid stick moves"""
        for point in game.connected_points:
            if not player.can_place(point):
                continue
            if game.coord_in_claimed_region(point.c):
                continue
            for d in point.empty_directions:
                if not game.intersects_stick(point.c, d):
                    end_c = calculate_end(point.c, d)
                    if game.coord_in_claimed_region(end_c):
                        continue
                    yield Move(point.x, point.y, d.name)

    @staticmethod
    def _scored_gain_from_area(area: int) -> float:
        if HALF_AREA_COUNTS or area != 1:
            return float(area)
        return 0.0

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
        for d in D:
            dx, dy = d.delta
            if (x + dx, y + dy) in rock_coords:
                adjacent_rocks += 1
                if adjacent_rocks >= 2:
                    return True
        return False

class OnePlyGreedyPlayer(AIPlayer):
    def get_move(self, game: Game) -> Move:
        self._clear_search_caches()
        moves = sorted(game.get_possible_moves(self), key=move_sort_key)
        best = moves[0]
        best_v = float("-inf")
        for mv in moves:
            with applied_move(game, self, mv):
                v = self._eval_game_probability(game, self) + random.uniform(-0.04, 0.04)
            if v > best_v:
                best_v = v
                best = mv
        return best

class AlphaBetaPlayer(AIPlayer):
    def __init__(self, player_number: int, depth: int = ALPHABETA_DEPTH, use_gnn: bool = False, pass_penalty: float = 0.75):
        super().__init__(player_number)
        self.depth = depth
        self.use_gnn_eval = use_gnn
        self.pass_penalty = pass_penalty

    def get_move(self, game: Game) -> Move:
        if game.num_players != 2:
            raise ValueError("Alpha-beta AI is only implemented for 2 players")
        self._clear_search_caches()
        move, _ = self.alpha_beta(game, self.depth, float("-inf"), float("inf"), True)
        return move

    def alpha_beta(self, game: Game, depth: int, a: float, b: float, maximising: bool) -> tuple[Move, float]:
        if depth == 0 or game.winner is not None:
            return (PASS, self._evaluate_position(game, self))

        best_move = PASS

        if maximising:
            value = float("-inf")
            for move in self.search_moves_all(game, self):
                with applied_move(game, self, move):
                    _, v2 = self.alpha_beta(game, depth - 1, a, b, False)
                if move is PASS:
                    # Mildly demote PASS so it's only chosen when it truly out-values other moves.
                    v2 -= self.pass_penalty
                if v2 > value:
                    value = v2
                    best_move = move
                if value > b:
                    break
                a = max(a, value)
            return (best_move, value)
        value = float("inf")
        opp = game.players[1 - self.number]
        for move in self.search_moves_all(game, opp):
            with applied_move(game, opp, move):
                _, v2 = self.alpha_beta(game, depth - 1, a, b, True)
            if move is PASS:
                v2 += self.pass_penalty
            if v2 < value:
                value = v2
                best_move = move
            if value < a:
                break
            b = min(b, value)
        return (best_move, value)

# based on https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
class MCTSPlayer(AIPlayer):
    "Monte Carlo tree searcher. First rollout the tree then choose a move."

    def __init__(
        self,
        player_number: int,
        use_gnn: bool = False,
        exploration_weight: float = 1.0,
        seed: int | None = None,
        n_rollouts: int = 1000,
        max_sim_depth: int = 20,
        time_limit: float | None = None,
        tactical_root_limit: int = 20,
        tactical_branch_limit: int = 8,
        progressive_widening_c: float = 1.6,
        progressive_widening_alpha: float = 0.55,
        prior_eval_cap: int = 48,
        rave_k: float = 250.0,
        rock_prior_bonus_connected: float = 1.5,
        rock_prior_bonus_disconnected: float = 0.06,
        rock_rollout_bonus_connected: float = 1.0,
        rock_rollout_bonus_disconnected: float = 0.02,
        stick_between_opp_rocks_bonus: float = 0.4,
    ):
        super().__init__(player_number)
        self.use_gnn_eval = use_gnn
        # Transposition-table MCTS:
        # - Stats are stored per state key, and per (state, move) edge.
        # - This merges equivalent positions reached via different paths.
        self.Ns: DefaultDict[StateKey, int] = defaultdict(int)
        self.Nsa: DefaultDict[tuple[StateKey, MoveKey], int] = defaultdict(int)
        self.Wsa: DefaultDict[tuple[StateKey, MoveKey], float] = defaultdict(float)
        self.Psa: dict[tuple[StateKey, MoveKey], float] = {}
        self._legal_moves: dict[StateKey, list[Move]] = {}
        self._expanded_count: DefaultDict[StateKey, int] = defaultdict(int)

        # Root-only RAVE/AMAF stats.
        self.N_amaf: DefaultDict[tuple[StateKey, MoveKey], int] = defaultdict(int)
        self.W_amaf: DefaultDict[tuple[StateKey, MoveKey], float] = defaultdict(float)
        self.rave_k = rave_k
        self._root_key: StateKey | None = None

        # Interpreting exploration_weight as the PUCT exploration constant.
        self.c_puct: float = exploration_weight
        self._rng = random.Random(seed)
        self.n_rollouts = n_rollouts
        self.max_sim_depth = max_sim_depth
        self.time_limit = time_limit
        self.tactical_root_limit = tactical_root_limit
        self.tactical_branch_limit = tactical_branch_limit
        self.progressive_widening_c = progressive_widening_c
        self.progressive_widening_alpha = progressive_widening_alpha
        self.prior_eval_cap = prior_eval_cap
        self.rock_prior_bonus_connected = rock_prior_bonus_connected
        self.rock_prior_bonus_disconnected = rock_prior_bonus_disconnected
        self.rock_rollout_bonus_connected = rock_rollout_bonus_connected
        self.rock_rollout_bonus_disconnected = rock_rollout_bonus_disconnected
        self.stick_between_opp_rocks_bonus = stick_between_opp_rocks_bonus
        self.last_rollouts: int = 0

    def _rock_bonus_for_cell(self, game: Game, c: tuple[int, int], connected_bonus: float, disconnected_bonus: float,) -> float:
        p = game.points.get(c)
        if p is not None and p.connected:
            return connected_bonus
        return disconnected_bonus

    def _stick_between_opp_rocks(self, game: Game, player: Player, mv: Move) -> bool:
        if mv.t not in D._member_names_:
            return False
        start = game.points.get(mv.c)
        if start is None or start.rocked_by is None:
            return False
        opp = game.players[1 - player.number]
        if start.rocked_by is not opp:
            return False
        end_c = calculate_end(mv.c, D[mv.t])
        end = game.points.get(end_c)
        if end is None or end.rocked_by is not opp:
            return False
        return True
    
    def score_after(self, working_game: Game, p: Player, mv: Move) -> float:
        with applied_move(working_game, p, mv):
            return self._eval_game_probability(working_game, p)

    def _rollout_pick_move(self, game: Game) -> Move:
        """Pick a rollout move: mostly greedy by heuristic, occasionally random."""
        player = game.players[game.current_player]
        moves = list(self.search_moves_all(game, player))
        if len(moves) <= 1:
            return moves[0]

        # Exploration inside rollout.
        if self._rng.random() < 0.20:
            return self._rng.choice(moves)

        # Evaluate a subset to keep rollouts fast.
        if len(moves) > 9:
            candidates = self._rng.sample(moves, 9)
        else:
            candidates = moves
        candidates = sorted(candidates, key=move_sort_key)

        best_move = candidates[0]
        best_score = float("-inf")
        for m in candidates:
            s = self.score_after(game, player, m)
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

    def allows_forced_loss_next_round(self, my_move: Move, working_game: Game, player: Player) -> bool:
        """True if opponent has a reply such that every response loses (2-ply)."""

        # Cache move orderings per (state, player) to avoid recomputing scores
        # while this routine mutates and rolls back `working_game`.
        move_order_cache: dict[tuple[StateKey, int], list[tuple[float, MoveKey, Move]]] = {}

        def top_k_by_heuristic(p: Player) -> list[Move]:
            state_key = _game_key(working_game)
            cache_key = (state_key, p.number)

            scored = move_order_cache.get(cache_key)
            if scored is None:
                moves = list(self.search_moves_sticks(working_game, p))
                scored = [(self.score_after(working_game, p, mv), move_sort_key(mv), mv) for mv in moves]
                move_order_cache[cache_key] = scored

            k = self.tactical_branch_limit
            if k <= 0 or len(scored) <= k:
                return [mv for (_s, _k, mv) in sorted(scored, key=lambda t: t[1])]

            top = sorted(scored, reverse=True)[:k]
            return [mv for (_s, _k, mv) in top]

        with applied_move(working_game, player, my_move):
            if working_game.winner is not None and working_game.winner != self.number:
                return True

            opp = working_game.players[working_game.current_player]
            opp_moves = top_k_by_heuristic(opp)

            for opp_move in opp_moves:
                with applied_move(working_game, opp, opp_move):
                    if working_game.winner == opp.number:
                        return True

                    me = working_game.players[working_game.current_player]
                    responses = top_k_by_heuristic(me)

                    defended = False
                    for response in responses:
                        with applied_move(working_game, me, response):
                            if working_game.winner != opp.number:
                                defended = True
                                break

                    if not defended:
                        return True

            return False

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
            self.search_moves_all(working_game, player), key=move_sort_key
        )

        safe_root_moves: list[Move] = []
        for move in root_moves:
            with applied_move(working_game, player, move):
                if working_game.winner == self.number:
                    return move
                if working_game.winner is None:
                    safe_root_moves.append(move)


        # Prefer moves that do NOT allow a forced loss next round.
        allowed_root_moves = safe_root_moves
        start_time = time.perf_counter()
        deadline = start_time + self.time_limit if self.time_limit is not None else None
        if safe_root_moves:
            to_check = safe_root_moves[: self.tactical_root_limit]
            non_forced_loss = [m for m in to_check if not self.allows_forced_loss_next_round(m, working_game, player)]
            if non_forced_loss:
                allowed_root_moves = non_forced_loss
        rollouts_done = 0
        while True:
            if rollouts_done >= self.n_rollouts:
                break
            now = time.perf_counter()
            if deadline is not None and now >= deadline:
                break
            self.do_rollout(root_key, working_game)
            rollouts_done += 1

        self.last_rollouts = rollouts_done

        print(f"rollouts = {rollouts_done} for player {self.number}. time taken: {time.perf_counter() - start_time:.2f}s")

        best_move = self.choose(root_key, working_game, allowed_root_moves=allowed_root_moves)

        # Final tactical safety pass: avoid one-move blunders where our move
        # immediately hands the opponent the win at end-of-round, or where the
        # opponent has a forcing line next round.
        def visits(m: Move) -> int:
            return self.Nsa[(root_key, move_sort_key(m))]

        ranked_moves = sorted(allowed_root_moves if allowed_root_moves else root_moves, key=lambda m: (-visits(m), move_sort_key(m)))

        safety_limit = min(len(ranked_moves), max(self.tactical_root_limit, 40))
        for m in ranked_moves[:safety_limit]:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            with applied_move(working_game, player, m):
                if working_game.winner == self.number:
                    return m
                if working_game.winner is not None and working_game.winner != self.number:
                    continue

            if self.allows_forced_loss_next_round(m, working_game, player):
                continue
            return m
        

        return best_move

    def choose(self, root_key: StateKey, game: Game, allowed_root_moves: list[Move] | None = None) -> Move:
        "Choose the best move from the root using search statistics."
        if root_key[0] is not None:
            return PASS

        self._ensure_state_initialized(root_key, game)
        legal = self._legal_moves.get(root_key, [])
        if not legal:
            return PASS

        candidates = legal
        if allowed_root_moves:
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
            return self.Nsa[(root_key, move_sort_key(m))]

        max_visits = max((visits(m) for m in candidates), default=0)
        best = [m for m in candidates if visits(m) == max_visits]
        return min(best, key=move_sort_key)

    def do_rollout(self, root_key: StateKey, game: Game) -> None:
        "Make the tree one layer better. (Train for one iteration.)"
        with rollback_to(game):
            path_edges, _n_applied = self._select(root_key, game)
            reward, _n_sim_applied, sim_moves, sim_start_player = self._simulate(game)

            # Root-only AMAF/RAVE update: any move played by the root player
            # later in the rollout is treated as if it were played first.
            root_player_idx = cast(int, root_key[2])
            root_played: set[MoveKey] = set()
            for state_key, move in path_edges:
                mover = cast(int, state_key[2])
                if mover == root_player_idx:
                    root_played.add(move_sort_key(move))
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

    def _select(self, root_key: StateKey, game: Game) -> tuple[List[tuple[StateKey, Move]], int]:
        "Select a path using PUCT + progressive widening, mutating `game` along the path."
        path_edges: List[tuple[StateKey, Move]] = []
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
            if self.Nsa[(state_key, move_sort_key(move))] == 0:
                return path_edges, n_applied

            state_key = _game_key(game)

    def _ensure_state_initialized(self, state_key: StateKey, game: Game) -> None:
        if state_key in self._legal_moves:
            return
        if state_key[0] is not None:
            self._legal_moves[state_key] = []
            self._expanded_count[state_key] = 0
            return

        player = game.players[game.current_player]
        moves = sorted(self.search_moves_all(game, player), key=move_sort_key)

        # Compute heuristic priors for a capped subset to keep expansion cheap.
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
                p = self._eval_game_probability(game, player)
            if m.t == "R":
                p = min(0.999, p + self._rock_bonus_for_cell(game, m.c, self.rock_prior_bonus_connected, self.rock_prior_bonus_disconnected))
            elif self._stick_between_opp_rocks(game, player, m):
                p = min(0.999, p + self.stick_between_opp_rocks_bonus)
            priors.append((m, float(p)))

        # For non-evaluated moves, assign a small prior so they still become
        # available via progressive widening.
        if len(eval_moves) < len(moves):
            min_p = min((p for _, p in priors), default=0.01)
            floor_p = max(0.005, 0.25 * min_p)
            for m in moves[len(eval_moves) :]:
                p = floor_p
                if m.t == "R":
                    p = min(0.999, p + self._rock_bonus_for_cell(game, m.c, self.rock_prior_bonus_connected, self.rock_prior_bonus_disconnected))
                elif self._stick_between_opp_rocks(game, player, m):
                    p = min(0.999, p + self.stick_between_opp_rocks_bonus)
                priors.append((m, p))

        # PASS is always legal but is rarely correct; give it a very small
        # prior so it won't dominate progressive widening/PUCT unless search
        # evidence makes it clearly best.
        if len(moves) > 1:
            min_p = min((p for _, p in priors), default=0.01)
            pass_p = max(1e-6, 0.05 * min_p)
            priors = [(m, (pass_p if m is PASS else p)) for (m, p) in priors]

        total = sum(p for _, p in priors)
        if total <= 0:
            total = float(len(priors))
            priors = [(m, 1.0) for (m, _) in priors]

        for m, p in priors:
            self.Psa[(state_key, move_sort_key(m))] = p / total

        # Store legal moves in descending prior order (stable tie-break), but
        # keep PASS last when there are other moves.
        priors.sort(key=lambda mp: (-mp[1], move_sort_key(mp[0])))
        if len(priors) > 1:
            non_pass = [mp for mp in priors if mp[0] is not PASS]
            only_pass = [mp for mp in priors if mp[0] is PASS]
            priors = [*non_pass, *only_pass]
        self._legal_moves[state_key] = [m for (m, _) in priors]
        self._expanded_count[state_key] = 0

    def _ensure_progressive_widening(self, state_key: StateKey) -> None:
        # Decide how many moves to consider from this state.
        legal = self._legal_moves.get(state_key)
        if not legal:
            return
        ns = self.Ns[state_key]
        target = int(self.progressive_widening_c * ((ns + 1) ** self.progressive_widening_alpha))
        # Ensure the root considers multiple moves before it ever seriously
        # considers passing.

        if self._root_key is not None and state_key == self._root_key and len(legal) > 1:
            min_k = 6
        else:
            min_k = 1

        target = max(min_k, min(target, len(legal)))
        if self._expanded_count[state_key] < target:
            self._expanded_count[state_key] = target

    def _expanded_moves(self, state_key: StateKey) -> list[Move]:
        legal = self._legal_moves.get(state_key, [])
        k = self._expanded_count[state_key]
        return legal[:k]

    def _puct_select_move(self, state_key: StateKey) -> Move:
        expanded = self._expanded_moves(state_key)
        if not expanded:
            return PASS

        ns = self.Ns[state_key]
        sqrt_ns = math.sqrt(ns + 1e-9)

        def score(m: Move) -> float:
            edge = (state_key, move_sort_key(m))
            nsa = self.Nsa[edge]
            if nsa > 0:
                q_ucb = self.Wsa[edge] / nsa
            else:
                q_ucb = 0.5  # neutral prior value for unseen edges

            q = q_ucb
            if self._root_key is not None and state_key == self._root_key and self.rave_k > 0:
                n_amaf = self.N_amaf[edge]
                if n_amaf > 0:
                    q_amaf = self.W_amaf[edge] / n_amaf
                    beta = self.rave_k / (self.rave_k + float(nsa))
                    q = (1.0 - beta) * q_ucb + beta * q_amaf
            p = self.Psa.get(edge, 1.0 / max(1, len(self._legal_moves.get(state_key, []))))
            u = self.c_puct * p * (sqrt_ns / (1.0 + nsa))
            return q + u

        best_score = max((score(m) for m in expanded), default=float("-inf"))
        best_moves = [m for m in expanded if score(m) == best_score]
        return min(best_moves, key=move_sort_key)

    def _simulate(self, game: Game) -> tuple[float, int, list[tuple[int, MoveKey]], int]:
        "Returns (reward, moves_applied, sim_moves, sim_start_player) from current `game` state."
        n_applied = 0
        sim_moves: list[tuple[int, MoveKey]] = []
        # Reward is measured from the perspective of the player-to-move at the
        # *start* of this simulation (i.e., at the leaf)
        sim_start_player = game.current_player

        while n_applied < self.max_sim_depth:
            if game.winner is not None:
                reward = 1.0 if game.winner == sim_start_player else 0.0
                return reward, n_applied, sim_moves, sim_start_player

            mover_idx = game.current_player
            player = game.players[mover_idx]
            move = self._rollout_pick_move(game)
            game.do_move(player, move)
            sim_moves.append((mover_idx, move_sort_key(move)))
            n_applied += 1

        # Depth limit reached: use a heuristic evaluation as a soft reward.
        # This prevents MCTS from behaving weirdly when cut off.
        reward = self._eval_game_probability(game, game.players[sim_start_player])
        return reward, n_applied, sim_moves, sim_start_player

    def _backpropagate(self, path_edges: List[tuple[StateKey, Move]], reward: float) -> None:
        "Backpropagate along edges; values are from the current-player perspective at each state."
        for state_key, move in reversed(path_edges):
            self.Ns[state_key] += 1
            edge = (state_key, move_sort_key(move))
            self.Nsa[edge] += 1
            self.Wsa[edge] += reward
            reward = 1.0 - reward
