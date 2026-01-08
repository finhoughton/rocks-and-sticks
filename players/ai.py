from __future__ import annotations

import math
import random
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from constants import ALPHABETA_DEPTH, HALF_AREA_COUNTS
from models import PASS, D, Move, MoveKey, calculate_area, calculate_end, move_key

from .base import Player, StateKey, _game_key

if TYPE_CHECKING:
    from game import GameProtocol as Game
    from gnn.encode import EncodedGraph


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
    # cache possible moves per (state_key, player_idx)
    _moves_cache: dict[tuple[StateKey, int], list[Move]] = {}
    # cache encoded graphs per state_key to avoid repeated encoding
    _enc_cache: dict[StateKey, EncodedGraph] = {}

    def __init__(self, player_number: int, use_gnn_eval: bool):
        super().__init__(player_number)
        self.use_gnn_eval: bool = use_gnn_eval

    @abstractmethod
    def get_move(self, game: Game) -> Move:
        raise NotImplementedError("AIPlayer is an abstract base class")

    def _clear_heuristic_caches(self) -> None:
        self._eval_cache.clear()
        self._pot_cache.clear()
        self._sticks_opp_cache.clear()
        self._rock_value_cache.clear()
        self._tactical_cache.clear()
        self._closure_area_cache.clear()
        self._moves_cache.clear()
        self._enc_cache.clear()

    def _eval_game_probability(self, game: Game, perspective_player: Player) -> float:
        value = self._evaluate_position(game, perspective_player)
        if math.isinf(value):
            return 1.0 if value > 0 else 0.0
        return 0.5 + 0.5 * math.tanh(value / 6.0)

    def _evaluate_position(self, game: Game, player: Player) -> float:
        state_key = _game_key(game)
        cache_key = (state_key, player.number, self.use_gnn_eval)
        cached = self._eval_cache.get(cache_key)
        if cached is not None:
            return cached

        if self.use_gnn_eval:
            v = self._evaluate_with_gnn(game, player)
        else:
            v = self._evaluate_position_handcrafted(game, player, state_key)

        self._eval_cache[cache_key] = v
        return v

    def _evaluate_with_gnn(self, game: Game, player: Player) -> float:
        assert self.use_gnn_eval
        from gnn.model import evaluate_encoding
        enc = self._get_encoded_graph(game)
        prob = evaluate_encoding(enc)
        if player.number != game.current_player:
            prob = 1.0 - prob

        prob = min(max(prob, 1e-4), 1.0 - 1e-4)
        logit = math.log(prob / (1.0 - prob))
        logit /= 2.0
        prob = 1.0 / (1.0 + math.exp(-logit))

        return 6.0 * math.atanh((prob - 0.5) * 2.0)

    def _evaluate_position_handcrafted(self, game: Game, player: Player, state_key: StateKey | None = None) -> float:
        if state_key is None:
            state_key = _game_key(game)
        cache_key = (state_key, player.number, self.use_gnn_eval)

        opp_idx = 1 - player.number

        if game.winner == player.number:
            v = float("inf")
        elif game.winner == opp_idx:
            v = float("-inf")
        else:
            my_ts = self._tactical_stats(game, player, state_key)
            opp_ts = self._tactical_stats(game, game.players[opp_idx], state_key)

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

        self._eval_cache[cache_key] = v
        return v
    
    def _tactical_stats(
        self,
        game: Game,
        player: Player,
        state_key: StateKey | None = None,
        eval_cap: int = 32,
        reply_lines: int = 4,
        include_reply: bool = True,
    ) -> TacticalStats:

        if state_key is None:
            state_key = _game_key(game)

        cache_key = (state_key, player.number, include_reply)
        cached = self._tactical_cache.get(cache_key)
        if cached is not None:
            return cached

        player = game.players[player.number]
        # Position-level heuristics that do not depend on sampled stick moves.
        potential_area = self._calculate_potential_area(game, player, state_key)
        stick_opportunities = self._count_stick_opportunities(game, player, state_key)
        rock_value = self._estimate_rock_opportunity_value(game, player, state_key)
        blocking_power = self._calculate_blocking_opponent(game, player)
        all_stick_moves = list(self.search_moves_sticks(game, player))
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
            self._tactical_cache[cache_key] = out
            return out

        before = game.players_scores[player.number]
        max_gain = 0.0
        scoring_count = 0
        winning_count = 0
        bad_closure_count = 0
        gains: list[float] = []

        cap = min(eval_cap, stick_moves_count)
        stick_sample = sorted(all_stick_moves, key=move_key)[:cap] # heapq.smallest is slower for small N

        closure_area_by_key: dict[MoveKey, int | None] = {}
        for mv in stick_sample:
            closure_area_by_key[move_key(mv)] = self._closure_area(game, state_key, mv)

        for mv in stick_sample:
            game.do_move(player.number, mv)
            if game.winner == player.number:
                max_gain = max(max_gain, 999.0)
                scoring_count += 1
                winning_count += 1
                gains.append(999.0)
                game.undo_move()
                continue
            gain = float(game.players_scores[player.number] - before)
            game.undo_move()
            if gain > 0:
                scoring_count += 1
                if gain > max_gain:
                    max_gain = gain
                gains.append(gain)
            else:
                area = closure_area_by_key.get(move_key(mv))
                if area == 1 and not HALF_AREA_COUNTS:
                    bad_closure_count += 1

        gains.sort(reverse=True)
        top3_sum = float(sum(gains[:3]))

        best_reply_gain = 0.0
        if include_reply and reply_lines > 0 and cap > 0:
            opp_idx = 1 - player.number

            def approx_gain(mv: Move) -> float:
                area = closure_area_by_key.get(move_key(mv))
                if area is None:
                    return 0.0
                return self._scored_gain_from_area(area)

            ranked = sorted(
                stick_sample,
                key=lambda mv: (-approx_gain(mv), move_key(mv)),
            )
            for mv in ranked[:reply_lines]:
                game.do_move(player.number, mv)
                if game.winner == player.number:
                    game.undo_move()
                    continue
                opp_ts = self._tactical_stats(
                    game,
                    game.players[opp_idx],
                    state_key=_game_key(game),
                    eval_cap=max(12, eval_cap // 2),
                    reply_lines=0,
                    include_reply=False,
                )
                best_reply_gain = max(best_reply_gain, opp_ts.max_immediate_gain)
                game.undo_move()

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
        self._tactical_cache[cache_key] = out
        return out

    def _get_encoded_graph(self, game: Game):
        key = _game_key(game)
        enc = self._enc_cache.get(key)
        if enc is None:
            from gnn.encode import encode_game_to_graph

            enc = encode_game_to_graph(game)
            self._enc_cache[key] = enc
        return enc

    def _closure_area(self, game: Game, state_key: StateKey, mv: Move) -> int | None:
        key = (state_key, move_key(mv))
        cached = self._closure_area_cache.get(key)
        if cached is not None or key in self._closure_area_cache:
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

        self._closure_area_cache[key] = area
        return area
  
    def _estimate_rock_opportunity_value(self, game: Game, player: Player, state_key: StateKey | None = None, rock_eval_cap: int = 24) -> float:

        if state_key is None:
            state_key = _game_key(game)
        k = (state_key, player_idx := player.number)
        cached = self._rock_value_cache.get(k)
        if cached is not None:
            return cached

        rocks_remaining = game.num_rocks[player_idx]
        if rocks_remaining <= 0:
            self._rock_value_cache[k] = 0.0
            return 0.0

        me = game.players[player_idx]
        opp = game.players[1 - player_idx]

        rock_moves = sorted([
            m
            for m in game.get_possible_moves(me.number)
            if m is not PASS and m.t == "R" and self._rock_is_search_worthy(game, m.c)
        ], key=move_key)

        if not rock_moves:
            self._rock_value_cache[k] = 0.0
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
                area = self._closure_area(game, state_key, Move(p.x, p.y, d.name))
                if area is not None:
                    impact += 0.30 * self._scored_gain_from_area(area)
            return impact

        best_impact = 0.0
        for m in rock_moves[:rock_eval_cap]:
            best_impact = max(best_impact, rock_impact(m.c))

        opportunity = min(3.0, best_impact / 4.0)
        value = float(rocks_remaining) * opportunity
        self._rock_value_cache[k] = value
        return value

    def _calculate_potential_area(self, game: Game, player: Player, state_key: StateKey | None = None) -> float:
        if state_key is None:
            state_key = _game_key(game)
        k = (state_key, player.number)
        v = self._pot_cache.get(k)
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
        self._pot_cache[k] = float(potential_area)
        return float(potential_area)
    
    def _count_stick_opportunities(self, game: Game, player: Player, state_key: StateKey | None = None) -> float:
        if state_key is None:
            state_key = _game_key(game)

        k = (state_key, player.number)
        v = self._sticks_opp_cache.get(k)

        if v is not None:
            return v

        return float(sum(1 for _ in self.search_moves_sticks(game, player)))
    
    def _calculate_blocking_opponent(self, game: Game, player: Player) -> float:
        intersects = game.intersects_stick
        blocked = 0.0
        state_key = _game_key(game)

        for rock in game.rocks:
            p = game.points.get(rock.c)
            if p is None:
                continue
            if p.rocked_by is not player:
                continue
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
                    area = self._closure_area(
                        game, state_key, Move(p.x, p.y, d.name)
                    )
                    if area is not None:
                        blocked += 0.25 * self._scored_gain_from_area(area)

        return float(blocked)

    def search_moves_all(self, game: Game, player: Player) -> Iterator[Move]:
        # all valid moves except rocks that are not search worthy
        state_key = _game_key(game)
        cache_key = (state_key, player.number)
        moves = self._moves_cache.get(cache_key)
        if moves is None:
            moves = list(game.get_possible_moves(player.number))
            self._moves_cache[cache_key] = moves
        for m in moves:
            if m.t == "R" and not self._rock_is_search_worthy(game, m.c):
                continue
            yield m

    def search_moves_sticks(self, game: Game, player: Player) -> Iterator[Move]:
        # Use cached possible moves for this state when available to avoid recomputing
        state_key = _game_key(game)
        cache_key = (state_key, player.number)
        moves = self._moves_cache.get(cache_key)
        if moves is None:
            # fallback to per-point generation
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
            return
        # filter cached moves to stick-only
        for m in moves:
            if m is PASS:
                continue
            if m.t == "R":
                continue
            yield m

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
    def get_move(self, game: "Game") -> Move:
        self._clear_heuristic_caches()
        moves = sorted(game.get_possible_moves(self.number), key=move_key)
        best = moves[0]
        best_v = float("-inf")
        for mv in moves:
            game.do_move(self.number, mv)
            v = self._eval_game_probability(game, self) + random.uniform(-0.04, 0.04)
            game.undo_move()
            if v > best_v:
                best_v = v
                best = mv
        return best


class AlphaBetaPlayer(AIPlayer):
    def __init__(self, player_number: int, depth: int = ALPHABETA_DEPTH, use_gnn: bool = False, pass_penalty: float = 1.2):
        super().__init__(player_number, use_gnn)
        self.depth = depth
        self.use_gnn_eval = use_gnn
        self.pass_penalty = pass_penalty

    def get_move(self, game: "Game") -> Move:
        if game.num_players != 2:
            raise ValueError("Alpha-beta AI is only implemented for 2 players")
        self._clear_heuristic_caches()
        move, _ = self.alpha_beta(game, self.depth, float("-inf"), float("inf"), True)
        return move

    def alpha_beta(self, game: "Game", depth: int, a: float, b: float, maximising: bool) -> tuple[Move, float]:
        if depth == 0 or game.winner is not None:
            return (PASS, self._evaluate_position(game, self))

        best_move = PASS

        if maximising:
            value = float("-inf")
            for move in self.search_moves_all(game, self):
                game.do_move(self.number, move)
                _, v2 = self.alpha_beta(game, depth - 1, a, b, False)
                game.undo_move()
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
            game.do_move(opp.number, move)
            _, v2 = self.alpha_beta(game, depth - 1, a, b, True)
            game.undo_move()
            if move is PASS:
                v2 += self.pass_penalty
            if v2 < value:
                value = v2
                best_move = move
            if value < a:
                break
            b = min(b, value)
        return (best_move, value)
