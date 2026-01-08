from __future__ import annotations

import math
import os
import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING, DefaultDict, List, cast

from models import PASS, D, Move, MoveKey, calculate_end, move_key

from .ai import AIPlayer
from .base import Player, StateKey, _game_key, applied_move, rollback_to

if TYPE_CHECKING:
    from game import GameProtocol as Game


# based on https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1
class MCTSPlayer(AIPlayer):
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
        check_forced_losses: bool = True,
    ):
        super().__init__(player_number, use_gnn)
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
        self.check_forced_losses = check_forced_losses
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


    def get_move(self, game: Game, reuse_tree: bool = False) -> Move:
        # Clear only the per-search game heuristics cache inherited from AIPlayer
        self._clear_heuristic_caches()
        if not reuse_tree:
            # Fresh search: drop all accumulated tree statistics
            self.Ns.clear()
            self.Nsa.clear()
            self.Wsa.clear()
            self.Psa.clear()
            self._legal_moves.clear()
            self._expanded_count.clear()
            self.N_amaf.clear()
            self.W_amaf.clear()
            self._root_key = None

        working_game = game # could deepcopy, but this is faster
        root_key = _game_key(working_game)
        self._root_key = root_key

        player = working_game.players[working_game.current_player]
        root_moves = sorted(
            self.search_moves_all(working_game, player), key=move_key
        )

        safe_root_moves: list[Move] = []
        for move in root_moves:
            working_game.do_move(player.number, move)
            if working_game.winner == self.number:
                working_game.undo_move()
                return move
            if working_game.winner is None:
                safe_root_moves.append(move)
            working_game.undo_move()

        # Prefer moves that do NOT allow a forced loss next round.
        allowed_root_moves = safe_root_moves
        start_time = time.perf_counter()
        deadline = start_time + self.time_limit if self.time_limit is not None else None

        if self.check_forced_losses:
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

        if os.environ.get("PYTEST_CURRENT_TEST") is None:
            print(f"rollouts = {rollouts_done} for player {self.number}. time taken: {time.perf_counter() - start_time:.2f}s")

        best_move = self.choose(root_key, working_game, allowed_root_moves=allowed_root_moves)

        def visits(m: Move) -> int:
            return self.Nsa[(root_key, move_key(m))]

        ranked_moves = sorted(allowed_root_moves if allowed_root_moves else root_moves, key=lambda m: (-visits(m), move_key(m)))

        safety_limit = min(len(ranked_moves), max(self.tactical_root_limit, 40))
        for m in ranked_moves[:safety_limit]:
            if deadline is not None and time.perf_counter() >= deadline:
                break
            working_game.do_move(player.number, m)
            if working_game.winner == self.number:
                working_game.undo_move()
                return m
            if working_game.winner is not None and working_game.winner != self.number:
                working_game.undo_move()
                continue
            working_game.undo_move()

            if self.allows_forced_loss_next_round(m, working_game, player, root_key=root_key):
                continue
            return m
        
        return best_move

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
        working_game.do_move(p.number, mv)
        score = self._eval_game_probability(working_game, p)
        working_game.undo_move()
        return score

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
        candidates = sorted(candidates, key=move_key)

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

    def allows_forced_loss_next_round(self, my_move: Move, working_game: Game, player: Player, *, root_key=None) -> bool:
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
                scored = [(self.score_after(working_game, p, mv), move_key(mv), mv) for mv in moves]
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

    def advance_root(self, move: Move, game: Game) -> None:
        new_key = _game_key(game)
        self._root_key = new_key

    def prune_tables(self, max_states: int) -> None:
        """Prune internal transposition tables approximately to `max_states` keys."""
        if max_states <= 0:
            return
        cur = len(self.Ns)
        if cur <= max_states:
            return

        # remove lowest-visited states
        items = sorted(self.Ns.items(), key=lambda kv: kv[1])
        remove_count = cur - max_states
        to_remove = {state_key for (state_key, _ns) in items[:remove_count]}

        for state_key in to_remove:
            self.Ns.pop(state_key, None)
            self._legal_moves.pop(state_key, None)
            self._expanded_count.pop(state_key, None)

        for e in list(self.Nsa.keys()):
            if e[0] in to_remove:
                self.Nsa.pop(e, None)
        for e in list(self.Wsa.keys()):
            if e[0] in to_remove:
                self.Wsa.pop(e, None)
        for e in list(self.Psa.keys()):
            if e[0] in to_remove:
                self.Psa.pop(e, None)
        for e in list(self.N_amaf.keys()):
            if e[0] in to_remove:
                self.N_amaf.pop(e, None)
        for e in list(self.W_amaf.keys()):
            if e[0] in to_remove:
                self.W_amaf.pop(e, None)

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
            game.do_move(player.number, m)
            if game.winner == self.number:
                game.undo_move()
                return m
            game.undo_move()

        # Robust choice: pick the most visited move (stable tie-break).
        def visits(m: Move) -> int:
            return self.Nsa[(root_key, move_key(m))]

        max_visits = max((visits(m) for m in candidates), default=0)
        best = [m for m in candidates if visits(m) == max_visits]
        return min(best, key=move_key)

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
                    root_played.add(move_key(move))
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
            game.do_move(player.number, move)
            n_applied += 1
            path_edges.append((state_key, move))

            # Stop after taking a previously-unvisited edge.
            if self.Nsa[(state_key, move_key(move))] == 0:
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
        moves = sorted(self.search_moves_all(game, player), key=move_key)

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

        from gnn.encode import encode_game_to_graph
        eval_encodings: list = []
        eval_moves_order: list[Move] = []
        for m in eval_moves:
            game.do_move(player.number, m)
            eval_encodings.append(encode_game_to_graph(game))
            game.undo_move()
            eval_moves_order.append(m)

        probs: list[float] = []
        if eval_encodings and self.use_gnn_eval:
            from gnn.model import evaluate_encodings
            probs = evaluate_encodings(eval_encodings)

        if not eval_encodings or not probs:
            for m in eval_moves:
                game.do_move(player.number, m)
                p = self._eval_game_probability(game, player)
                game.undo_move()
                if m.t == "R":
                    p = min(0.999, p + self._rock_bonus_for_cell(game, m.c, self.rock_prior_bonus_connected, self.rock_prior_bonus_disconnected))
                elif self._stick_between_opp_rocks(game, player, m):
                    p = min(0.999, p + self.stick_between_opp_rocks_bonus)
                priors.append((m, float(p)))
        else:
            for m, p in zip(eval_moves_order, probs):
                if m.t == "R":
                    p = min(0.999, p + self._rock_bonus_for_cell(game, m.c, self.rock_prior_bonus_connected, self.rock_prior_bonus_disconnected))
                elif self._stick_between_opp_rocks(game, player, m):
                    p = min(0.999, p + self.stick_between_opp_rocks_bonus)
                priors.append((m, float(p)))

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

        if len(moves) > 1:
            min_p = min((p for _, p in priors), default=0.01)
            pass_p = max(1e-6, 0.05 * min_p)
            priors = [(m, (pass_p if m is PASS else p)) for (m, p) in priors]

        total = sum(p for _, p in priors)
        if total <= 0:
            total = float(len(priors))
            priors = [(m, 1.0) for (m, _) in priors]

        for m, p in priors:
            self.Psa[(state_key, move_key(m))] = p / total

        priors.sort(key=lambda mp: (-mp[1], move_key(mp[0])))
        if len(priors) > 1:
            non_pass = [mp for mp in priors if mp[0] is not PASS]
            only_pass = [mp for mp in priors if mp[0] is PASS]
            priors = [*non_pass, *only_pass]
        self._legal_moves[state_key] = [m for (m, _) in priors]
        self._expanded_count[state_key] = 0

    def _ensure_progressive_widening(self, state_key: StateKey) -> None:
        legal = self._legal_moves.get(state_key)
        if not legal:
            return
        ns = self.Ns[state_key]
        target = int(self.progressive_widening_c * ((ns + 1) ** self.progressive_widening_alpha))

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
            edge = (state_key, move_key(m))
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
        return min(best_moves, key=move_key)

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
            game.do_move(player.number, move)
            sim_moves.append((mover_idx, move_key(move)))
            n_applied += 1

        # Depth limit reached: use a heuristic evaluation as a soft reward.
        # This prevents MCTS from behaving weirdly when cut off.
        reward = self._eval_game_probability(game, game.players[sim_start_player])
        return reward, n_applied, sim_moves, sim_start_player

    def _backpropagate(self, path_edges: List[tuple[StateKey, Move]], reward: float) -> None:
        "Backpropagate along edges; values are from the current-player perspective at each state."
        for state_key, move in reversed(path_edges):
            self.Ns[state_key] += 1
            edge = (state_key, move_key(move))
            self.Nsa[edge] += 1
            self.Wsa[edge] += reward
            reward = 1.0 - reward
