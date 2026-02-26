#include "mcts.hpp"

#include <cmath>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <fstream>

MCTSEngine::MCTSEngine(int seed, double c_puct_)
        : rng(seed),
            c_puct(c_puct_),
            progressive_widening_c(1.6),
            progressive_widening_alpha(0.55),
            rave_k(250.0),
            prior_eval_cap(48),
            max_sim_depth(20),
            check_forced_losses(true),
            tactical_root_limit(20),
            rock_prior_bonus_connected(1.5),
            rock_prior_bonus_disconnected(0.06),
            stick_between_opp_rocks_bonus(0.4)
{
    // Default to silent; tests won't be noisy. Use set_verbose_level() to adjust.
    verbose_level = 0;
}

void MCTSEngine::set_seed(int seed)
{
    rng.seed((std::uint32_t)seed);
}

Move MCTSEngine::choose_move(const GameState &root, int n_rollouts)
{
    auto &game = const_cast<GameState &>(root);
    const int root_player = game.current_player;

    const auto t_start = std::chrono::high_resolution_clock::now();
    auto ret = [&](const Move &mv) -> Move
    {
        if (verbose_level >= 1)
        {
            const auto t_end = std::chrono::high_resolution_clock::now();
            const double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            std::cerr << std::fixed << std::setprecision(2)
                      << "MCTS choose_move: rollouts=" << n_rollouts
                      << " time_ms=" << elapsed_ms
                      << " turn=" << game.turn_number
                      << " cur=" << game.current_player
                      << " move=(" << mv.x << "," << mv.y << "," << mv.t << ")"
                      << std::endl;
        }
        return mv;
    };

    struct RolloutRngGuard
    {
        GameState &g;
        std::mt19937 snapshot;
        explicit RolloutRngGuard(GameState &gs) : g(gs), snapshot(gs.rng_snapshot()) {}
        ~RolloutRngGuard() { g.rng_restore(snapshot); }
    } rollout_rng_guard(game);

    TTKey root_key = game.tt_key();
    std::uint64_t root_key_digest = ttkey_digest(root_key);

    // Debug stuff (currently no-op)
    auto log_stage = [](const char *) {};
    auto log_value = [](const char *, int, int, int) {};
    auto log_rollout = [](const char *, int, int, int, int, size_t) {};
    log_stage("stage1_start");

    auto explain_illegal = [&](const Move &m, int player_number) -> std::string
    {
        std::ostringstream os;
        os << "m=(" << m.x << "," << m.y << "," << m.t << ")";
        os << " turn=" << game.turn_number << " cur=" << game.current_player;
        if (m.t == 'P')
            return os.str() + " pass";
        if (m.t == 'R')
        {
            if (game.turn_number == 0)
                return os.str() + " illegal: turn0";
            if (game.num_rocks[player_number] <= 0)
                return os.str() + " illegal: no_rocks";
            if (game.coord_in_claimed_region_cached({m.x, m.y}))
                return os.str() + " illegal: claimed";
            bool near_anchor = false;
            for (Node *a : game.connected_points)
                if (std::abs(a->x - m.x) <= 1 && std::abs(a->y - m.y) <= 1)
                    near_anchor = true;
            if (!near_anchor)
                for (Node *a : game.rocks)
                    if (std::abs(a->x - m.x) <= 1 && std::abs(a->y - m.y) <= 1)
                        near_anchor = true;
            if (!near_anchor)
                return os.str() + " illegal: not_adjacent";
            auto it = game.points.find(GameState::key_from_coord({m.x, m.y}));
            if (it != game.points.end() && it->second->rocked_by != -1)
                return os.str() + " illegal: occupied";
            return os.str() + " legal?";
        }

        Coord sc{m.x, m.y};
        if (game.coord_in_claimed_region_cached(sc))
            return os.str() + " illegal: start_claimed";
        auto it_start = game.points.find(GameState::key_from_coord(sc));
        if (it_start == game.points.end())
            return os.str() + " illegal: start_missing";
        Node *start = it_start->second.get();
        {
            bool connected = false;
            for (int dd = 0; dd < 8; ++dd)
                if (start->neighbours[dd])
                    connected = true;
            if (!connected)
                return os.str() + " illegal: start_disconnected";
        }
        if (start->rocked_by != -1 && start->rocked_by != player_number)
            return os.str() + " illegal: cannot_place";
        int d = GameState::dir_from_name(m.t);
        if (d < 0 || d > 7)
            return os.str() + " illegal: bad_dir";
        if (start->neighbours[d])
            return os.str() + " illegal: edge_occupied";
        if (game.intersects_stick(sc, d))
            return os.str() + " illegal: intersects";
        Coord endc = calc_end(sc, d);
        if (game.coord_in_claimed_region_cached(endc))
            return os.str() + " illegal: end_claimed";
        auto it_end = game.points.find(GameState::key_from_coord(endc));
        if (it_end != game.points.end())
        {
            Node *end = it_end->second.get();
            int rd = 7 - d;
            if (end->neighbours[rd])
                return os.str() + " illegal: reverse_occupied";
        }
        return os.str() + " legal?";
    };

    auto mk_of = [](const Move &m) -> MoveKey
    { return MoveKey{m.x, m.y, m.t}; };

    auto move_less = [](const Move &a, const Move &b) -> bool
    {
        if (a.t != b.t)
            return a.t < b.t;
        if (a.x != b.x)
            return a.x < b.x;
        return a.y < b.y;
    };

    auto ensure_state_initialized = [&](const TTKey &skey, GameState &g)
    {
        auto it_existing = legal_moves.find(skey);
        if (it_existing == legal_moves.end())
        {
            log_stage("stage2_gen_moves");
            auto gen_moves = g.get_possible_moves_for_player(g.current_player);
            auto it_ins = legal_moves.emplace(skey, std::move(gen_moves)).first;
            it_existing = it_ins;
            expanded_count[skey] = 0;
            Ns.emplace(skey, 0);
        }

        auto &moves = it_existing->second;
    };

    auto visits = [&](const Move &m) -> int
    {
        EdgeKey ek{root_key, mk_of(m)};
        auto it = Nsa.find(ek);
        return (it == Nsa.end()) ? 0 : it->second;
    };

    auto Q = [&](const TTKey &s, const Move &m) -> double
    {
        EdgeKey ek{s, mk_of(m)};
        auto itN = Nsa.find(ek);
        auto itW = Wsa.find(ek);
        if (itN == Nsa.end() || itW == Wsa.end() || itN->second == 0)
            return 0.5; // neutral prior value for unseen edges (match Python)
        return itW->second / (double)itN->second;
    };

    auto P = [&](const TTKey &s, const Move &m) -> double
    {
        EdgeKey ek{s, mk_of(m)};
        auto itP = Psa.find(ek);
        if (itP != Psa.end())
            return itP->second;
        auto itR = root_priors.find(mk_of(m));
        if (itR != root_priors.end())
            return itR->second;
        // Fallback to uniform prior (match Python behavior).
        auto it_moves = legal_moves.find(s);
        if (it_moves != legal_moves.end() && !it_moves->second.empty())
            return 1.0 / (double)it_moves->second.size();
        return 0.0;
    };

    auto U = [&](const TTKey &s, const Move &m) -> double
    {
        int N_s = 0;
        auto itNs = Ns.find(s);
        if (itNs != Ns.end())
            N_s = itNs->second;
        EdgeKey ek{s, mk_of(m)};
        int N_sa = 0;
        auto itNsa = Nsa.find(ek);
        if (itNsa != Nsa.end())
            N_sa = itNsa->second;
        double p = P(s, m);
        return c_puct * p * std::sqrt((double)(N_s + 1)) / (double)(1 + N_sa);
    };

    auto temper_prob = [](double p) -> double
    {
        p = std::max(1e-6, std::min(1.0 - 1e-6, p));
        double logit = std::log(p / (1.0 - p));
        logit *= 0.5; // match Python's logit/2 tempering
        double prob2 = 1.0 / (1.0 + std::exp(-logit));
        return std::max(0.0, std::min(1.0, prob2));
    };

    auto random_shuffle_moves = [&](std::vector<Move> &moves)
    {
        std::shuffle(moves.begin(), moves.end(), rng);
    };

    auto apply_dirichlet_noise = [&](const TTKey &s, std::vector<Move> &moves)
    {
        if (dirichlet_alpha <= 0.0 || dirichlet_epsilon <= 0.0)
            return;

        std::gamma_distribution<double> gamma(dirichlet_alpha, 1.0);
        std::vector<double> noise;
        noise.reserve(moves.size());
        double sum = 0.0;
        for (size_t i = 0; i < moves.size(); ++i)
        {
            double v = gamma(rng);
            noise.push_back(v);
            sum += v;
        }
        if (sum <= 0.0)
            return;
        for (double &v : noise)
            v /= sum;

        for (size_t i = 0; i < moves.size(); ++i)
        {
            EdgeKey ek{s, mk_of(moves[i])};
            double oldp = P(s, moves[i]);
            double newp = (1.0 - dirichlet_epsilon) * oldp + dirichlet_epsilon * noise[i];
            Psa[ek] = newp;
        }
    };

    auto is_pass = [&](const Move &m) -> bool
    { return m.t == 'P'; };

    auto is_rock = [&](const Move &m) -> bool
    { return m.t == 'R'; };

    auto is_stick = [&](const Move &m) -> bool
    { return (!is_pass(m) && !is_rock(m)); };

    auto move_type_rank = [&](const Move &m) -> int
    {
        if (is_stick(m))
            return 0;
        if (is_rock(m))
            return 1;
        return 2;
    };

    auto order_moves_inplace = [&](std::vector<Move> &moves)
    {
        // Order moves to match Python's `move_key` ordering: primary by
        // move-type token `t` (lexical), then x, then y. Python sorts by
        // `(m.t, m.c[0], m.c[1])` so use the same comparator here to ensure
        // the deterministic prefix selection aligns between backends.
        std::sort(moves.begin(), moves.end(), [&](const Move &a, const Move &b)
                  {
                      if (a.t != b.t)
                          return a.t < b.t;
                      if (a.x != b.x)
                          return a.x < b.x;
                      return a.y < b.y; });
    };

    auto rollout = [&](GameState &g, int root_player) -> int
    {
        int steps = 0;
        while (g.winner == -1 && steps < max_sim_depth)
        {
            Move m = g.rollout_pick_move(g);
            g.do_move(m, g.current_player);
            steps++;
        }
        return g.winner;
    };

    auto tactical_filter = [&](const std::vector<Move> &moves, GameState &g) -> std::vector<Move>
    {
        if (!check_forced_losses)
            return moves;

        std::vector<Move> out;
        out.reserve(moves.size());

        GameState tmp = g;
        for (const auto &m : moves)
        {
            if ((int)out.size() >= tactical_root_limit)
                break;
            if (!tmp.is_move_legal(m, tmp.current_player))
                continue;
            if (!tmp.allows_forced_loss_next_round(m, tmp, tmp.current_player))
                out.push_back(m);
        }

        if (out.empty())
            return moves;
        return out;
    };

    auto maybe_eval_priors = [&](const TTKey &skey, GameState &g)
    {
        auto it_exp = expanded_count.find(skey);
        int &exp = (it_exp == expanded_count.end()) ? expanded_count[skey] : it_exp->second;
        if (exp > 0)
            return;
        exp++;

        auto it_moves = legal_moves.find(skey);
        if (it_moves == legal_moves.end())
            return;
        auto &moves = it_moves->second;
        if (moves.empty())
            return;

        if (prior_eval_cap <= 0)
        {
            const double p = 1.0 / (double)moves.size();
            for (const auto &m : moves)
            {
                EdgeKey ek{skey, mk_of(m)};
                Psa[ek] = p;
            }
            apply_dirichlet_noise(skey, moves);
            return;
        }

        const size_t cap = (size_t)prior_eval_cap;
        const size_t n_cand = std::min(cap, moves.size());
        std::vector<Move> candidates;
        candidates.reserve(n_cand);

        // Candidate selection: match Python MCTS deterministic behavior.
        // Take the first `prior_eval_cap` moves in deterministic order and
        // include up to 12 additional rock moves from the remainder. Avoid
        // random sampling here to keep deterministic evals reproducible and
        // consistent with the Python engine.
        if (n_cand == moves.size())
        {
            candidates = moves;
        }
        else
        {
            // Ensure deterministic ordering of the full move list, then take
            // a prefix plus optional extra rock moves from the tail.
            std::vector<Move> ordered = moves;
            order_moves_inplace(ordered);

            size_t take = std::min(n_cand, ordered.size());
            for (size_t i = 0; i < take; ++i)
                candidates.push_back(ordered[i]);

            // Collect up to 12 extra rock moves from the remainder, like Python.
            std::vector<Move> extra_rocks;
            for (size_t i = take; i < ordered.size() && extra_rocks.size() < 12; ++i)
            {
                if (ordered[i].t == 'R')
                    extra_rocks.push_back(ordered[i]);
            }
            if (!extra_rocks.empty())
            {
                candidates.insert(candidates.end(), extra_rocks.begin(), extra_rocks.end());
            }
        }

        // Match Python: do not apply tactical filtering to the candidate set
        // prior to policy evaluation; evaluate the deterministic prefix (and
        // extra rocks) directly. Use candidates directly without re-sorting,
        // since it is already built from sorted moves (prefix + extra rocks).
        std::vector<Move> filtered = candidates;

        std::vector<Move> used;
        used.reserve(filtered.size());
        std::vector<double> priors;
        // acquire GIL
        py::gil_scoped_acquire gil;

        players_ext_internal::ensure_py_gnn_modules(py_mods);

        if (verbose_level >= 2)
        {
            std::cerr << "MCTS prior eval: prior_eval_cap=" << prior_eval_cap
                      << " candidates=" << candidates.size()
                      << " filtered=" << filtered.size() << std::endl;
            std::cerr << "MCTS prior eval: policy_model_override=" << (!policy_model_override.is_none())
                      << " model_override=" << (!model_override.is_none()) << std::endl;
        }

        // Policy-head priors: if a policy model is available, evaluate the current
        // state once and score candidate moves via the policy head's softmax.
        // This is much faster than the value-child path (one NN call vs N).
        if (!policy_model_override.is_none())
        {
            // Build move features for ALL legal moves (policy head handles all in one pass).
            std::vector<Move> policy_used;
            policy_used.reserve(moves.size());
            py::list policy_move_feats;
            for (auto &m : moves)
            {
                if (!g.is_move_legal(m, g.current_player))
                    continue;
                policy_used.push_back(m);

                const double x = (double)m.x;
                const double y = (double)m.y;
                const double is_p = (m.t == 'P') ? 1.0 : 0.0;
                const double is_r = (m.t == 'R') ? 1.0 : 0.0;

                int dir_idx = -1;
                double dx = 0.0;
                double dy = 0.0;
                double end_x = x;
                double end_y = y;
                std::array<double, 8> onehot{};
                onehot.fill(0.0);

                if (!is_p && !is_r)
                {
                    dir_idx = GameState::dir_from_name(m.t);
                    if (dir_idx >= 0 && dir_idx < 8)
                    {
                        dx = (double)DIR_DELTAS[dir_idx][0];
                        dy = (double)DIR_DELTAS[dir_idx][1];
                        end_x = x + dx;
                        end_y = y + dy;
                        onehot[(size_t)dir_idx] = 1.0;
                    }
                }

                policy_move_feats.append(py::make_tuple(
                    x, y, is_p, is_r, end_x, end_y, dx, dy,
                    onehot[0], onehot[1], onehot[2], onehot[3],
                    onehot[4], onehot[5], onehot[6], onehot[7]));
            }

            if (policy_used.empty())
            {
                if (verbose_level >= 2)
                    std::cerr << "MCTS prior eval (policy head): no legal moves." << std::endl;
                return;
            }

            // Encode the current state (not children).
            py::object enc = players_ext_internal::encode_state_common(
                g, py_mods, enc_cache, ENC_CACHE_MAX, &total_encode_time);

            policy_prior_calls += 1;
            policy_prior_items += policy_used.size();

            try
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                py::object policy_infer = py::module::import("rl.policy_infer");
                py::list softmax_priors = policy_infer.attr("policy_priors_from_enc_and_moves")(
                    policy_model_override, enc, policy_move_feats,
                    py::cast(policy_model_device));
                auto t1 = std::chrono::high_resolution_clock::now();
                policy_total_time += std::chrono::duration<double>(t1 - t0).count();

                const size_t np = (size_t)py::len(softmax_priors);
                if (np != policy_used.size())
                {
                    if (verbose_level >= 1)
                        std::cerr << "MCTS policy head: size mismatch " << np
                                  << " vs " << policy_used.size() << std::endl;
                    return;
                }

                // Mix with uniform prior for exploration robustness.
                const double uniform_p = 1.0 / (double)policy_used.size();
                for (size_t i = 0; i < np; ++i)
                {
                    double p = py::cast<double>(softmax_priors[i]);
                    if (!std::isfinite(p) || p < 0.0)
                        p = 0.0;
                    // Apply prior_mix_uniform blending.
                    p = (1.0 - prior_mix_uniform) * p + prior_mix_uniform * uniform_p;
                    // Scale by prior_scale.
                    // (prior_scale acts on the non-uniform portion; applied after mix for simplicity.)

                    EdgeKey ek{skey, mk_of(policy_used[i])};
                    Psa[ek] = p;
                }

                if (verbose_level >= 2)
                {
                    std::cerr << "MCTS prior eval (policy head): priors:";
                    for (size_t i = 0; i < np; ++i)
                    {
                        const Move &um = policy_used[i];
                        std::cerr << " (" << um.x << "," << um.y << "," << um.t
                                  << ")=" << Psa[EdgeKey{skey, mk_of(um)}];
                    }
                    std::cerr << std::endl;
                }
            }
            catch (const std::exception &e)
            {
                if (verbose_level >= 1)
                    std::cerr << "MCTS policy head error: " << e.what() << std::endl;
                // Fall through to re-set uniform priors on failure.
                const double p = 1.0 / (double)moves.size();
                for (const auto &m : moves)
                    Psa[EdgeKey{skey, mk_of(m)}] = p;
            }

            // Re-order moves by prior (best first) for progressive widening.
            std::vector<std::pair<Move, double>> all_with_p;
            all_with_p.reserve(moves.size());
            for (const auto &m : moves)
            {
                double p = Psa[EdgeKey{skey, mk_of(m)}];
                all_with_p.emplace_back(m, p);
            }
            std::sort(all_with_p.begin(), all_with_p.end(), [&](const auto &a, const auto &b)
                      {
                          if (a.second != b.second)
                              return a.second > b.second;
                          return move_less(a.first, b.first); });

            std::vector<Move> new_moves;
            new_moves.reserve(moves.size());
            for (auto &pr : all_with_p)
            {
                if (pr.first.t != 'P')
                    new_moves.push_back(pr.first);
            }
            for (auto &pr : all_with_p)
            {
                if (pr.first.t == 'P')
                    new_moves.push_back(pr.first);
            }
            moves.swap(new_moves);

            apply_dirichlet_noise(skey, moves);
            return;  // Done — skip the value-child path below.
        }

        // ---- Value-child fallback path (no policy model available) ----
        py::list probs_list;

        // Build move feature list and used move vector for the policy path.
        // We'll collect legal `used` moves and their features first, then
        // construct the Python `cand_list` from `used` so instrumentation
        // preserves the exact ordering/mapping sent to the NN.
        py::list move_feats;
        for (auto &m : filtered)
        {
            if (!g.is_move_legal(m, g.current_player))
                continue;
            used.push_back(m);

            const double x = (double)m.x;
            const double y = (double)m.y;
            const double is_p = (m.t == 'P') ? 1.0 : 0.0;
            const double is_r = (m.t == 'R') ? 1.0 : 0.0;

            int dir_idx = -1;
            double dx = 0.0;
            double dy = 0.0;
            double end_x = x;
            double end_y = y;
            std::array<double, 8> onehot{};
            onehot.fill(0.0);

            if (!is_p && !is_r)
            {
                dir_idx = GameState::dir_from_name(m.t);
                if (dir_idx >= 0 && dir_idx < 8)
                {
                    dx = (double)DIR_DELTAS[dir_idx][0];
                    dy = (double)DIR_DELTAS[dir_idx][1];
                    end_x = x + dx;
                    end_y = y + dy;
                    onehot[(size_t)dir_idx] = 1.0;
                }
            }

            move_feats.append(py::make_tuple(
                x,
                y,
                is_p,
                is_r,
                end_x,
                end_y,
                dx,
                dy,
                onehot[0],
                onehot[1],
                onehot[2],
                onehot[3],
                onehot[4],
                onehot[5],
                onehot[6],
                onehot[7]));
        }

        // Build candidate instrumentation list from the actual `used` moves
        // (legal and ordered) so offline comparisons index correctly.
        py::list cand_list;
        try
        {
            for (const auto &cm : used)
            {
                cand_list.append(py::make_tuple(cm.x, cm.y, std::string(1, cm.t)));
            }
        }
        catch (const std::exception &e)
        {
            if (verbose_level >= 1)
                std::cerr << "Policy candidate instrumentation error: " << e.what() << std::endl;
        }

        if (used.empty())
        {
            if (verbose_level >= 2)
                std::cerr << "MCTS prior eval: no usable moves after filtering (used.empty())." << std::endl;
            return;
        }

        prior_model_calls += 1;
        prior_model_batch_items += used.size();

        py::list eval_encs;
        for (const auto &m : used)
        {
            g.do_move(m, g.current_player);
            eval_encs.append(players_ext_internal::encode_state_common(g, py_mods, enc_cache, ENC_CACHE_MAX, &total_encode_time));
            g.undo_move();
        }

        try
        {
            auto t0 = std::chrono::high_resolution_clock::now();
            probs_list = players_ext_internal::eval_probs_common(
                py_mods,
                model_override,
                model_device,
                eval_encs,
                &total_model_time,
                &value_model_calls,
                &value_model_batch_items,
                verbose_level);
            auto t1 = std::chrono::high_resolution_clock::now();
            policy_total_time += std::chrono::duration<double>(t1 - t0).count();

            const size_t m = (size_t)py::len(probs_list);
            if (m != used.size())
                return;

            priors.reserve(m);
            for (size_t i = 0; i < m; ++i)
            {
                double p = py::cast<double>(probs_list[i]);
                if (!std::isfinite(p))
                    p = 0.0;
                p = std::max(0.0, std::min(1.0, p));
                // Flip: NN returns P(opponent wins | child_state) since the
                // child's current_player is the opponent of the mover.
                // We want the prior to reflect P(mover wins | this move).
                p = 1.0 - p;
                // Apply rock prior bonus similar to Python MCTS for evaluated moves.
                if (used[i].t == 'R')
                {
                    Coord cc{used[i].x, used[i].y};
                    auto itnode = g.points.find(GameState::key_from_coord(cc));
                    double bonus = rock_prior_bonus_disconnected;
                    if (itnode != g.points.end())
                    {
                        Node *n = itnode->second.get();
                        bool connected = false;
                        for (auto *cn : g.connected_points)
                        {
                            if (cn == n)
                            {
                                connected = true;
                                break;
                            }
                        }
                        if (connected)
                            bonus = rock_prior_bonus_connected;
                    }
                    p = std::min(0.999, p + bonus);
                }
                else if (g.stick_between_opp_rocks_public(g.current_player, used[i]))
                {
                    p = std::min(0.999, p + stick_between_opp_rocks_bonus);
                }
                priors.push_back(p);
            }
            if (verbose_level >= 2)
            {
                std::cerr << "MCTS prior eval: evaluated priors:";
                for (size_t i = 0; i < priors.size(); ++i)
                {
                    const Move &um = used[i];
                    std::cerr << " (" << um.x << "," << um.y << "," << um.t << ")=" << priors[i];
                }
                std::cerr << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            if (verbose_level >= 1)
                std::cerr << "MCTS prior eval error (value path): " << e.what() << std::endl;
            throw;
        }

        const size_t n = used.size();
        if (n == 0 || priors.size() != n)
            return;

        // Adjust pass probability (reduce pass relative to others), matching Python.
        if (moves.size() > 1)
        {
            double min_p_local = std::numeric_limits<double>::infinity();
            for (double p : priors)
                min_p_local = std::min(min_p_local, p);
            if (!std::isfinite(min_p_local) || min_p_local <= 0.0)
                min_p_local = 0.01;
            double pass_p = std::max(1e-6, 0.05 * min_p_local);
            for (size_t i = 0; i < n; ++i)
            {
                if (used[i].t == 'P')
                    priors[i] = pass_p;
            }
        }

        // Build full raw prior scores across all legal moves (Python-style):
        // - use NN policy probs for evaluated `used` moves (already adjusted for rocks),
        // - assign a small floor to unevaluated moves (with rock bonus where applicable),
        // - reduce `pass` probability relative to min prior, then normalize.
        std::unordered_map<MoveKey, double, MoveKeyHash> raw_map;
        raw_map.reserve(moves.size() * 2);

        // Map evaluated moves
        for (size_t i = 0; i < n; ++i)
            raw_map[mk_of(used[i])] = priors[i];

        // Determine baseline floor from evaluated priors
        double min_prior = std::numeric_limits<double>::infinity();
        for (double p : priors)
            min_prior = std::min(min_prior, p);
        if (!std::isfinite(min_prior) || min_prior <= 0.0)
            min_prior = 0.01;
        const double floor_p = std::max(0.005, 0.25 * min_prior);

        // Assign baseline to unevaluated moves (apply rock bonus if applicable)
        for (const auto &m : moves)
        {
            MoveKey k = mk_of(m);
            if (raw_map.find(k) != raw_map.end())
                continue;
            double p = floor_p;
            if (m.t == 'R')
            {
                Coord cc{m.x, m.y};
                auto itnode = g.points.find(GameState::key_from_coord(cc));
                double bonus = rock_prior_bonus_disconnected;
                if (itnode != g.points.end())
                {
                    Node *nnode = itnode->second.get();
                    bool connected = false;
                    for (auto *cn : g.connected_points)
                    {
                        if (cn == nnode)
                        {
                            connected = true;
                            break;
                        }
                    }
                    if (connected)
                        bonus = rock_prior_bonus_connected;
                }
                p = std::min(0.999, p + bonus);
            }
            else if (g.stick_between_opp_rocks_public(g.current_player, m))
            {
                p = std::min(0.999, p + stick_between_opp_rocks_bonus);
            }
            raw_map[k] = p;
        }

        // Pass adjustment: reduce pass probability relative to min prior
        if (moves.size() > 1)
        {
            double min_p_all = std::numeric_limits<double>::infinity();
            for (const auto &m : moves)
            {
                double rp = raw_map[mk_of(m)];
                min_p_all = std::min(min_p_all, rp);
            }
            if (!std::isfinite(min_p_all) || min_p_all <= 0.0)
                min_p_all = 0.01;
            double pass_p = std::max(1e-6, 0.05 * min_p_all);
            for (const auto &m : moves)
            {
                if (m.t == 'P')
                    raw_map[mk_of(m)] = pass_p;
            }
        }

        // Normalize and store into Psa
        double total_score = 0.0;
        for (const auto &m : moves)
            total_score += raw_map[mk_of(m)];
        if (total_score <= 0.0)
            total_score = (double)moves.size();

        std::vector<std::pair<Move, double>> all_with_p;
        all_with_p.reserve(moves.size());
        for (const auto &m : moves)
        {
            double p = raw_map[mk_of(m)] / total_score;
            EdgeKey ek{skey, mk_of(m)};
            Psa[ek] = p;
            all_with_p.emplace_back(m, p);
        }

        // Instrumentation: also dump the full raw_map and final normalized Psa
        // so we can compare C++ post-policy processing with Python.
        try
        {
            py::dict rec2;
            rec2["ts"] = py::module::import("time").attr("time")();
            rec2["turn"] = g.turn_number;
            py::list raw_list;
            py::list psa_list;
            for (const auto &m : moves)
            {
                MoveKey k = mk_of(m);
                double rawv = 0.0;
                auto itrm = raw_map.find(k);
                if (itrm != raw_map.end())
                    rawv = itrm->second;
                double normv = Psa[EdgeKey{skey, k}];
                py::tuple tup = py::make_tuple(m.x, m.y, std::string(1, m.t), rawv, normv);
                raw_list.append(tup);
                psa_list.append(normv);
            }
            rec2["raw_map"] = raw_list;
            rec2["Psa"] = psa_list;
            // Also include the candidate list in the post-process dump to aid
            // offline analysis linking `raw_map`/`Psa` to the original candidates.
            rec2["candidates"] = cand_list;
            // Disabled: this logging grows very large
            // py::object json = py::module::import("json");
            // std::string s2 = py::cast<std::string>(json.attr("dumps")(rec2, py::arg("ensure_ascii")=false));
            // std::ofstream ofs2("logs/cpp_policy_calls.jsonl", std::ios::app);
            // if (ofs2)
            //     ofs2 << s2 << std::endl;
        }
        catch (const std::exception &e)
        {
            if (verbose_level >= 1)
                std::cerr << "Policy post-process instrumentation error: " << e.what() << std::endl;
        }

        std::sort(all_with_p.begin(), all_with_p.end(), [&](const auto &a, const auto &b)
                  {
                          if (a.second != b.second)
                              return a.second > b.second;
                          return move_less(a.first, b.first); });

        std::vector<Move> new_moves;
        new_moves.reserve(moves.size());
        for (auto &pr : all_with_p)
        {
            if (pr.first.t != 'P')
                new_moves.push_back(pr.first);
        }
        for (auto &pr : all_with_p)
        {
            if (pr.first.t == 'P')
                new_moves.push_back(pr.first);
        }
        moves.swap(new_moves);

        apply_dirichlet_noise(skey, moves);
    };

    auto select = [&](const TTKey &skey, GameState &g) -> Move
    {
        ensure_state_initialized(skey, g);
        maybe_eval_priors(skey, g);
        auto it = legal_moves.find(skey);
        if (it == legal_moves.end() || it->second.empty())
            return Move{0, 0, 'P'};
        auto &moves = it->second;

        // progressive widening
        size_t k = moves.size();
        auto itNs = Ns.find(skey);
        const int ns = (itNs == Ns.end()) ? 0 : itNs->second;
        if (progressive_widening_c > 0.0 && progressive_widening_alpha > 0.0)
        {
            double kk = progressive_widening_c * std::pow((double)(ns + 1), progressive_widening_alpha);
            size_t k2 = (size_t)std::ceil(kk);
            size_t min_k = 1;
            if (skey == root_key && moves.size() > 1)
                min_k = 6;
            k = std::min(moves.size(), std::max(min_k, k2));
        }

        Move best = moves[0];
        double best_score = -1e300;
        constexpr double SCORE_TIE_EPS = 1e-12;
        for (size_t i = 0; i < k; ++i)
        {
            const auto &m = moves[i];
            EdgeKey ek{skey, mk_of(m)};
            int nsa = 0;
            auto itNsa = Nsa.find(ek);
            if (itNsa != Nsa.end())
                nsa = itNsa->second;

            double q_ucb = Q(skey, m);
            double q = q_ucb;
            if (rave_k > 0.0 && skey == root_key)
            {
                auto itNa = N_amaf.find(ek);
                auto itWa = W_amaf.find(ek);
                if (itNa != N_amaf.end() && itWa != W_amaf.end() && itNa->second > 0)
                {
                    double q_amaf = itWa->second / (double)itNa->second;
                    double beta = rave_k / (rave_k + (double)nsa);
                    q = (1.0 - beta) * q_ucb + beta * q_amaf;
                }
            }

            double score = q + U(skey, m);
            if (score > best_score + SCORE_TIE_EPS)
            {
                best_score = score;
                best = m;
            }
            else if (std::fabs(score - best_score) <= SCORE_TIE_EPS)
            {
                // Deterministic tie-break: choose smallest move_key (match Python).
                if (move_less(m, best))
                    best = m;
            }
        }
        return best;
    };

    auto update = [&](const TTKey &s, const Move &m, double value)
    {
        Ns[s] += 1;
        EdgeKey ek{s, mk_of(m)};
        Nsa[ek] += 1;
        Wsa[ek] += value;
    };

    auto backup_path = [&](const std::vector<std::pair<TTKey, Move>> &path, double leaf_value)
    {
        // leaf_value is from the perspective of the player to move at the leaf.
        // The last entry in the path was made by the OPPONENT of leaf_player,
        // so we must flip before the first update.
        double value = leaf_value;
        for (auto it = path.rbegin(); it != path.rend(); ++it)
        {
            value = 1.0 - value;
            update(it->first, it->second, value);
        }
    };

    struct PendingValueEval
    {
        TTKey leaf_key;
        py::object enc;
        std::vector<std::pair<TTKey, Move>> path;
        int leaf_player = 0;
    };

    constexpr size_t VALUE_BATCH_TARGET = 16;

    auto simulate_one = [&](GameState &g, int root_player, PendingValueEval *pending_out) -> bool
    {
        std::vector<std::pair<TTKey, Move>> path;
        path.reserve(64);

        while (g.winner == -1)
        {
            TTKey skey = g.tt_key();
            ensure_state_initialized(skey, g);

            Move m = select(skey, g);
            if (!g.is_move_legal(m, g.current_player))
            {
                if (verbose_level >= 2)
                    std::cerr << "Illegal cached move: " << explain_illegal(m, g.current_player) << std::endl;
                // regenerate legal moves and try again
                legal_moves[skey] = g.get_possible_moves_for_player(g.current_player);
                m = select(skey, g);
                if (!g.is_move_legal(m, g.current_player))
                    break;
            }

            path.emplace_back(skey, m);
            int mover = g.current_player;
            g.do_move(m, mover);

            EdgeKey ek{skey, mk_of(m)};
            auto itNsa = Nsa.find(ek);
            if (itNsa == Nsa.end() || itNsa->second == 0)
                break;
        }

        // leaf: expand + rollout
        TTKey leaf_key = g.tt_key();
        ensure_state_initialized(leaf_key, g);
        maybe_eval_priors(leaf_key, g);

        // Leaf value from the perspective of the player to move at the leaf.
        const int leaf_player = g.current_player;
        double value = 0.0;

        if (g.winner != -1)
        {
            value = (g.winner == leaf_player) ? 1.0 : 0.0;
        }
        else if (use_nn_value)
        {
            auto itV = V.find(leaf_key);
            if (itV != V.end())
            {
                // V stores P(player-to-move wins) = leaf_player's perspective.
                value = itV->second;
            }
            else
            {
                if (model_override.is_none())
                {
                    throw std::runtime_error("MCTS value eval error: use_nn_value enabled but no value model loaded; call set_model_checkpoint(path, device)");
                }

                if (pending_out)
                {
                    // Defer NN value evaluation so we can batch multiple leaf requests.
                    pending_out->leaf_key = leaf_key;
                    pending_out->enc = players_ext_internal::encode_state_common(g, py_mods, enc_cache, ENC_CACHE_MAX, &total_encode_time);
                    pending_out->path = std::move(path);
                    pending_out->leaf_player = leaf_player;
                    return true;
                }
                else
                {
                    // Fallback: immediate (unbatched) value evaluation.
                    double v_leaf = 0.0;
                    try
                    {
                        py::list encs;
                        encs.append(players_ext_internal::encode_state_common(g, py_mods, enc_cache, ENC_CACHE_MAX, &total_encode_time));
                        py::list probs_list = players_ext_internal::eval_probs_common(
                            py_mods,
                            model_override,
                            model_device,
                            encs,
                            &total_model_time,
                            &value_model_calls,
                            &value_model_batch_items,
                            verbose_level);
                        if ((size_t)py::len(probs_list) == 1)
                        {
                            double p = py::cast<double>(probs_list[0]);
                            p = std::max(0.0, std::min(1.0, p));
                            v_leaf = temper_prob(p);
                        }
                    }
                    catch (const std::exception &e)
                    {
                        if (verbose_level >= 1)
                            std::cerr << "MCTS value eval error: " << e.what() << std::endl;
                        throw;
                    }
                    if (value_calibration_enabled)
                    {
                        v_leaf = value_calibration_a * v_leaf + value_calibration_b;
                        v_leaf = std::max(0.0, std::min(1.0, v_leaf));
                    }
                    V[leaf_key] = v_leaf;
                    // v_leaf is P(leaf_player wins) — pass leaf perspective to backup.
                    value = v_leaf;
                }
            }
        }
        else if (use_heuristic_rollout)
        {
            // Use heuristic evaluation instead of random rollout.
            // Returns P(leaf_player wins) via sigmoid of heuristic score.
            double score = heval_evaluate(g, leaf_player);
            value = heval_score_to_prob(score, heuristic_temperature);
            V[leaf_key] = value;
        }
        else
        {
            int winner = rollout(g, root_player);
            value = (winner == -1) ? 0.5 : ((winner == leaf_player) ? 1.0 : 0.0);
        }

        // AMAF/RAVE update for root moves (match Python behavior).
        if (!path.empty())
        {
            double root_reward = (leaf_player == root_player) ? value : (1.0 - value);
            for (const auto &pm : path)
            {
                const TTKey &ps = pm.first;
                if (ps.current_player != root_player)
                    continue;
                EdgeKey ek{root_key, mk_of(pm.second)};
                N_amaf[ek] += 1;
                W_amaf[ek] += root_reward;
            }
        }

        backup_path(path, value);
        return false;
    };

    // Main rollout loop
    ensure_state_initialized(root_key, game);
    maybe_eval_priors(root_key, game);

    std::vector<PendingValueEval> pending_vals;
    pending_vals.reserve(VALUE_BATCH_TARGET * 2);

    auto flush_pending_values = [&]()
    {
        if (pending_vals.empty())
            return;
        try
        {
            py::list encs;
            for (auto &pv : pending_vals)
                encs.append(pv.enc);

            py::list probs_list = players_ext_internal::eval_probs_common(
                py_mods,
                model_override,
                model_device,
                encs,
                &total_model_time,
                &value_model_calls,
                &value_model_batch_items,
                verbose_level);

            // Instrumentation: compute basic stats on returned probabilities
            // to help diagnose value-model issues (ordering, collapse to const,
            // device-mismatch, etc.). This prints a compact summary to stderr.
            if (verbose_level >= 2)
            {
                try
                {
                    const size_t n = (size_t)py::len(probs_list);
                    if (n > 0)
                    {
                        double sum = 0.0, sumsq = 0.0;
                        double minp = 1.0, maxp = 0.0;
                        size_t sample_n = std::min<size_t>(n, 8);
                        std::vector<double> samples;
                        samples.reserve(sample_n);
                        for (size_t i = 0; i < n; ++i)
                        {
                            double p = py::cast<double>(probs_list[i]);
                            if (!std::isfinite(p))
                                p = 0.0;
                            p = std::max(0.0, std::min(1.0, p));
                            sum += p;
                            sumsq += p * p;
                            minp = std::min(minp, p);
                            maxp = std::max(maxp, p);
                            if (i < sample_n)
                                samples.push_back(p);
                        }
                        double mean = sum / (double)n;
                        double var = std::max(0.0, sumsq / (double)n - mean * mean);
                        std::ostringstream os;
                        os << "MCTS value-batch: n=" << n << " min=" << minp << " max=" << maxp << " mean=" << mean << " var=" << var << " samples=[";
                        for (size_t i = 0; i < samples.size(); ++i)
                        {
                            if (i)
                                os << ",";
                            os << samples[i];
                        }
                        os << "]";
                        std::cerr << os.str() << std::endl;
                    }
                }
                catch (const std::exception &)
                {
                    // Non-fatal: continue even if logging fails.
                }
            }

            const size_t n = (size_t)py::len(probs_list);
            if (n == pending_vals.size())
            {
                for (size_t i = 0; i < n; ++i)
                {
                    double p = py::cast<double>(probs_list[i]);
                    if (!std::isfinite(p))
                        p = 0.0;
                    p = std::max(0.0, std::min(1.0, p));
                    double v_leaf = temper_prob(p);
                    if (value_calibration_enabled)
                    {
                        v_leaf = value_calibration_a * v_leaf + value_calibration_b;
                        v_leaf = std::max(0.0, std::min(1.0, v_leaf));
                    }
                    const TTKey leaf_key = pending_vals[i].leaf_key;
                    V[leaf_key] = v_leaf;
                    // v_leaf is P(leaf_player wins) — pass leaf perspective to backup.
                    backup_path(pending_vals[i].path, v_leaf);
                }
            }
            else
            {
                throw std::runtime_error("MCTS value batch eval error: model returned unexpected number of results");
            }
        }
        catch (const std::exception &e)
        {
            if (verbose_level >= 1)
                std::cerr << "MCTS value batch eval error: " << e.what() << std::endl;
            throw;
        }
        pending_vals.clear();
    };

    for (int i = 0; i < n_rollouts; ++i)
    {
        GameState g_copy = game;
        PendingValueEval pv;
        const bool pending = simulate_one(g_copy, root_player, use_nn_value ? &pv : nullptr);
        if (pending)
        {
            pending_vals.push_back(std::move(pv));
            if (pending_vals.size() >= VALUE_BATCH_TARGET)
                flush_pending_values();
        }
        log_rollout("rollout", i, (int)Ns.size(), (int)Nsa.size(), (int)legal_moves.size(), enc_cache.size());
    }

    flush_pending_values();

    auto it_moves = legal_moves.find(root_key);
    if (it_moves == legal_moves.end() || it_moves->second.empty())
        return ret(Move{0, 0, 'P'});

    std::vector<Move> ranked = it_moves->second;
    order_moves_inplace(ranked);

    // Choose argmax visits (break ties randomly but deterministically from RNG seed)
    Move best = ranked[0];
    int best_v = -1;
    int tie_count = 0;
    for (const auto &m : ranked)
    {
        int v = visits(m);
        if (v > best_v)
        {
            best_v = v;
            best = m;
            tie_count = 1;
        }
        else if (v == best_v)
        {
            tie_count += 1;
            std::uniform_int_distribution<int> uid(0, tie_count - 1);
            if (uid(rng) == 0)
                best = m;
        }
    }

    // Basic safety filter: ensure move is legal and not obviously losing.
    // If best is illegal, fall back to next.
    std::vector<Move> safe_moves;
    std::vector<double> safe_visits;
    safe_moves.reserve(ranked.size());
    safe_visits.reserve(ranked.size());
    for (const auto &m : ranked)
    {
        if (!game.is_move_legal(m, game.current_player))
            continue;

        safe_moves.push_back(m);
        safe_visits.push_back((double)std::max(0, visits(m)));
    }

    if (!safe_moves.empty())
    {
        // If any move produces an immediate win for the current player, take it.
        // This avoids random tie-breaks/exploration skipping forced wins.
        {
            GameState tmp = game;
            const int mover = game.current_player;
            for (const auto &m : safe_moves)
            {
                tmp.do_move(m, mover);
                const bool is_immediate_win = (tmp.winner == mover);
                tmp.undo_move();
                if (is_immediate_win)
                    return ret(m);
            }
        }

        const bool explore = (temperature > 0.0 && temperature_moves > 0 && game.turn_number < temperature_moves);
        if (explore)
        {
            double inv_temp = 1.0 / std::max(1e-9, temperature);
            std::vector<double> weights;
            weights.reserve(safe_visits.size());
            for (double v : safe_visits)
                weights.push_back(std::pow(std::max(1e-12, v), inv_temp));
            std::discrete_distribution<size_t> dd(weights.begin(), weights.end());
            return ret(safe_moves[dd(rng)]);
        }
        // Greedy: pick among max-visit moves uniformly at random (seeded).
        Move chosen = safe_moves[0];
        int chosen_v = -1;
        int chosen_ties = 0;
        for (const auto &m : safe_moves)
        {
            int v = visits(m);
            if (v > chosen_v)
            {
                chosen_v = v;
                chosen = m;
                chosen_ties = 1;
            }
            else if (v == chosen_v)
            {
                chosen_ties += 1;
                std::uniform_int_distribution<int> uid(0, chosen_ties - 1);
                if (uid(rng) == 0)
                    chosen = m;
            }
        }
        return ret(chosen);
    }

    for (const auto &m : ranked)
    {
        if (game.is_move_legal(m, game.current_player))
            return ret(m);
    }

    (void)root_key_digest;
    log_value("root", 0, 0, 0);
    return ret(Move{0, 0, 'P'});
}

void MCTSEngine::set_c_puct(double v) { c_puct = v; }
void MCTSEngine::set_verbose(bool v) { verbose_level = v ? 1 : 0; }
void MCTSEngine::set_verbose_level(int v) { verbose_level = std::max(0, v); }
void MCTSEngine::set_use_nn_value(bool v) { use_nn_value = v; }
void MCTSEngine::set_use_heuristic_rollout(bool v) { use_heuristic_rollout = v; }
void MCTSEngine::set_heuristic_temperature(double t) { heuristic_temperature = t; }

void MCTSEngine::py_set_value_calibration(double a, double b, bool enabled)
{
    value_calibration_a = a;
    value_calibration_b = b;
    value_calibration_enabled = enabled;
}

void MCTSEngine::set_progressive_widening(double c, double alpha)
{
    progressive_widening_c = c;
    progressive_widening_alpha = alpha;
}

void MCTSEngine::set_rave_k(double v) { rave_k = v; }

void MCTSEngine::py_set_prior_params(double mix_uniform, double scale)
{
    set_prior_params(mix_uniform, scale);
}
void MCTSEngine::set_prior_eval_cap(int v) { prior_eval_cap = v; }
void MCTSEngine::set_max_sim_depth(int v) { max_sim_depth = v; }
void MCTSEngine::clear_root_priors() { root_priors.clear(); }

void MCTSEngine::set_exploration(double alpha, double epsilon, double temp, int temp_moves)
{
    dirichlet_alpha = std::max(0.0, alpha);
    dirichlet_epsilon = std::max(0.0, std::min(1.0, epsilon));
    temperature = std::max(0.0, temp);
    temperature_moves = std::max(0, temp_moves);
}

void MCTSEngine::set_model_checkpoint(const std::string &path, const std::string &device)
{
    py::gil_scoped_acquire gil;

    players_ext_internal::ensure_py_gnn_modules(py_mods);

    py::object sample_enc = py::module::import("gnn.encode").attr("SAMPLE_ENC");
    int node_dim = sample_enc.attr("data").attr("x").attr("size")(1).cast<int>();
    int global_dim = sample_enc.attr("data").attr("global_feats").attr("size")(1).cast<int>();

    py::object GNNEval = py_mods.gnn_module.attr("GNNEval");
    py::object model = GNNEval("node_feat_dim"_a = node_dim, "global_feat_dim"_a = global_dim);
    py::object state = py_mods.torch_module.attr("load")(py::cast(path), "map_location"_a = py::cast(device));
    model.attr("load_state_dict")(state);
    model.attr("to")(py::cast(device));
    model.attr("eval")();

    model_override = model;
    model_device = device;

    // Search caches are model-dependent.
    clear_stats();
}

void MCTSEngine::set_policy_checkpoint(const std::string &path, const std::string &device)
{
    py::gil_scoped_acquire gil;

    players_ext_internal::ensure_py_gnn_modules(py_mods);

    py::object policy_infer = py::module::import("rl.policy_infer");
    py::object model = policy_infer.attr("load_policy_model")(py::cast(path), py::cast(device));

    policy_model_override = model;
    policy_model_device = device;

    // Search caches are model-dependent.
    clear_stats();
}

void MCTSEngine::reset_search()
{
    clear_stats();
}

std::uint64_t MCTSEngine::get_current_root_key() const { return _root_key; }

py::list MCTSEngine::get_root_visit_stats_py(const GameState &root)
{
    auto &game = const_cast<GameState &>(root);
    TTKey root_key = game.tt_key();

    const std::vector<Move> *moves_ptr = nullptr;
    std::vector<Move> tmp;
    auto it = legal_moves.find(root_key);
    if (it != legal_moves.end())
    {
        moves_ptr = &it->second;
    }
    else
    {
        tmp = game.get_possible_moves_for_player(game.current_player);
        moves_ptr = &tmp;
    }

    py::list out;
    for (const auto &m : *moves_ptr)
    {
        EdgeKey ek{root_key, MoveKey{m.x, m.y, m.t}};
        int v = 0;
        auto itv = Nsa.find(ek);
        if (itv != Nsa.end())
            v = itv->second;

        py::dict d;
        d["x"] = m.x;
        d["y"] = m.y;
        d["t"] = py::cast(std::string(1, m.t));
        d["visits"] = v;
        out.append(d);
    }
    return out;
}

py::list MCTSEngine::get_root_priors_py(const GameState &root)
{
    auto &game = const_cast<GameState &>(root);
    TTKey root_key = game.tt_key();

    const std::vector<Move> *moves_ptr = nullptr;
    std::vector<Move> tmp;
    auto it = legal_moves.find(root_key);
    if (it != legal_moves.end())
    {
        moves_ptr = &it->second;
    }
    else
    {
        tmp = game.get_possible_moves_for_player(game.current_player);
        moves_ptr = &tmp;
    }

    py::list out;
    for (const auto &m : *moves_ptr)
    {
        EdgeKey ek{root_key, MoveKey{m.x, m.y, m.t}};
        double p = 0.0;
        auto itp = Psa.find(ek);
        if (itp != Psa.end())
            p = itp->second;
        auto itr = root_priors.find(MoveKey{m.x, m.y, m.t});
        if (itr != root_priors.end())
            p = itr->second;

        py::dict d;
        d["x"] = m.x;
        d["y"] = m.y;
        d["t"] = py::cast(std::string(1, m.t));
        d["prior"] = p;
        out.append(d);
    }
    return out;
}

py::list MCTSEngine::get_root_values_py(const GameState &root)
{
    auto &game = const_cast<GameState &>(root);
    TTKey root_key = game.tt_key();

    std::vector<GameState> copies;
    std::vector<Move> moves_list;

    auto it = legal_moves.find(root_key);
    if (it == legal_moves.end())
    {
        // fallback: get possible moves
        std::vector<Move> tmp = game.get_possible_moves_for_player(game.current_player);
        for (const auto &m : tmp)
        {
            GameState gcopy = game;
            gcopy.do_move(m, gcopy.current_player);
            copies.push_back(std::move(gcopy));
            moves_list.push_back(m);
        }
    }
    else
    {
        for (const auto &m : it->second)
        {
            GameState gcopy = game;
            gcopy.do_move(m, gcopy.current_player);
            copies.push_back(std::move(gcopy));
            moves_list.push_back(m);
        }
    }

    py::list out;
    if (copies.empty())
        return out;

    try
    {
        py::gil_scoped_acquire gil;
        players_ext_internal::ensure_py_gnn_modules(py_mods);
        py::list encs;
        for (auto &gc : copies)
            encs.append(players_ext_internal::encode_state_common(gc, py_mods, enc_cache, ENC_CACHE_MAX, &total_encode_time));

        py::list probs_list = players_ext_internal::eval_probs_common(
            py_mods,
            model_override,
            model_device,
            encs,
            &total_model_time,
            &value_model_calls,
            &value_model_batch_items,
            verbose_level);

        const size_t n = (size_t)py::len(probs_list);
        for (size_t i = 0; i < n && i < moves_list.size(); ++i)
        {
            double p = py::cast<double>(probs_list[i]);
            if (!std::isfinite(p))
                p = 0.0;
            p = std::max(0.0, std::min(1.0, p));
            py::dict d;
            d["x"] = moves_list[i].x;
            d["y"] = moves_list[i].y;
            d["t"] = py::cast(std::string(1, moves_list[i].t));
            d["prob"] = p;
            out.append(d);
        }
    }
    catch (const std::exception &e)
    {
        if (verbose_level >= 1)
            std::cerr << "get_root_values error: " << e.what() << std::endl;
    }
    return out;
}

void MCTSEngine::set_root_priors_py(py::iterable priors)
{
    root_priors.clear();
    for (auto item : priors)
    {
        py::sequence seq = py::cast<py::sequence>(item);
        if (seq.size() != 4)
            continue;
        int x = seq[0].cast<int>();
        int y = seq[1].cast<int>();
        std::string t = seq[2].cast<std::string>();
        double p = seq[3].cast<double>();
        char tc = t.empty() ? 'P' : t[0];
        root_priors[MoveKey{x, y, tc}] = p;
    }
}

void MCTSEngine::clear_stats()
{
    Ns.clear();
    Nsa.clear();
    Wsa.clear();
    Psa.clear();
    N_amaf.clear();
    W_amaf.clear();
    legal_moves.clear();
    expanded_count.clear();
    V.clear();
    enc_cache.clear();
    _root_key = 0;

    total_encode_time = 0.0;
    total_model_time = 0.0;

    policy_total_time = 0.0;

    prior_model_calls = 0;
    prior_model_batch_items = 0;
    value_model_calls = 0;
    value_model_batch_items = 0;

    policy_prior_calls = 0;
    policy_prior_items = 0;
}

py::dict MCTSEngine::get_profile_stats()
{
    py::dict d;
    d["total_encode_time"] = total_encode_time;
    d["total_model_time"] = total_model_time;
    d["prior_model_calls"] = prior_model_calls;
    d["prior_model_batch_items"] = prior_model_batch_items;
    d["value_model_calls"] = value_model_calls;
    d["value_model_batch_items"] = value_model_batch_items;
    d["policy_priors_enabled"] = !policy_model_override.is_none();
    d["policy_prior_calls"] = policy_prior_calls;
    d["policy_prior_items"] = policy_prior_items;
    d["policy_total_time"] = policy_total_time;
    d["use_nn_value"] = use_nn_value;
    return d;
}

void MCTSEngine::advance_root(const GameState &game)
{
    _root_key = ttkey_digest(game.tt_key());
    // Clear old root priors since they're invalid for the new position.
    // This prevents stale policy estimates from polluting the new search.
    root_priors.clear();
}

void MCTSEngine::prune_tables(int max_states)
{
    if (max_states <= 0)
        return;
    int cur = (int)Ns.size();
    if (cur <= max_states)
        return;

    std::vector<std::pair<TTKey, int>> items;
    items.reserve(Ns.size());
    for (auto &kv : Ns)
        items.emplace_back(kv.first, kv.second);
    std::sort(items.begin(), items.end(), [](auto &a, auto &b)
              { return a.second < b.second; });
    int remove_count = cur - max_states;
    std::unordered_set<TTKey, TTKeyHash> to_remove;
    for (int i = 0; i < remove_count; ++i)
        to_remove.insert(items[i].first);

    for (auto &k : to_remove)
    {
        Ns.erase(k);
        legal_moves.erase(k);
        expanded_count.erase(k);
    }

    auto erase_edges_for_removed = [&](auto &map_like)
    {
        std::vector<typename std::remove_reference_t<decltype(map_like)>::key_type> keys;
        keys.reserve(map_like.size());
        for (auto &kv : map_like)
            keys.push_back(kv.first);
        for (auto &key : keys)
        {
            if (to_remove.find(key.s) != to_remove.end())
                map_like.erase(key);
        }
    };

    erase_edges_for_removed(Nsa);
    erase_edges_for_removed(Wsa);
    erase_edges_for_removed(Psa);
    erase_edges_for_removed(N_amaf);
    erase_edges_for_removed(W_amaf);
}
