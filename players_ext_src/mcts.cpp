#include "mcts.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>

MCTSEngine::MCTSEngine(int seed, double c_puct_)
    : rng(seed),
      c_puct(c_puct_),
      progressive_widening_c(1.6),
      progressive_widening_alpha(0.55),
      rave_k(250.0),
      prior_eval_cap(48),
      max_sim_depth(50),
      check_forced_losses(true),
      tactical_root_limit(20)
{
    // silence when running as test
    verbose = (std::getenv("PYTEST_CURRENT_TEST") == nullptr);
}

Move MCTSEngine::choose_move(const GameState &root, int n_rollouts)
{
    auto &game = const_cast<GameState &>(root);

    const auto t_start = std::chrono::high_resolution_clock::now();
    auto ret = [&](const Move &mv) -> Move
    {
        if (verbose)
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
        if (a.x != b.x)
            return a.x < b.x;
        if (a.y != b.y)
            return a.y < b.y;
        return a.t < b.t;
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

        // The move list is cached by TTKey (transposition table). In practice, we've observed
        // rare cases where the cached list can become inconsistent with the current state's
        // geometry due to incremental caches on the GameState side. To be robust, we re-check
        // legality before using cached moves.
        if (!moves.empty())
        {
            bool any_illegal = false;
            for (auto &m : moves)
            {
                if (!g.is_move_legal(m, g.current_player))
                {
                    any_illegal = true;
                    break;
                }
            }
            if (any_illegal)
            {
                log_stage("stage2_regen_moves_due_to_illegal");
                moves = g.get_possible_moves_for_player(g.current_player);
            }
        }
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
            return 0.0;
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
        std::sort(moves.begin(), moves.end(), [&](const Move &a, const Move &b)
                  {
					  int ra = move_type_rank(a);
					  int rb = move_type_rank(b);
					  if (ra != rb)
						  return ra < rb;
					  return move_less(a, b); });
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
        for (size_t i = 0; i < n_cand; ++i)
            candidates.push_back(moves[i]);

        std::vector<Move> filtered = tactical_filter(candidates, g);
        order_moves_inplace(filtered);

        try
        {
            py::list encs;
            std::vector<Move> used;
            used.reserve(filtered.size());
            for (auto &m : filtered)
            {
                if (!g.is_move_legal(m, g.current_player))
                    continue;
                int mover = g.current_player;
                struct UndoGuard
                {
                    GameState *g;
                    bool active;
                    explicit UndoGuard(GameState &gs) : g(&gs), active(true) {}
                    ~UndoGuard()
                    {
                        if (active)
                            g->undo_move();
                    }
                } undo_guard(g);

                g.do_move(m, mover);
                py::object enc_obj = players_ext_internal::encode_state_common(g, py_mods, enc_cache, ENC_CACHE_MAX, &total_encode_time);
                encs.append(enc_obj);
                used.push_back(m);
            }
            if ((size_t)py::len(encs) == 0)
                return;

            py::list probs_list = players_ext_internal::eval_probs_common(py_mods, model_override, model_device, encs, &total_model_time, nullptr, nullptr);
            const size_t n = (size_t)py::len(probs_list);
            if (n != used.size())
                return;

            // Convert list of probs into priors and store in Psa.
            double sum_p = 0.0;
            std::vector<double> probs;
            probs.reserve(n);
            for (size_t i = 0; i < n; ++i)
            {
                double p = py::cast<double>(probs_list[i]);
                p = std::max(0.0, std::min(1.0, p));
                probs.push_back(p);
                sum_p += p;
            }
            if (sum_p <= 0.0)
            {
                // fallback uniform
                sum_p = (double)n;
                for (double &p : probs)
                    p = 1.0;
            }

            for (size_t i = 0; i < n; ++i)
            {
                EdgeKey ek{skey, mk_of(used[i])};
                Psa[ek] = probs[i] / sum_p;
            }
        }
        catch (const std::exception &e)
        {
            if (verbose)
                std::cerr << "MCTS prior eval error: " << e.what() << std::endl;
        }

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
        random_shuffle_moves(moves);

        Move best = moves[0];
        double best_score = -1e300;
        for (const auto &m : moves)
        {
            double score = Q(skey, m) + U(skey, m);
            if (score > best_score || (score == best_score && move_less(m, best)))
            {
                best_score = score;
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

        // AMAF/RAVE
        N_amaf[ek] += 1;
        W_amaf[ek] += value;
    };

    auto simulate_one = [&](GameState &g, int root_player)
    {
        std::vector<std::pair<TTKey, Move>> path;
        path.reserve(64);

        while (g.winner == -1)
        {
            TTKey skey = g.tt_key();
            ensure_state_initialized(skey, g);

            auto itNs = Ns.find(skey);
            const int ns = (itNs == Ns.end()) ? 0 : itNs->second;
            if (ns == 0)
                break;

            Move m = select(skey, g);
            if (!g.is_move_legal(m, g.current_player))
            {
                if (verbose)
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
        }

        // leaf: expand + rollout
        TTKey leaf_key = g.tt_key();
        ensure_state_initialized(leaf_key, g);
        maybe_eval_priors(leaf_key, g);

        int winner = rollout(g, root_player);
        double value = (winner == -1) ? 0.0 : ((winner == root_player) ? 1.0 : -1.0);

        for (auto it = path.rbegin(); it != path.rend(); ++it)
        {
            update(it->first, it->second, value);
            value = -value;
        }
    };

    // Main rollout loop
    ensure_state_initialized(root_key, game);
    maybe_eval_priors(root_key, game);

    for (int i = 0; i < n_rollouts; ++i)
    {
        GameState g_copy = game;
        simulate_one(g_copy, game.current_player);
        log_rollout("rollout", i, (int)Ns.size(), (int)Nsa.size(), (int)legal_moves.size(), enc_cache.size());
    }

    auto it_moves = legal_moves.find(root_key);
    if (it_moves == legal_moves.end() || it_moves->second.empty())
        return ret(Move{0, 0, 'P'});

    std::vector<Move> ranked = it_moves->second;
    order_moves_inplace(ranked);

    // Choose argmax visits (break ties deterministically)
    Move best = ranked[0];
    int best_v = -1;
    for (const auto &m : ranked)
    {
        int v = visits(m);
        if (v > best_v || (v == best_v && move_less(m, best)))
        {
            best_v = v;
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
        return ret(safe_moves[0]);
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
void MCTSEngine::set_verbose(bool v) { verbose = v; }

void MCTSEngine::set_progressive_widening(double c, double alpha)
{
    progressive_widening_c = c;
    progressive_widening_alpha = alpha;
}

void MCTSEngine::set_rave_k(double v) { rave_k = v; }
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
    enc_cache.clear();
    _root_key = 0;

    total_encode_time = 0.0;
    total_model_time = 0.0;
}

py::dict MCTSEngine::get_profile_stats()
{
    py::dict d;
    d["total_encode_time"] = total_encode_time;
    d["total_model_time"] = total_model_time;
    return d;
}

void MCTSEngine::advance_root(const GameState &game)
{
    _root_key = ttkey_digest(game.tt_key());
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
