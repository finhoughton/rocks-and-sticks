#include "alphabeta.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>

AlphaBetaEngine::AlphaBetaEngine(int seed, double pass_penalty)
    : rng((seed == 0) ? std::mt19937(std::random_device{}()) : std::mt19937((std::uint32_t)seed)),
      pass_penalty(pass_penalty)
{
    // Zero-initialize killer moves and history tables
    for (int p = 0; p < MAX_PLY; ++p)
    {
        killers[p][0] = Move{0, 0, 'P'};
        killers[p][1] = Move{0, 0, 'P'};
    }
    std::memset(history, 0, sizeof(history));
}

// ---- Search enhancement helpers ----

bool AlphaBetaEngine::is_time_up() const
{
    if (search_time_limit_ms <= 0) return false;
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - search_start).count();
    return elapsed >= search_time_limit_ms;
}

std::uint32_t AlphaBetaEngine::move_hash(const Move &m)
{
    // Simple hash for indexing into the history table
    std::uint32_t h = (std::uint32_t)(m.x + 50) * 101 + (std::uint32_t)(m.y + 50);
    h = h * 257 + (std::uint32_t)(unsigned char)m.t;
    return h & (HISTORY_MAX - 1);
}

void AlphaBetaEngine::update_killers(int ply, const Move &m)
{
    if (ply >= MAX_PLY) return;
    // Don't store pass as killer
    if (m.t == 'P') return;
    // Don't store if already killer[0]
    if (killers[ply][0].x == m.x && killers[ply][0].y == m.y && killers[ply][0].t == m.t)
        return;
    killers[ply][1] = killers[ply][0];
    killers[ply][0] = m;
}

bool AlphaBetaEngine::is_killer(int ply, const Move &m) const
{
    if (ply >= MAX_PLY) return false;
    return (killers[ply][0].x == m.x && killers[ply][0].y == m.y && killers[ply][0].t == m.t) ||
           (killers[ply][1].x == m.x && killers[ply][1].y == m.y && killers[ply][1].t == m.t);
}

void AlphaBetaEngine::update_history(int player, const Move &m, int depth)
{
    if (m.t == 'P') return;
    std::uint32_t idx = move_hash(m);
    history[player][idx] += depth * depth;
    // Prevent overflow
    if (history[player][idx] > 1000000)
    {
        for (int i = 0; i < HISTORY_MAX; ++i)
            history[player][i] >>= 1;
    }
}

bool AlphaBetaEngine::has_scoring_move(GameState &g, int player)
{
    // Quick check: does the given player have a move that scores immediately.
    int before_score = g.players_scores[player];
    for (Node *cp : g.connected_points)
    {
        if (!can_place(cp, player)) continue;
        if (g.coord_in_claimed_region_cached(cp->c())) continue;
        for (int d = 0; d < 8; ++d)
        {
            if (cp->neighbours[d] != nullptr) continue;
            if (g.intersects_stick(cp->c(), d)) continue;
            Coord end_c = calc_end(cp->c(), d);
            if (g.coord_in_claimed_region_cached(end_c)) continue;
            Move mv{cp->x, cp->y, GameState::dir_name_char(d)};
            if (!g.is_move_legal(mv, player)) continue;
            g.do_move(mv, player);
            bool scored = (g.players_scores[player] > before_score) || (g.winner == player);
            g.undo_move();
            if (scored) return true;
        }
    }
    return false;
}

void AlphaBetaEngine::order_moves_enhanced(std::vector<Move> &moves, GameState &g, int ply, bool maximising)
{
    // Enhanced move ordering:
    // 1. TT best move (highest priority)
    // 2. Killer moves
    // 3. Remaining sorted by: type rank, then history score
    TTKey key = g.tt_key();
    Move tt_move{0, 0, '\0'};  // sentinel
    auto tt_it = tt.find(key);
    if (tt_it != tt.end())
        tt_move = tt_it->second.best;

    int current_player = g.current_player;

    // Assign a sort score to each move
    struct ScoredMove
    {
        Move m;
        int priority;  // higher = searched first
    };
    std::vector<ScoredMove> scored;
    scored.reserve(moves.size());

    for (auto &m : moves)
    {
        int pri = 0;
        if (tt_move.t != '\0' && m.x == tt_move.x && m.y == tt_move.y && m.t == tt_move.t)
        {
            pri = 1000000000; // TT best move — always first
        }
        else if (is_killer(ply, m))
        {
            pri = 500000000; // Killers second
        }
        else
        {
            // Base priority by type
            if (m.t == 'P')
                pri = -100000000;
            else if (m.t == 'R')
                pri = 0;
            else
                pri = 100000; // sticks

            // Add history bonus
            std::uint32_t idx = move_hash(m);
            pri += history[current_player][idx];
        }
        scored.push_back({m, pri});
    }

    std::stable_sort(scored.begin(), scored.end(), [](const ScoredMove &a, const ScoredMove &b)
    {
        return a.priority > b.priority;
    });

    for (size_t i = 0; i < moves.size(); ++i)
        moves[i] = scored[i].m;
}

Move AlphaBetaEngine::choose_move(const GameState &root, int depth, int move_cap)
{
    auto &game = const_cast<GameState &>(root);

    struct RngGuard
    {
        GameState &g;
        std::mt19937 snapshot;
        explicit RngGuard(GameState &gs) : g(gs), snapshot(gs.rng_snapshot()) {}
        ~RngGuard() { g.rng_restore(snapshot); }
    } rng_guard(game);

    if (depth <= 0)
    {
        auto moves = game.get_possible_moves_for_player(game.current_player);
        if (moves.empty())
            return Move{0, 0, 'P'};
        return moves[0];
    }

    root_player = game.current_player;
    if (last_root_player != -1 && root_player != last_root_player)
    {
        // TT / eval caches depend on root_player perspective.
        tt.clear();
        eval_cache.clear();
    }
    last_root_player = root_player;
    this->move_cap = std::max(1, move_cap);

    {
        auto moves = game.get_possible_moves_for_player(game.current_player);
        order_moves_inplace(moves);
        for (auto &m : moves)
        {
            game.do_move(m, game.current_player);
            if (game.winner == root_player)
            {
                game.undo_move();
                return m;
            }
            game.undo_move();
        }
    }

    auto moves = game.get_possible_moves_for_player(game.current_player);
    if (!moves.empty())
    {
        moves = filter_search_moves(moves, game, game.current_player);
    }
    if (moves.empty())
        return Move{0, 0, 'P'};

    order_moves_inplace(moves);
    if ((int)moves.size() > this->move_cap)
        moves.resize((size_t)this->move_cap);

    Move best_move = moves[0];
    double best_value = -1e300;
    double alpha = -1e300;
    double beta = 1e300;

    for (auto &m : moves)
    {
        if (!game.is_move_legal(m, game.current_player))
            continue;
        int mover = game.current_player;
        game.do_move(m, mover);
        double v = alpha_beta(game, depth - 1, alpha, beta);
        game.undo_move();

        if (m.t == 'P')
            v -= pass_penalty;

        if (v > best_value || (v == best_value && move_less(m, best_move)))
        {
            best_value = v;
            best_move = m;
        }
        alpha = std::max(alpha, best_value);
        if (alpha >= beta)
            break;
    }

    return best_move;
}

std::vector<std::pair<Move, double>> AlphaBetaEngine::choose_move_with_values(
    const GameState &root, int depth, int move_cap)
{
    auto &game = const_cast<GameState &>(root);

    struct RngGuard
    {
        GameState &g;
        std::mt19937 snapshot;
        explicit RngGuard(GameState &gs) : g(gs), snapshot(gs.rng_snapshot()) {}
        ~RngGuard() { g.rng_restore(snapshot); }
    } rng_guard(game);

    root_player = game.current_player;
    if (last_root_player != -1 && root_player != last_root_player)
    {
        tt.clear();
        eval_cache.clear();
    }
    last_root_player = root_player;
    this->move_cap = std::max(1, move_cap);

    auto moves = game.get_possible_moves_for_player(game.current_player);
    if (!moves.empty())
        moves = filter_search_moves(moves, game, game.current_player);
    order_moves_inplace(moves);
    if ((int)moves.size() > this->move_cap)
        moves.resize((size_t)this->move_cap);

    std::vector<std::pair<Move, double>> results;
    for (auto &m : moves)
    {
        if (!game.is_move_legal(m, game.current_player))
            continue;
        int mover = game.current_player;
        game.do_move(m, mover);
        double v = alpha_beta(game, depth - 1, -1e300, 1e300);
        game.undo_move();
        if (m.t == 'P')
            v -= pass_penalty;
        results.push_back({m, v});
    }
    // Sort by descending value
    std::sort(results.begin(), results.end(),
              [](const auto &a, const auto &b) { return a.second > b.second; });
    return results;
}

Move AlphaBetaEngine::choose_move_iterative(const GameState &root, int max_depth, int time_limit_ms, int move_cap)
{
    auto &game = const_cast<GameState &>(root);

    struct RngGuard
    {
        GameState &g;
        std::mt19937 snapshot;
        explicit RngGuard(GameState &gs) : g(gs), snapshot(gs.rng_snapshot()) {}
        ~RngGuard() { g.rng_restore(snapshot); }
    } rng_guard(game);

    if (max_depth <= 0)
    {
        auto moves = game.get_possible_moves_for_player(game.current_player);
        if (moves.empty())
            return Move{0, 0, 'P'};
        return moves[0];
    }

    root_player = game.current_player;
    if (last_root_player != -1 && root_player != last_root_player)
    {
        tt.clear();
        eval_cache.clear();
    }
    last_root_player = root_player;
    this->move_cap = std::max(1, move_cap);

    nodes_searched = 0;
    search_aborted = false;
    search_start = std::chrono::steady_clock::now();
    search_time_limit_ms = time_limit_ms;

    for (int p = 0; p < MAX_PLY; ++p)
    {
        killers[p][0] = Move{0, 0, 'P'};
        killers[p][1] = Move{0, 0, 'P'};
    }

    {
        auto moves = game.get_possible_moves_for_player(game.current_player);
        order_moves_inplace(moves);
        for (auto &m : moves)
        {
            game.do_move(m, game.current_player);
            if (game.winner == root_player)
            {
                game.undo_move();
                last_depth_completed = 1;
                last_nodes_searched = 1;
                return m;
            }
            game.undo_move();
        }
    }

    auto moves = game.get_possible_moves_for_player(game.current_player);
    if (!moves.empty())
        moves = filter_search_moves(moves, game, game.current_player);
    if (moves.empty())
        return Move{0, 0, 'P'};

    order_moves_inplace(moves);
    if ((int)moves.size() > this->move_cap)
        moves.resize((size_t)this->move_cap);

    // NN ordering at root level for better initial move ordering
    if (native_nn_.is_loaded())
    {
        bool root_maximising = (game.current_player == root_player);
        order_moves_by_native_nn(moves, game, root_maximising);
    }

    Move best_move_overall = moves[0];
    int depth_completed = 0;
    double prev_best_value = 0.0;   // for aspiration windows

    // Iterative deepening with aspiration windows
    for (int depth = 1; depth <= max_depth; ++depth)
    {
        if (is_time_up())
            break;

        // Aspiration window: use narrow window around previous depth's value
        double asp_alpha, asp_beta;
        static constexpr double ASP_DELTA = 1.5;  // initial aspiration half-width
        if (depth >= 3 && prev_best_value > -900.0 && prev_best_value < 900.0)
        {
            asp_alpha = prev_best_value - ASP_DELTA;
            asp_beta  = prev_best_value + ASP_DELTA;
        }
        else
        {
            asp_alpha = -1e300;
            asp_beta  = 1e300;
        }

        // Aspiration re-search loop: widen window if search fails high/low
        int asp_retries = 0;
        static constexpr int ASP_MAX_RETRIES = 2;
        double alpha, beta;

asp_retry:
        alpha = asp_alpha;
        beta  = asp_beta;

        Move best_move_this_depth = moves[0];
        double best_value = -1e300;
        bool completed_this_depth = true;

        for (size_t i = 0; i < moves.size(); ++i)
        {
            auto &m = moves[i];
            if (!game.is_move_legal(m, game.current_player))
                continue;

            int mover = game.current_player;
            game.do_move(m, mover);

            double v;
            if (i == 0)
            {
                // First move: full window PVS search
                v = alpha_beta_pvs(game, depth - 1, 1, alpha, beta, true, 0);
            }
            else
            {
                // PVS: null-window search for subsequent moves
                v = alpha_beta_pvs(game, depth - 1, 1, alpha, alpha + 0.01, true, 0);
                if (v > alpha && v < beta && !search_aborted)
                {
                    // Re-search with full window
                    v = alpha_beta_pvs(game, depth - 1, 1, alpha, beta, true, 0);
                }
            }
            game.undo_move();

            if (search_aborted)
            {
                completed_this_depth = false;
                break;
            }

            if (m.t == 'P')
                v -= pass_penalty;

            if (v > best_value || (v == best_value && move_less(m, best_move_this_depth)))
            {
                best_value = v;
                best_move_this_depth = m;
            }
            alpha = std::max(alpha, best_value);

            // If we found a winning move, no need to search deeper
            if (best_value > 900.0)
                break;
        }

        // Handle aspiration window failures: widen and re-search
        if (completed_this_depth && best_value <= asp_alpha && asp_retries < ASP_MAX_RETRIES)
        {
            asp_alpha = -1e300;  // fail low: widen downward
            asp_retries++;
            goto asp_retry;
        }
        if (completed_this_depth && best_value >= asp_beta && asp_retries < ASP_MAX_RETRIES)
        {
            asp_beta = 1e300;  // fail high: widen upward
            asp_retries++;
            goto asp_retry;
        }

        if (completed_this_depth || best_value > 900.0)
        {
            best_move_overall = best_move_this_depth;
            depth_completed = depth;
            prev_best_value = best_value;

            // Re-order moves: put best move first for next iteration
            // This is the key benefit of iterative deepening
            if (moves.size() > 1)
            {
                for (size_t i = 0; i < moves.size(); ++i)
                {
                    if (moves[i].x == best_move_this_depth.x &&
                        moves[i].y == best_move_this_depth.y &&
                        moves[i].t == best_move_this_depth.t)
                    {
                        std::swap(moves[0], moves[i]);
                        break;
                    }
                }
            }

            if (best_value > 900.0)
                break;
        }
        else
        {
            break;
        }
    }

    last_depth_completed = depth_completed;
    last_nodes_searched = nodes_searched;
    return best_move_overall;
}

void AlphaBetaEngine::set_model_checkpoint(const std::string &path, const std::string &device)
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

    clear_stats();
}

void AlphaBetaEngine::clear_stats()
{
    tt.clear();
    eval_cache.clear();
    enc_cache.clear();
    total_encode_time = 0.0;
    total_model_time = 0.0;
}

py::dict AlphaBetaEngine::get_profile_stats()
{
    py::dict d;
    d["total_encode_time"] = total_encode_time;
    d["total_model_time"] = total_model_time;
    d["model_calls"] = (int)model_calls;
    d["model_batch_items"] = (int)model_batch_items;
    d["tt_entries"] = (int)tt.size();
    d["eval_cache_entries"] = (int)eval_cache.size();
    d["enc_cache_entries"] = (int)enc_cache.size();
    return d;
}

bool AlphaBetaEngine::move_less(const Move &a, const Move &b)
{
    if (a.x != b.x)
        return a.x < b.x;
    if (a.y != b.y)
        return a.y < b.y;
    return a.t < b.t;
}

int AlphaBetaEngine::move_type_rank(const Move &m)
{
    if (m.t == 'P')
        return 3;
    if (m.t == 'R')
        return 2;
    return 1;
}

void AlphaBetaEngine::order_moves_inplace(std::vector<Move> &moves)
{
    std::sort(moves.begin(), moves.end(), [](const Move &a, const Move &b)
              {
				  int ra = move_type_rank(a);
				  int rb = move_type_rank(b);
				  if (ra != rb)
					  return ra < rb;
				  return move_less(a, b); });
}

bool AlphaBetaEngine::rock_is_search_worthy(GameState &g, const Move &m)
{
    if (m.t != 'R')
        return true;
    auto it = g.points.find(GameState::key_from_coord({m.x, m.y}));
    if (it != g.points.end())
    {
        Node *p = it->second.get();
        if (p->in_connected_points)
            return true;
        for (int d = 0; d < 8; ++d)
            if (p->neighbours[d])
                return true;
    }

    std::unordered_set<std::uint64_t> rock_coords;
    rock_coords.reserve(g.rocks.size() * 2 + 8);
    for (Node *r : g.rocks)
        rock_coords.insert(GameState::key_from_coord({r->x, r->y}));
    int adjacent = 0;
    for (int dx = -1; dx <= 1; ++dx)
    {
        for (int dy = -1; dy <= 1; ++dy)
        {
            if (dx == 0 && dy == 0)
                continue;
            if (rock_coords.find(GameState::key_from_coord({m.x + dx, m.y + dy})) != rock_coords.end())
            {
                adjacent++;
                if (adjacent >= 2)
                    return true;
            }
        }
    }
    return false;
}

std::vector<Move> AlphaBetaEngine::filter_search_moves(const std::vector<Move> &moves, GameState &g, int player)
{
    std::vector<Move> out;
    out.reserve(moves.size());
    for (const auto &m : moves)
    {
        if (m.t == 'R' && !rock_is_search_worthy(g, m))
            continue;
        out.push_back(m);
    }
    return out;
}

double AlphaBetaEngine::clamp_prob(double p)
{
    if (p < 1e-4)
        p = 1e-4;
    if (p > 1.0 - 1e-4)
        p = 1.0 - 1e-4;
    return p;
}

double AlphaBetaEngine::prob_to_value(double prob)
{
    prob = clamp_prob(prob);
    double logit = std::log(prob / (1.0 - prob));
    logit /= 2.0;
    double p2 = 1.0 / (1.0 + std::exp(-logit));
    double x = (p2 - 0.5) * 2.0;
    x = std::max(-0.999999, std::min(0.999999, x));
    return 6.0 * std::atanh(x);
}

void AlphaBetaEngine::ensure_py_modules()
{
    players_ext_internal::ensure_py_gnn_modules(py_mods);
}

py::object AlphaBetaEngine::encode_state(GameState &g)
{
    return players_ext_internal::encode_state_common(g, py_mods, enc_cache, ENC_CACHE_MAX, &total_encode_time);
}

double AlphaBetaEngine::gnn_prob_root(GameState &g)
{
    // Returns P(root_player wins) from current state.
    if (g.winner != -1)
        return (g.winner == root_player) ? 1.0 : 0.0;
    if (g.connected_points.empty() && g.rocks.empty())
        return 0.5;

    ensure_py_modules();
    py::gil_scoped_acquire gil;
    py::list encs;
    encs.append(encode_state(g));

    double p = 0.5;
    try
    {
        py::list probs_list = players_ext_internal::eval_probs_common(
            py_mods, model_override, model_device, encs, &total_model_time, &model_calls, &model_batch_items, 0);
        p = (py::len(probs_list) > 0) ? py::cast<double>(probs_list[0]) : 0.5;
    }
    catch (const py::error_already_set &e)
    {
        throw std::runtime_error(
            std::string("GNN evaluation is mandatory for AlphaBetaEngine and failed. ") +
            "Ensure a GNN evaluator is loaded in Python (call gnn.model.load_model(...)) or use set_model_checkpoint(...). " +
            std::string("Python error: ") + e.what());
    }

    if (g.current_player != root_player)
        p = 1.0 - p;
    return clamp_prob(p);
}

std::vector<double> AlphaBetaEngine::gnn_probs_root_for_encodings(const py::list &encs)
{
    // Returns P(root_player wins) for each encoding.
    ensure_py_modules();
    py::gil_scoped_acquire gil;

    std::vector<double> probs;
    probs.reserve((size_t)py::len(encs));

    try
    {
        py::list probs_list = players_ext_internal::eval_probs_common(
            py_mods, model_override, model_device, encs, &total_model_time, &model_calls, &model_batch_items, 0);
        for (auto v : probs_list)
            probs.push_back(py::cast<double>(v));
    }
    catch (const py::error_already_set &e)
    {
        throw std::runtime_error(
            std::string("GNN evaluation is mandatory for AlphaBetaEngine and failed. ") +
            "Ensure a GNN evaluator is loaded in Python (call gnn.model.load_model(...)) or use set_model_checkpoint(...). " +
            std::string("Python error: ") + e.what());
    }

    if (probs.size() != (size_t)py::len(encs))
    {
        probs.resize((size_t)py::len(encs), 0.5);
    }
    for (auto &p : probs)
        p = clamp_prob(p);
    return probs;
}

void AlphaBetaEngine::set_use_heuristic(bool v)
{
    if (v != use_heuristic_eval)
    {
        use_heuristic_eval = v;
        tt.clear();
        eval_cache.clear();
    }
}

bool AlphaBetaEngine::load_native_model(const std::string &path)
{
    bool ok = native_nn_.load_weights(path);
    if (ok)
    {
        // Clear caches since eval function changed
        tt.clear();
        eval_cache.clear();
    }
    return ok;
}

void AlphaBetaEngine::set_nn_ordering_depth(int min_depth)
{
    nn_ordering_min_depth_ = std::max(1, min_depth);
}

NNInput AlphaBetaEngine::encode_for_native_nn(GameState &g) const
{
    NNInput input;

    // Build node list (same logic as gnn_helpers.cpp::encode_state_common)
    g.scratch_nodes.clear();
    g.scratch_nodes.reserve(g.connected_points.size() + g.rocks.size());
    for (auto *p : g.connected_points)
        g.scratch_nodes.push_back(p);
    for (auto *p : g.rocks)
        g.scratch_nodes.push_back(p);
    std::sort(g.scratch_nodes.begin(), g.scratch_nodes.end(), [](Node *a, Node *b)
              {
                  if (a == b) return false;
                  if (a->x != b->x) return a->x < b->x;
                  return a->y < b->y; });
    g.scratch_nodes.erase(std::unique(g.scratch_nodes.begin(), g.scratch_nodes.end(),
                                      [](Node *a, Node *b)
                                      { return a == b || (a->x == b->x && a->y == b->y); }),
                          g.scratch_nodes.end());

    g.scratch_idx_map.clear();
    if (!g.scratch_nodes.empty())
        g.scratch_idx_map.reserve(g.scratch_nodes.size() * 2);
    for (size_t i = 0; i < g.scratch_nodes.size(); ++i)
        g.scratch_idx_map[g.scratch_nodes[i]] = (int)i;

    int N = (int)g.scratch_nodes.size();
    input.num_nodes = N;

    // Node features: [owner_one_hot(3), deg, is_leaf, x, y, r2] = 8
    int nfd = GameState::num_players + 1 + 5; // 3 + 5 = 8
    input.node_feats.resize(N * nfd, 0.0f);
    for (int i = 0; i < N; ++i)
    {
        Node *n = g.scratch_nodes[i];
        float *f = &input.node_feats[i * nfd];

        // Owner one-hot: [none, P0, P1]
        int owner_idx = (n->rocked_by >= 0) ? (n->rocked_by + 1) : 0;
        if (owner_idx >= 0 && owner_idx < GameState::num_players + 1)
            f[owner_idx] = 1.0f;

        // Degree
        int nc = 0;
        for (int d = 0; d < 8; ++d)
            if (n->neighbours[d])
                nc++;
        f[3] = (float)nc / 8.0f;
        f[4] = (nc == 1) ? 1.0f : 0.0f;
        f[5] = (float)n->x;
        f[6] = (float)n->y;
        f[7] = (float)(n->x * n->x + n->y * n->y);
    }

    // Edges
    for (int i = 0; i < N; ++i)
    {
        Node *p = g.scratch_nodes[i];
        for (int d = 0; d < 8; ++d)
        {
            Node *q = p->neighbours[d];
            if (!q)
                continue;
            auto it = g.scratch_idx_map.find(q);
            if (it == g.scratch_idx_map.end())
                continue;
            int j = it->second;

            float dx = (float)(q->x - p->x);
            float dy = (float)(q->y - p->y);
            float is_diag = (std::abs(dx) == 1.0f && std::abs(dy) == 1.0f) ? 1.0f : 0.0f;
            float orth = 1.0f - is_diag;

            input.edge_src.push_back(i);
            input.edge_dst.push_back(j);
            input.edge_attr.push_back(orth);
            input.edge_attr.push_back(is_diag);
        }
    }
    input.num_edges = (int)input.edge_src.size();

    // Global features: [turn, cur_one_hot(2), scores(2), rocks_left(2), rocks_placed(2), max_r2]
    input.global_feats.clear();
    input.global_feats.push_back((float)g.turn_number);
    for (int i = 0; i < GameState::num_players; ++i)
        input.global_feats.push_back(g.current_player == i ? 1.0f : 0.0f);
    for (auto s : g.players_scores)
        input.global_feats.push_back((float)s);
    for (auto r : g.num_rocks)
        input.global_feats.push_back((float)r);

    // Rocks placed per player
    std::vector<float> rp(GameState::num_players, 0.0f);
    for (auto *n : g.scratch_nodes)
        if (n->rocked_by != -1)
            rp[n->rocked_by] += 1.0f;
    for (auto v : rp)
        input.global_feats.push_back(v);

    // max_r2
    float max_r2 = 0.0f;
    for (auto *n : g.scratch_nodes)
    {
        float r2 = (float)(n->x * n->x + n->y * n->y);
        max_r2 = std::max(max_r2, r2);
    }
    input.global_feats.push_back(max_r2);

    return input;
}

double AlphaBetaEngine::native_nn_evaluate(GameState &g)
{
    // Returns value from root_player's perspective using native C++ NN
    if (g.winner != -1)
        return (g.winner == root_player) ? 1000.0 : -1000.0;

    if (g.connected_points.empty() && g.rocks.empty())
        return 0.0;

    NNInput input = encode_for_native_nn(g);
    float logit = native_nn_.evaluate(input);
    float prob = 1.0f / (1.0f + std::exp(-logit));

    // Model outputs P(current_player wins)
    // We need P(root_player wins)
    if (g.current_player != root_player)
        prob = 1.0f - prob;

    return prob_to_value(clamp_prob((double)prob));
}

void AlphaBetaEngine::order_moves_by_native_nn(std::vector<Move> &moves, GameState &g, bool maximising)
{
    if (moves.size() <= 1 || !native_nn_.is_loaded())
        return;

    std::vector<double> scores(moves.size(), 0.0);

    for (size_t i = 0; i < moves.size(); ++i)
    {
        const auto &m = moves[i];
        int mover = g.current_player;
        g.do_move(m, mover);

        // Check eval cache first
        TTKey key = g.tt_key();
        auto it = eval_cache.find(key);
        if (it != eval_cache.end())
        {
            scores[i] = it->second;
        }
        else if (g.winner != -1)
        {
            double v = (g.winner == root_player) ? 1000.0 : -1000.0;
            // Terminal values are scale-independent, safe to cache
            eval_cache[key] = v;
            scores[i] = v;
        }
        else if (g.connected_points.empty() && g.rocks.empty())
        {
            double v = prob_to_value(0.5);
            // Empty-board values are also scale-independent
            eval_cache[key] = v;
            scores[i] = v;
        }
        else
        {
            NNInput input = encode_for_native_nn(g);
            float logit = native_nn_.evaluate(input);
            float prob = 1.0f / (1.0f + std::exp(-logit));
            if (g.current_player != root_player)
                prob = 1.0f - prob;
            double v = prob_to_value(clamp_prob((double)prob));
            // IMPORTANT: only write to eval_cache when NOT in heuristic mode.
            // In heuristic mode, eval_cache stores heuristic values (scale ~±20).
            // NN values (scale ~±6) must not pollute the shared cache.
            if (!use_heuristic_eval)
                eval_cache[key] = v;
            scores[i] = v;
        }

        g.undo_move();

        if (moves[i].t == 'P')
        {
            if (maximising)
                scores[i] -= pass_penalty;
            else
                scores[i] += pass_penalty;
        }
    }

    // Sort by score
    std::vector<size_t> idx(moves.size());
    for (size_t i = 0; i < idx.size(); ++i)
        idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](size_t ia, size_t ib)
                     {
                         double a = scores[ia];
                         double b = scores[ib];
                         if (a != b)
                             return maximising ? (a > b) : (a < b);
                         int ra = move_type_rank(moves[ia]);
                         int rb = move_type_rank(moves[ib]);
                         if (ra != rb)
                             return ra < rb;
                         return move_less(moves[ia], moves[ib]); });

    std::vector<Move> reordered;
    reordered.reserve(moves.size());
    for (size_t i : idx)
        reordered.push_back(moves[i]);
    moves.swap(reordered);
}

// ---- Handcrafted heuristic eval (same as players/ai.py) ----

bool AlphaBetaEngine::can_place(const Node *n, int player_number) const
{
    return n->rocked_by == -1 || n->rocked_by == player_number;
}

double AlphaBetaEngine::scored_gain_from_area(int area2)
{
    // area2 is 2x the actual area
    // HALF_AREA_COUNTS is false, so area=1 (area2=2) scores 0.
    int area = area2 / 2;
    if (area <= 0)
        return 0.0;
    if (area == 1)
        return 0.0; // HALF_AREA_COUNTS = false
    return (double)area;
}

int AlphaBetaEngine::closure_area2_for_stick(GameState &g, const Move &m)
{
    // Check if placing this stick would create a cycle, and return 2*area.
    auto it = g.points.find(GameState::key_from_coord({m.x, m.y}));
    if (it == g.points.end())
        return 0;
    Node *start = it->second.get();
    int d = GameState::dir_from_name(m.t);
    if (d < 0)
        return 0;
    Node *end = start->neighbours[d];
    if (end != nullptr)
        return 0; // stick already exists
    Coord end_c = calc_end({m.x, m.y}, d);
    auto it2 = g.points.find(GameState::key_from_coord(end_c));
    if (it2 == g.points.end())
        return 0; // end node doesn't exist, can't form cycle
    end = it2->second.get();
    if (!end->in_connected_points)
        return 0;

    std::uint64_t edge_key = 0;
    int area2 = g.best_new_cycle_area2(start, end, edge_key);
    return area2;
}

AlphaBetaEngine::TacticalInfo AlphaBetaEngine::compute_tactical(GameState &g, int player_number)
{
    TacticalInfo info;

    // Gather stick moves for this player
    std::vector<Move> stick_moves;
    stick_moves.reserve(48);
    for (Node *cp : g.connected_points)
    {
        if (!can_place(cp, player_number))
            continue;
        if (g.coord_in_claimed_region_cached(cp->c()))
            continue;
        for (int d = 0; d < 8; ++d)
        {
            if (cp->neighbours[d] != nullptr)
                continue; // stick already placed
            if (g.intersects_stick(cp->c(), d))
                continue;
            Coord end_c = calc_end(cp->c(), d);
            if (g.coord_in_claimed_region_cached(end_c))
                continue;
            stick_moves.push_back(Move{cp->x, cp->y, GameState::dir_name_char(d)});
        }
    }

    info.stick_move_count = (int)stick_moves.size();
    if (!stick_moves.empty())
    {
        // Use closure_area2_for_stick to compute gains WITHOUT state mutation.
        // This is ~10x faster than do_move/undo_move.
        std::vector<double> gains;
        gains.reserve(stick_moves.size());

        int cap = std::min((int)stick_moves.size(), 32);

        for (int i = 0; i < cap; ++i)
        {
            const Move &mv = stick_moves[i];
            int a2 = closure_area2_for_stick(g, mv);
            double gain = scored_gain_from_area(a2);

            if (a2 == 2) // area==1: bad closure
            {
                info.bad_closure_count++;
            }
            else if (gain > 0.0)
            {
                info.scoring_move_count++;
                info.max_immediate_gain = std::max(info.max_immediate_gain, gain);
                gains.push_back(gain);
            }
        }

        std::sort(gains.begin(), gains.end(), std::greater<double>());
        for (int i = 0; i < std::min((int)gains.size(), 3); ++i)
            info.top3_gain_sum += gains[i];

        // best_reply_gain: find our best scoring move via closure, play it once,
        // then check opponent replies with closure (1 do_move/undo_move total).
        if (info.scoring_move_count > 0)
        {
            Move best_scoring_move{0, 0, 'P'};
            double best_gain = 0.0;
            for (int i = 0; i < cap; ++i)
            {
                const Move &mv = stick_moves[i];
                int a2 = closure_area2_for_stick(g, mv);
                double gain = scored_gain_from_area(a2);
                if (gain > best_gain)
                {
                    best_gain = gain;
                    best_scoring_move = mv;
                }
            }
            if (best_gain > 0.0 && g.is_move_legal(best_scoring_move, player_number))
            {
                // Play the best scoring move, then estimate opponent's best reply via closure
                g.do_move(best_scoring_move, player_number);
                int opp = 1 - player_number;
                double opp_best_reply = 0.0;
                for (Node *cp : g.connected_points)
                {
                    if (!can_place(cp, opp)) continue;
                    if (g.coord_in_claimed_region_cached(cp->c())) continue;
                    for (int d = 0; d < 8; ++d)
                    {
                        if (cp->neighbours[d] != nullptr) continue;
                        if (g.intersects_stick(cp->c(), d)) continue;
                        Coord end_c = calc_end(cp->c(), d);
                        if (g.coord_in_claimed_region_cached(end_c)) continue;
                        Move rmv{cp->x, cp->y, GameState::dir_name_char(d)};
                        int ra2 = closure_area2_for_stick(g, rmv);
                        double rgain = scored_gain_from_area(ra2);
                        if (rgain > opp_best_reply)
                        {
                            opp_best_reply = rgain;
                            if (opp_best_reply > 0.0) goto opp_done; // found a reply, stop
                        }
                    }
                }
                opp_done:
                g.undo_move();
                info.best_reply_gain = opp_best_reply;
            }
        }
    }

    // potential_area: count (connected point, empty direction) pairs where both ends are placeable
    {
        double pa = 0.0;
        for (Node *cp : g.connected_points)
        {
            if (!can_place(cp, player_number))
                continue;
            for (int d = 0; d < 8; ++d)
            {
                if (cp->neighbours[d] != nullptr)
                    continue; // stick exists
                Coord end_c = calc_end(cp->c(), d);
                auto it = g.points.find(GameState::key_from_coord(end_c));
                if (it != g.points.end() && can_place(it->second.get(), player_number))
                    pa += 1.0;
            }
        }
        info.potential_area = pa;
    }

    // blocking_power: count empty directions from our rocks that block opponent
    {
        double blocked = 0.0;
        for (Node *rock : g.rocks)
        {
            if (rock->rocked_by != player_number)
                continue;
            if (!rock->in_connected_points)
                continue;
            if (g.coord_in_claimed_region_cached(rock->c()))
                continue;
            for (int d = 0; d < 8; ++d)
            {
                if (rock->neighbours[d] != nullptr)
                    continue;
                if (g.intersects_stick(rock->c(), d))
                    continue;
                Coord end_c = calc_end(rock->c(), d);
                if (g.coord_in_claimed_region_cached(end_c))
                    continue;
                blocked += 1.0;
                int a2 = closure_area2_for_stick(g, Move{rock->x, rock->y, GameState::dir_name_char(d)});
                if (a2 > 0)
                    blocked += 0.25 * scored_gain_from_area(a2);
            }
        }
        info.blocking_power = blocked;
    }

    // rock_value: estimate value of rock placement opportunities
    {
        int rocks_remaining = g.num_rocks[player_number];
        if (rocks_remaining <= 0)
        {
            info.rock_value = 0.0;
        }
        else
        {
            int opp = 1 - player_number;
            double best_impact = 0.0;
            int rock_count = 0;
            auto all_moves = g.get_possible_moves_for_player(player_number);
            for (const auto &m : all_moves)
            {
                if (m.t != 'R')
                    continue;
                auto it = g.points.find(GameState::key_from_coord({m.x, m.y}));
                if (it == g.points.end())
                    continue;
                Node *p = it->second.get();
                if (!p->in_connected_points)
                    continue;
                // Only matters if opponent could place here
                if (!can_place(p, opp))
                    continue;

                double impact = 0.0;
                for (int d = 0; d < 8; ++d)
                {
                    if (p->neighbours[d] != nullptr)
                        continue;
                    if (g.intersects_stick(p->c(), d))
                        continue;
                    impact += 1.0;
                    int a2 = closure_area2_for_stick(g, Move{p->x, p->y, GameState::dir_name_char(d)});
                    if (a2 > 0)
                        impact += 0.30 * scored_gain_from_area(a2);
                }
                best_impact = std::max(best_impact, impact);
                if (++rock_count >= 24)
                    break;
            }
            double opportunity = std::min(3.0, best_impact / 4.0);
            info.rock_value = (double)rocks_remaining * opportunity;
        }
    }

    return info;
}

double AlphaBetaEngine::heuristic_evaluate(GameState &g)
{
    if (g.winner != -1)
        return (g.winner == root_player) ? 1000.0 : -1000.0;

    if (g.connected_points.empty() && g.rocks.empty())
        return 0.0;

    int me = root_player;
    int opp = 1 - me;

    TacticalInfo my_ts = compute_tactical(g, me);
    TacticalInfo opp_ts = compute_tactical(g, opp);

    bool my_turn = (g.current_player == me);
    double w_me  = my_turn ? 1.0 : 0.4;
    double w_opp = my_turn ? 0.4 : 1.0;

    double v =
        1.2 * (my_ts.blocking_power - opp_ts.blocking_power) +
        1.5 * (my_ts.potential_area - opp_ts.potential_area) +
        1.8 * (my_ts.rock_value - opp_ts.rock_value) +
        1.5 * (my_ts.max_immediate_gain - opp_ts.max_immediate_gain) +
        0.5 * (my_ts.top3_gain_sum - opp_ts.top3_gain_sum) +
        0.4 * (double)(my_ts.scoring_move_count - opp_ts.scoring_move_count) +
        0.2 * (double)(my_ts.stick_move_count - opp_ts.stick_move_count) -
        0.2 * (double)(my_ts.bad_closure_count - opp_ts.bad_closure_count);

    // Best-reply gain: how much the opponent scores after our best scoring move
    // (matches Python _evaluate_position_handcrafted)
    v -= 0.5 * w_me  * my_ts.best_reply_gain;
    v += 0.5 * w_opp * opp_ts.best_reply_gain;

    // Add score difference (very important in endgame)
    v += 10.0 * (double)(g.players_scores[me] - g.players_scores[opp]);

    return v;
}

// ---- End heuristic eval ----

double AlphaBetaEngine::evaluate(GameState &g)
{
    TTKey key = g.tt_key();
    auto it = eval_cache.find(key);
    if (it != eval_cache.end())
        return it->second;

    double v;
    if (use_heuristic_eval)
        v = heuristic_evaluate(g);
    else if (native_nn_.is_loaded())
        v = native_nn_evaluate(g);
    else
    {
        double p = gnn_prob_root(g);
        v = prob_to_value(p);
    }
    eval_cache[key] = v;
    return v;
}

std::vector<double> AlphaBetaEngine::evaluate_children_depth1_batched(
    GameState &g, const std::vector<Move> &moves, bool parent_maximising)
{
    // For depth==1, children are evaluated at depth 0.
    // In heuristic mode, evaluate each child directly (no GNN batching needed).
    if (use_heuristic_eval)
    {
        std::vector<double> values;
        values.resize(moves.size(), 0.0);
        for (size_t i = 0; i < moves.size(); ++i)
        {
            const auto &m = moves[i];
            int mover = g.current_player;
            g.do_move(m, mover);
            values[i] = evaluate(g);
            g.undo_move();
            if (m.t == 'P')
            {
                if (parent_maximising)
                    values[i] -= pass_penalty;
                else
                    values[i] += pass_penalty;
            }
        }
        return values;
    }

    // GNN batched evaluation path
    std::vector<double> values;
    values.resize(moves.size(), 0.0);

    std::vector<TTKey> keys;
    keys.resize(moves.size());

    std::vector<size_t> need_eval_indices;
    need_eval_indices.reserve(moves.size());

    std::vector<char> flip_flags;
    flip_flags.reserve(moves.size());

    py::gil_scoped_acquire gil;
    py::list encs;

    for (size_t i = 0; i < moves.size(); ++i)
    {
        const auto &m = moves[i];
        int mover = g.current_player;
        g.do_move(m, mover);

        TTKey key = g.tt_key();
        keys[i] = key;
        auto it = eval_cache.find(key);
        if (it != eval_cache.end())
        {
            values[i] = it->second;
            g.undo_move();
            continue;
        }

        // Terminal/empty fast paths without model.
        if (g.winner != -1)
        {
            double p = (g.winner == root_player) ? 1.0 : 0.0;
            if (g.current_player != root_player)
                p = 1.0 - p;
            double v = prob_to_value(p);
            eval_cache[key] = v;
            values[i] = v;
            g.undo_move();
            continue;
        }
        if (g.connected_points.empty() && g.rocks.empty())
        {
            double v = prob_to_value(0.5);
            eval_cache[key] = v;
            values[i] = v;
            g.undo_move();
            continue;
        }

        // Need model: build encoding for this child.
        encs.append(encode_state(g));
        need_eval_indices.push_back(i);
        flip_flags.push_back((g.current_player != root_player) ? 1 : 0);
        g.undo_move();
    }

    if (!need_eval_indices.empty())
    {
        auto probs = gnn_probs_root_for_encodings(encs);
        for (size_t j = 0; j < need_eval_indices.size(); ++j)
        {
            size_t i = need_eval_indices[j];
            double p = probs[j];
            if (flip_flags[j])
                p = 1.0 - p;

            double v = prob_to_value(p);
            eval_cache[keys[i]] = v;
            values[i] = v;
        }
    }

    // Apply pass penalty at the parent node (matches alpha_beta recursion).
    for (size_t i = 0; i < moves.size(); ++i)
    {
        if (moves[i].t == 'P')
        {
            if (parent_maximising)
                values[i] -= pass_penalty;
            else
                values[i] += pass_penalty;
        }
    }
    return values;
}

void AlphaBetaEngine::order_moves_by_child_eval_inplace(std::vector<Move> &moves, GameState &g, bool parent_maximising)
{
    if (moves.size() <= 1)
        return;

    // In heuristic mode, evaluate children directly and sort.
    if (use_heuristic_eval)
    {
        std::vector<double> scores(moves.size(), 0.0);
        for (size_t i = 0; i < moves.size(); ++i)
        {
            int mover = g.current_player;
            g.do_move(moves[i], mover);
            scores[i] = evaluate(g);
            g.undo_move();
            if (moves[i].t == 'P')
            {
                if (parent_maximising)
                    scores[i] -= pass_penalty;
                else
                    scores[i] += pass_penalty;
            }
        }
        std::vector<size_t> idx(moves.size());
        for (size_t i = 0; i < idx.size(); ++i)
            idx[i] = i;
        std::stable_sort(idx.begin(), idx.end(), [&](size_t ia, size_t ib)
                         {
                             double a = scores[ia];
                             double b = scores[ib];
                             if (a != b)
                                 return parent_maximising ? (a > b) : (a < b);
                             int ra = move_type_rank(moves[ia]);
                             int rb = move_type_rank(moves[ib]);
                             if (ra != rb)
                                 return ra < rb;
                             return move_less(moves[ia], moves[ib]);
                         });
        std::vector<Move> reordered;
        reordered.reserve(moves.size());
        for (size_t i : idx)
            reordered.push_back(moves[i]);
        moves.swap(reordered);
        return;
    }

    std::vector<double> scores;
    scores.resize(moves.size(), 0.0);

    std::vector<TTKey> keys;
    keys.resize(moves.size());

    std::vector<size_t> need_eval_indices;
    need_eval_indices.reserve(moves.size());
    std::vector<char> flip_flags;
    flip_flags.reserve(moves.size());

    py::gil_scoped_acquire gil;
    py::list encs;

    for (size_t i = 0; i < moves.size(); ++i)
    {
        const auto &m = moves[i];
        int mover = g.current_player;
        g.do_move(m, mover);

        TTKey key = g.tt_key();
        keys[i] = key;
        auto it = eval_cache.find(key);
        if (it != eval_cache.end())
        {
            scores[i] = it->second;
            g.undo_move();
            continue;
        }

        if (g.winner != -1)
        {
            double p = (g.winner == root_player) ? 1.0 : 0.0;
            if (g.current_player != root_player)
                p = 1.0 - p;
            double v = prob_to_value(p);
            eval_cache[key] = v;
            scores[i] = v;
            g.undo_move();
            continue;
        }
        if (g.connected_points.empty() && g.rocks.empty())
        {
            double v = prob_to_value(0.5);
            eval_cache[key] = v;
            scores[i] = v;
            g.undo_move();
            continue;
        }

        encs.append(encode_state(g));
        need_eval_indices.push_back(i);
        flip_flags.push_back((g.current_player != root_player) ? 1 : 0);
        g.undo_move();
    }

    if (!need_eval_indices.empty())
    {
        auto probs = gnn_probs_root_for_encodings(encs);
        for (size_t j = 0; j < need_eval_indices.size(); ++j)
        {
            size_t i = need_eval_indices[j];
            double p = probs[j];
            if (flip_flags[j])
                p = 1.0 - p;
            double v = prob_to_value(p);
            eval_cache[keys[i]] = v;
            scores[i] = v;
        }
    }

    for (size_t i = 0; i < moves.size(); ++i)
    {
        if (moves[i].t == 'P')
        {
            if (parent_maximising)
                scores[i] -= pass_penalty;
            else
                scores[i] += pass_penalty;
        }
    }

    std::vector<size_t> idx(moves.size());
    for (size_t i = 0; i < idx.size(); ++i)
        idx[i] = i;

    std::stable_sort(idx.begin(), idx.end(), [&](size_t ia, size_t ib)
                     {
						 double a = scores[ia];
						 double b = scores[ib];
						 if (a != b)
							 return parent_maximising ? (a > b) : (a < b);
						 int ra = move_type_rank(moves[ia]);
						 int rb = move_type_rank(moves[ib]);
						 if (ra != rb)
							 return ra < rb;
						 return move_less(moves[ia], moves[ib]); });

    std::vector<Move> reordered;
    reordered.reserve(moves.size());
    for (size_t i : idx)
        reordered.push_back(moves[i]);
    moves.swap(reordered);
}

// ---- Quiescence search ----
// Called from depth=0 leaf nodes to resolve immediate scoring chains.
// Only explores moves that immediately score (area > 1), preventing horizon effect.
// qd = remaining quiescence depth (starts at QSEARCH_MAX, decrements each level).
double AlphaBetaEngine::quiescence(GameState &g, double alpha, double beta, int ply, int qd)
{
    if (search_aborted) return 0.0;
    if (ply >= MAX_PLY - 2 || qd <= 0)
        return evaluate(g);

    // Handle terminal positions
    if (g.winner != -1)
        return evaluate(g);

    bool maximising = (g.current_player == root_player);

    // Stand-pat: the static eval (we can always choose to stop searching)
    double stand_pat = evaluate(g);

    if (maximising)
    {
        if (stand_pat >= beta) return stand_pat;
        if (stand_pat > alpha) alpha = stand_pat;
    }
    else
    {
        if (stand_pat <= alpha) return stand_pat;
        if (stand_pat < beta) beta = stand_pat;
    }

    // Collect scoring moves BEFORE any do_move (do_move may modify connected_points)
    int mover = g.current_player;
    std::vector<Move> scoring_moves;
    scoring_moves.reserve(8);
    for (Node *cp : g.connected_points)
    {
        if (!can_place(cp, mover)) continue;
        if (g.coord_in_claimed_region_cached(cp->c())) continue;
        for (int d = 0; d < 8; ++d)
        {
            if (cp->neighbours[d] != nullptr) continue;
            if (g.intersects_stick(cp->c(), d)) continue;
            Coord end_c = calc_end(cp->c(), d);
            if (g.coord_in_claimed_region_cached(end_c)) continue;

            // Only explore scoring moves (area > 1, i.e. a2 > 2)
            int a2 = closure_area2_for_stick(g, Move{cp->x, cp->y, GameState::dir_name_char(d)});
            if (a2 <= 2) continue;

            scoring_moves.push_back(Move{cp->x, cp->y, GameState::dir_name_char(d)});
            if ((int)scoring_moves.size() >= 6) goto scoring_done; // cap to limit branching
        }
    }
    scoring_done:

    for (const Move &mv : scoring_moves)
    {
        g.do_move(mv, mover);
        double v = quiescence(g, alpha, beta, ply + 1, qd - 1);
        g.undo_move();

        if (search_aborted) return 0.0;

        if (maximising)
        {
            if (v > stand_pat) stand_pat = v;
            if (v >= beta) return v;
            if (v > alpha) alpha = v;
        }
        else
        {
            if (v < stand_pat) stand_pat = v;
            if (v <= alpha) return v;
            if (v < beta) beta = v;
        }
    }

    return stand_pat;
}

// ---- Enhanced alpha-beta with PVS, null-move, killers, history ----
// Uses minimax convention: all values from root_player's perspective.
// alpha = best value root_player can guarantee;  beta = worst value root_player allows.

double AlphaBetaEngine::alpha_beta_pvs(GameState &g, int depth, int ply,
                                        double alpha, double beta,
                                        bool allow_null_move, int extensions_left)
{
    nodes_searched++;

    // Time check every 256 nodes
    if ((nodes_searched & 255) == 0 && is_time_up())
    {
        search_aborted = true;
        return 0.0;
    }

    // Bug fix: save original alpha/beta BEFORE TT adjustment for correct flag determination
    double a0 = alpha;
    double b0 = beta;

    TTKey key = g.tt_key();
    auto tt_it = tt.find(key);
    Move tt_best{0, 0, '\0'}; // sentinel for "no TT move"

    if (tt_it != tt.end())
    {
        const TTEntry &e = tt_it->second;
        tt_best = e.best;
        if (e.depth >= depth)
        {
            if (e.flag == 0) return e.value;              // exact
            if (e.flag == 1) alpha = std::max(alpha, e.value); // lower
            else if (e.flag == 2) beta = std::min(beta, e.value); // upper
            if (alpha >= beta) return e.value;
        }
    }

    // Terminal positions
    if (g.winner != -1)
    {
        if (use_heuristic_eval)
            return (g.winner == root_player) ? (1000.0 + depth) : (-1000.0 - depth);
        else
            return (g.winner == root_player) ? prob_to_value(1.0) + depth : prob_to_value(0.0) - depth;
    }

    // Leaf evaluation: call quiescence to resolve immediate scoring chains
    if (depth <= 0)
    {
        return quiescence(g, alpha, beta, ply, 2);
    }

    bool maximising = (g.current_player == root_player);

    // ---- Null-move pruning (heuristic mode only) ----
    // Disabled in narrow PVS windows to avoid false cutoffs (Bug #3 fix)
    bool wide_window = (beta - alpha) > 1.0;
    if (allow_null_move && wide_window && depth >= 5 && use_heuristic_eval && ply > 0)
    {
        // Safety: only try null-move if pass is legal
        Move null_move{0, 0, 'P'};
        if (g.is_move_legal(null_move, g.current_player))
        {
            int mover = g.current_player;
            g.do_move(null_move, mover);
            int R = NULL_MOVE_R;
            int reduced = std::max(0, depth - 1 - R);
            double null_val = alpha_beta_pvs(g, reduced, ply + 1, alpha, beta, false, 0);
            g.undo_move();

            if (search_aborted) return 0.0;

            if (maximising && null_val >= beta)
                return beta;
            if (!maximising && null_val <= alpha)
                return alpha;
        }
    }

    double best = maximising ? -1e300 : 1e300;
    Move best_move{0, 0, 'P'};

    auto moves = g.get_possible_moves_for_player(g.current_player);
    moves = filter_search_moves(moves, g, g.current_player);

    // Depth-adaptive move cap: cut branching at deep plies to enable depth 7+.
    // NN ordering puts the best moves first, so the tail is safe to cut.
    // Keep ply 0-3 wide (48) to not miss important root/near-root moves.
    int effective_cap = move_cap;
    if (ply >= 4)
        effective_cap = std::min(effective_cap, 20);
    if (ply >= 6)
        effective_cap = std::min(effective_cap, 14);

    // Enhanced move ordering: TT best move → killers → history
    order_moves_enhanced(moves, g, ply, maximising);

    // NN-based move ordering: evaluate all children with native NN and sort
    if (native_nn_.is_loaded() && depth >= nn_ordering_min_depth_)
    {
        order_moves_by_native_nn(moves, g, maximising);
    }
    else if (depth == 2)
    {
        order_moves_by_child_eval_inplace(moves, g, maximising);
    }

    // Cap AFTER ordering so we keep the strongest candidates rather than
    // truncating by simple type/coordinate order.
    if ((int)moves.size() > effective_cap)
    {
        moves.resize((size_t)effective_cap);
    }

    if (moves.empty())
        return evaluate(g);

    if (depth == 1)
    {
        auto vals = evaluate_children_depth1_batched(g, moves, maximising);
        for (size_t i = 0; i < moves.size(); ++i)
        {
            const auto &m = moves[i];
            double v = vals[i];

            if (maximising)
            {
                if (v > best || (v == best && move_less(m, best_move)))
                {
                    best = v;
                    best_move = m;
                }
                alpha = std::max(alpha, best);
                if (alpha >= beta) break;
            }
            else
            {
                if (v < best || (v == best && move_less(m, best_move)))
                {
                    best = v;
                    best_move = m;
                }
                beta = std::min(beta, best);
                if (alpha >= beta) break;
            }
        }
        // Store in TT
        TTEntry e;
        e.depth = depth;
        e.value = best;
        e.best = best_move;
        if (best <= a0) e.flag = 2;
        else if (best >= b0) e.flag = 1;
        else e.flag = 0;
        tt[key] = e;
        return best;
    }

    // ---- Futility pruning: skip this node if static eval is far from alpha/beta ----
    // Only at shallow depth, not in PV nodes (wide window)
    static constexpr double FUTILITY_MARGIN_D1 = 3.0;
    static constexpr double FUTILITY_MARGIN_D2 = 6.0;
    bool futility_prune = false;
    if (depth <= 2 && !wide_window && ply > 0)
    {
        double static_eval = evaluate(g);
        double margin = (depth == 1) ? FUTILITY_MARGIN_D1 : FUTILITY_MARGIN_D2;
        if (maximising && static_eval + margin <= alpha)
            futility_prune = true;
        if (!maximising && static_eval - margin >= beta)
            futility_prune = true;
    }

    int move_count = 0;

    for (auto &m : moves)
    {
        if (!g.is_move_legal(m, g.current_player))
            continue;

        // Futility pruning: skip non-first moves at shallow depth
        if (futility_prune && move_count > 0 && m.t != 'R')
        {
            // Still search scoring-potential moves (rocks can create/block area)
            continue;
        }

        int mover = g.current_player;
        g.do_move(m, mover);

        // Check for immediate win — never reduce/prune these
        bool gives_win = (g.winner != -1);

        double v;
        if (move_count == 0)
        {
            // First move: full window search
            v = alpha_beta_pvs(g, depth - 1, ply + 1, alpha, beta, true, extensions_left);
        }
        else
        {
            // ---- Late Move Reductions (LMR) ----
            // Reduce search depth for late quiet moves that are unlikely to be best
            int lmr_reduction = 0;
            if (depth >= 3 && move_count >= 4 && !gives_win && m.t != 'R')
            {
                // Reduce more for later moves
                lmr_reduction = 1;
                if (move_count >= 8)
                    lmr_reduction = 2;
                // Don't reduce below depth 1
                if (depth - 1 - lmr_reduction < 1)
                    lmr_reduction = std::max(0, depth - 2);
            }

            int search_depth = depth - 1 - lmr_reduction;

            // PVS: null-window search
            if (maximising)
            {
                v = alpha_beta_pvs(g, search_depth, ply + 1, alpha, alpha + 0.01, true, extensions_left);
                // Re-search at full depth if reduced search improved alpha
                if (lmr_reduction > 0 && v > alpha && !search_aborted)
                    v = alpha_beta_pvs(g, depth - 1, ply + 1, alpha, alpha + 0.01, true, extensions_left);
                if (v > alpha && v < beta && !search_aborted)
                    v = alpha_beta_pvs(g, depth - 1, ply + 1, alpha, beta, true, extensions_left);
            }
            else
            {
                v = alpha_beta_pvs(g, search_depth, ply + 1, beta - 0.01, beta, true, extensions_left);
                if (lmr_reduction > 0 && v < beta && !search_aborted)
                    v = alpha_beta_pvs(g, depth - 1, ply + 1, beta - 0.01, beta, true, extensions_left);
                if (v < beta && v > alpha && !search_aborted)
                    v = alpha_beta_pvs(g, depth - 1, ply + 1, alpha, beta, true, extensions_left);
            }
        }
        g.undo_move();
        move_count++;

        if (search_aborted) return 0.0;

        // Apply pass penalty
        if (m.t == 'P')
        {
            if (maximising)
                v -= pass_penalty;
            else
                v += pass_penalty;
        }

        if (maximising)
        {
            if (v > best || (v == best && move_less(m, best_move)))
            {
                best = v;
                best_move = m;
            }
            alpha = std::max(alpha, best);
            if (alpha >= beta)
            {
                update_killers(ply, m);
                update_history(mover, m, depth);
                break;
            }
        }
        else
        {
            if (v < best || (v == best && move_less(m, best_move)))
            {
                best = v;
                best_move = m;
            }
            beta = std::min(beta, best);
            if (alpha >= beta)
            {
                update_killers(ply, m);
                update_history(mover, m, depth);
                break;
            }
        }
    }

    // Store in TT
    TTEntry e;
    e.depth = depth;
    e.value = best;
    e.best = best_move;
    if (best <= a0)
        e.flag = 2; // upper
    else if (best >= b0)
        e.flag = 1; // lower
    else
        e.flag = 0; // exact
    tt[key] = e;
    return best;
}

// ---- Original alpha_beta (kept for choose_move and choose_move_with_values backward compat) ----

double AlphaBetaEngine::alpha_beta(GameState &g, int depth, double alpha, double beta)
{
    nodes_searched++;

    // Time check every 256 nodes (for iterative deepening)
    if (search_time_limit_ms > 0 && (nodes_searched & 255) == 0 && is_time_up())
    {
        search_aborted = true;
        return 0.0;
    }

    // Save original alpha/beta BEFORE TT adjustment for correct flag determination
    double a0 = alpha;
    double b0 = beta;

    TTKey key = g.tt_key();
    auto it = tt.find(key);
    if (it != tt.end() && it->second.depth >= depth)
    {
        const TTEntry &e = it->second;
        if (e.flag == 0)
            return e.value;
        if (e.flag == 1)
            alpha = std::max(alpha, e.value);
        else if (e.flag == 2)
            beta = std::min(beta, e.value);
        if (alpha >= beta)
            return e.value;
    }

    // Terminal positions: use depth-adjusted values so the engine
    // prefers quick wins and delayed losses (doesn't "give up").
    if (g.winner != -1)
    {
        if (use_heuristic_eval)
            return (g.winner == root_player) ? (1000.0 + depth) : (-1000.0 - depth);
        else
            return (g.winner == root_player) ? prob_to_value(1.0) + depth : prob_to_value(0.0) - depth;
    }
    if (depth <= 0)
        return evaluate(g);

    bool maximising = (g.current_player == root_player);
    double best = maximising ? -1e300 : 1e300;
    Move best_move{0, 0, 'P'};

    auto moves = g.get_possible_moves_for_player(g.current_player);
    moves = filter_search_moves(moves, g, g.current_player);
    order_moves_inplace(moves);

    // Use a one-shot batched child evaluation for move ordering at depth==2.
    // This adds one model call but can improve pruning significantly.
    if (depth == 2)
    {
        order_moves_by_child_eval_inplace(moves, g, maximising);
    }

    // Cap after ordering, mirroring the PVS pipeline redesign.
    if ((int)moves.size() > move_cap)
        moves.resize((size_t)move_cap);

    if (moves.empty())
        return evaluate(g);

    if (depth == 1)
    {
        auto vals = evaluate_children_depth1_batched(g, moves, maximising);
        for (size_t i = 0; i < moves.size(); ++i)
        {
            const auto &m = moves[i];
            double v = vals[i];

            if (maximising)
            {
                if (v > best || (v == best && move_less(m, best_move)))
                {
                    best = v;
                    best_move = m;
                }
                alpha = std::max(alpha, best);
                if (alpha >= beta)
                    break;
            }
            else
            {
                if (v < best || (v == best && move_less(m, best_move)))
                {
                    best = v;
                    best_move = m;
                }
                beta = std::min(beta, best);
                if (alpha >= beta)
                    break;
            }
        }
    }
    else
    {
        for (auto &m : moves)
        {
            int mover = g.current_player;
            g.do_move(m, mover);
            double v = alpha_beta(g, depth - 1, alpha, beta);
            g.undo_move();

            if (search_aborted) return 0.0;

            if (m.t == 'P')
            {
                if (maximising)
                    v -= pass_penalty;
                else
                    v += pass_penalty;
            }

            if (maximising)
            {
                if (v > best || (v == best && move_less(m, best_move)))
                {
                    best = v;
                    best_move = m;
                }
                alpha = std::max(alpha, best);
                if (alpha >= beta)
                    break;
            }
            else
            {
                if (v < best || (v == best && move_less(m, best_move)))
                {
                    best = v;
                    best_move = m;
                }
                beta = std::min(beta, best);
                if (alpha >= beta)
                    break;
            }
        }
    }

    TTEntry e;
    e.depth = depth;
    e.value = best;
    e.best = best_move;
    if (best <= a0)
        e.flag = 2; // upper
    else if (best >= b0)
        e.flag = 1; // lower
    else
        e.flag = 0; // exact
    tt[key] = e;
    return best;
}
