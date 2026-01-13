#include "alphabeta.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <unordered_set>

AlphaBetaEngine::AlphaBetaEngine(int seed, double pass_penalty)
    : rng((seed == 0) ? std::mt19937(std::random_device{}()) : std::mt19937((std::uint32_t)seed)),
      pass_penalty(pass_penalty)
{
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

    // Caches depend on the model.
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
    // Match players/ai.py::_evaluate_with_gnn scaling.
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
            py_mods, model_override, model_device, encs, &total_model_time, &model_calls, &model_batch_items);
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
            py_mods, model_override, model_device, encs, &total_model_time, &model_calls, &model_batch_items);
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
        // Defensive: fall back to 0.5 for missing outputs.
        probs.resize((size_t)py::len(encs), 0.5);
    }
    for (auto &p : probs)
        p = clamp_prob(p);
    return probs;
}

double AlphaBetaEngine::evaluate(GameState &g)
{
    TTKey key = g.tt_key();
    auto it = eval_cache.find(key);
    if (it != eval_cache.end())
        return it->second;

    double p = gnn_prob_root(g);
    double v = prob_to_value(p);
    eval_cache[key] = v;
    return v;
}

std::vector<double> AlphaBetaEngine::evaluate_children_depth1_batched(
    GameState &g, const std::vector<Move> &moves, bool parent_maximising)
{
    // For depth==1, children are evaluated at depth 0.
    // We batch all uncached leaf evals into a single model call.
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

double AlphaBetaEngine::alpha_beta(GameState &g, int depth, double alpha, double beta)
{
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

    if (depth <= 0 || g.winner != -1)
        return evaluate(g);

    bool maximising = (g.current_player == root_player);
    double best = maximising ? -1e300 : 1e300;
    Move best_move{0, 0, 'P'};
    double a0 = alpha;
    double b0 = beta;

    auto moves = g.get_possible_moves_for_player(g.current_player);
    moves = filter_search_moves(moves, g, g.current_player);
    order_moves_inplace(moves);
    if ((int)moves.size() > move_cap)
        moves.resize((size_t)move_cap);

    // Use a one-shot batched child evaluation for move ordering at depth==2.
    // This adds one model call but can improve pruning significantly.
    if (depth == 2)
    {
        order_moves_by_child_eval_inplace(moves, g, maximising);
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
