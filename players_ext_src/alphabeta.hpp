#pragma once

#include "gnn_helpers.hpp"

class AlphaBetaEngine
{
public:
    AlphaBetaEngine(int seed = 0, double pass_penalty = 1.2);

    Move choose_move(const GameState &root, int depth = 3, int move_cap = 48);

    void set_model_checkpoint(const std::string &path, const std::string &device);

    void clear_stats();
    py::dict get_profile_stats();

private:
    struct TTEntry
    {
        int depth = 0;
        double value = 0.0;
        // 0 exact, 1 lower bound, 2 upper bound
        int flag = 0;
        Move best{0, 0, 'P'};
    };

    static bool move_less(const Move &a, const Move &b);
    static int move_type_rank(const Move &m);
    static void order_moves_inplace(std::vector<Move> &moves);
    static bool rock_is_search_worthy(GameState &g, const Move &m);
    static std::vector<Move> filter_search_moves(const std::vector<Move> &moves, GameState &g, int player);
    static double clamp_prob(double p);
    static double prob_to_value(double prob);

    void ensure_py_modules();
    py::object encode_state(GameState &g);
    double gnn_prob_root(GameState &g);
    std::vector<double> gnn_probs_root_for_encodings(const py::list &encs);
    double evaluate(GameState &g);
    std::vector<double> evaluate_children_depth1_batched(GameState &g, const std::vector<Move> &moves, bool parent_maximising);
    void order_moves_by_child_eval_inplace(std::vector<Move> &moves, GameState &g, bool parent_maximising);
    double alpha_beta(GameState &g, int depth, double alpha, double beta);

    std::mt19937 rng;
    double pass_penalty = 1.2;
    int move_cap = 48;
    int root_player = 0;
    int last_root_player = -1;

    std::unordered_map<TTKey, TTEntry, TTKeyHash> tt;
    std::unordered_map<TTKey, double, TTKeyHash> eval_cache;

    players_ext_internal::PyGNNModules py_mods;

    py::object model_override = py::none();
    std::string model_device = "cpu";

    std::unordered_map<std::uint64_t, py::object> enc_cache;
    static constexpr size_t ENC_CACHE_MAX = 4096;

    double total_encode_time = 0.0;
    double total_model_time = 0.0;

    size_t model_calls = 0;
    size_t model_batch_items = 0;
};
