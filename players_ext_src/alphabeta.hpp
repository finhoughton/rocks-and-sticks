#pragma once

#include "gnn_helpers.hpp"
#include "nn_eval.hpp"

class AlphaBetaEngine
{
public:
    AlphaBetaEngine(int seed = 0, double pass_penalty = 1.2);

    Move choose_move(const GameState &root, int depth = 3, int move_cap = 48);
    Move choose_move_iterative(const GameState &root, int max_depth, int time_limit_ms, int move_cap = 48);
    std::vector<std::pair<Move, double>> choose_move_with_values(const GameState &root, int depth, int move_cap);
    double heuristic_evaluate(GameState &g);

    void set_model_checkpoint(const std::string &path, const std::string &device);
    void set_use_heuristic(bool v);

    // Native C++ NN inference (no GIL needed)
    bool load_native_model(const std::string &path);
    bool has_native_model() const { return native_nn_.is_loaded(); }
    void set_nn_ordering_depth(int min_depth);
    int get_nn_ordering_depth() const { return nn_ordering_min_depth_; }
    double native_nn_evaluate(GameState &g);
    float native_nn_raw_logit(const NNInput &input) const { return native_nn_.evaluate(input); }
    NNInput encode_for_native_nn_public(GameState &g) const { return encode_for_native_nn(g); }
    void clear_stats();
    py::dict get_profile_stats();

    int root_player = 0;

    // Stats from last iterative-deepening search
    int last_depth_completed = 0;
    size_t last_nodes_searched = 0;

private:
    struct TTEntry
    {
        int depth = 0;
        double value = 0.0;
        // 0 exact, 1 lower bound, 2 upper bound
        int flag = 0;
        Move best{0, 0, 'P'};
    };

    // ---- Handcrafted heuristic evaluation (no NN) ----
    struct TacticalInfo
    {
        double max_immediate_gain = 0.0;
        int scoring_move_count = 0;
        int stick_move_count = 0;
        double top3_gain_sum = 0.0;
        int bad_closure_count = 0;
        double potential_area = 0.0;
        double blocking_power = 0.0;
        double rock_value = 0.0;
        double best_reply_gain = 0.0;
    };

    TacticalInfo compute_tactical(GameState &g, int player_number);
    static double scored_gain_from_area(int area2);
    bool can_place(const Node *n, int player_number) const;
    int closure_area2_for_stick(GameState &g, const Move &m);
    // ---------------------------------------------------

    // ---- Search enhancements ----
    static constexpr int MAX_PLY = 64;
    static constexpr int NULL_MOVE_R = 2;
    static constexpr int HISTORY_MAX = 1 << 20;  // mask for history table

    // Killer moves: 2 slots per ply
    Move killers[MAX_PLY][2];

    // History heuristic: indexed by [player][move_hash & mask]
    int history[2][HISTORY_MAX];

    // Node counter for iterative deepening stats
    size_t nodes_searched = 0;

    // Time management for iterative deepening
    std::chrono::steady_clock::time_point search_start;
    int search_time_limit_ms = 0;
    bool search_aborted = false;

    bool is_time_up() const;
    static std::uint32_t move_hash(const Move &m);
    void update_killers(int ply, const Move &m);
    bool is_killer(int ply, const Move &m) const;
    void update_history(int player, const Move &m, int depth);
    void order_moves_enhanced(std::vector<Move> &moves, GameState &g, int ply, bool maximising);
    bool has_scoring_move(GameState &g, int player);

    double alpha_beta_pvs(GameState &g, int depth, int ply, double alpha, double beta,
                          bool allow_null_move, int extensions_left);
    double quiescence(GameState &g, double alpha, double beta, int ply, int qd);
    // ---- End search enhancements ----

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

    // Native NN helpers
    NNInput encode_for_native_nn(GameState &g) const;
    void order_moves_by_native_nn(std::vector<Move> &moves, GameState &g, bool maximising);

    std::mt19937 rng;
    double pass_penalty = 1.2;
    int move_cap = 48;
    int last_root_player = -1;
    bool use_heuristic_eval = false;
    int nn_ordering_min_depth_ = 2;  // use NN ordering at depth >= this

    NativeNNEval native_nn_;

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
