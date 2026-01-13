#pragma once

#include "gnn_helpers.hpp"

class MCTSEngine
{
public:
    MCTSEngine(int seed = 0, double c_puct_ = 1.41421356);

    Move choose_move(const GameState &root, int n_rollouts);

    void set_c_puct(double v);
    void set_verbose(bool v);
    void set_progressive_widening(double c, double alpha);
    void set_rave_k(double v);
    void set_prior_eval_cap(int v);
    void set_max_sim_depth(int v);
    void clear_root_priors();

    void set_exploration(double alpha, double epsilon, double temp, int temp_moves);

    void set_model_checkpoint(const std::string &path, const std::string &device);

    void reset_search();

    std::uint64_t get_current_root_key() const;

    // Return per-move visit counts at the provided root.
    // Each element is a dict: {"x": int, "y": int, "t": str, "visits": int}
    py::list get_root_visit_stats_py(const GameState &root);

    // takes iterable of (x,y,t,prior)
    void set_root_priors_py(py::iterable priors);

    void clear_stats();
    py::dict get_profile_stats();

    void advance_root(const GameState &game);
    void prune_tables(int max_states);

private:
    std::mt19937 rng;
    double c_puct;
    bool verbose = true;
    double progressive_widening_c;
    double progressive_widening_alpha;
    double rave_k;
    int prior_eval_cap;
    int max_sim_depth;
    bool check_forced_losses;
    int tactical_root_limit;

    std::unordered_map<TTKey, int, TTKeyHash> Ns;
    std::unordered_map<EdgeKey, int, EdgeKeyHash> Nsa;
    std::unordered_map<EdgeKey, double, EdgeKeyHash> Wsa;
    std::unordered_map<EdgeKey, double, EdgeKeyHash> Psa;
    std::unordered_map<EdgeKey, int, EdgeKeyHash> N_amaf;
    std::unordered_map<EdgeKey, double, EdgeKeyHash> W_amaf;
    std::unordered_map<TTKey, std::vector<Move>, TTKeyHash> legal_moves;
    std::unordered_map<TTKey, int, TTKeyHash> expanded_count;
    std::uint64_t _root_key = 0;
    std::unordered_map<MoveKey, double, MoveKeyHash> root_priors;

    players_ext_internal::PyGNNModules py_mods;

    py::object model_override = py::none();
    std::string model_device = "cpu";

    double dirichlet_alpha = 0.0;
    double dirichlet_epsilon = 0.0;
    double temperature = 0.0;
    int temperature_moves = 0;

    std::unordered_map<std::uint64_t, py::object> enc_cache;
    static constexpr size_t ENC_CACHE_MAX = 4096;

    double total_encode_time = 0.0;
    double total_model_time = 0.0;
};
