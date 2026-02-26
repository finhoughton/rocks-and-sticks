#pragma once

#include "gnn_helpers.hpp"
#include "heuristic_eval.hpp"

class MCTSEngine
{
public:
    MCTSEngine(int seed = 0, double c_puct_ = 1.41421356);

    // Reseed the engine RNG (used for deterministic tie-breaks/noise).
    void set_seed(int seed);

    Move choose_move(const GameState &root, int n_rollouts);

    void set_c_puct(double v);
    void set_verbose(bool v);
    void set_verbose_level(int v);
    void set_progressive_widening(double c, double alpha);
    void set_rave_k(double v);
    // Set prior mixing and scaling parameters from Python.
    void py_set_prior_params(double mix_uniform, double scale);
    void set_prior_eval_cap(int v);
    void set_max_sim_depth(int v);
    void clear_root_priors();

    void set_exploration(double alpha, double epsilon, double temp, int temp_moves);

    void set_model_checkpoint(const std::string &path, const std::string &device);

    // Load a PolicyValueNet checkpoint to use its policy head for priors.
    // Value evaluation (leaf values) continues to use the GNNEval loaded via set_model_checkpoint.
    void set_policy_checkpoint(const std::string &path, const std::string &device);

    // If true, use the neural net to estimate the leaf value (AlphaZero-style)
    // instead of relying purely on random rollouts.
    void set_use_nn_value(bool v);

    // If true, use heuristic_evaluate at leaf nodes instead of random rollouts.
    // This is faster than NN and FAR better than random rollouts.
    // When both use_nn_value and use_heuristic_rollout are set, NN takes priority.
    void set_use_heuristic_rollout(bool v);

    // Set the sigmoid temperature for converting heuristic scores to probabilities.
    // Higher = more uncertain (closer to 0.5). Default = 6.0.
    void set_heuristic_temperature(double t);

    // Set linear calibration parameters (a,b) and enable/disable calibration.
    void py_set_value_calibration(double a, double b, bool enabled);

    void reset_search();

    std::uint64_t get_current_root_key() const;

    // Return per-move visit counts at the provided root.
    // Each element is a dict: {"x": int, "y": int, "t": str, "visits": int}
    py::list get_root_visit_stats_py(const GameState &root);

    // Return per-move priors at the provided root.
    // Each element is a dict: {"x": int, "y": int, "t": str, "prior": float}
    py::list get_root_priors_py(const GameState &root);
    py::list get_root_values_py(const GameState &root);
    // takes iterable of (x,y,t,prior)
    void set_root_priors_py(py::iterable priors);

    void clear_stats();
    py::dict get_profile_stats();

    void advance_root(const GameState &game);
    void prune_tables(int max_states);

private:
    std::mt19937 rng;
    double c_puct;
    // Verbosity level: 0 = silent, 1 = summaries (choose_move), 2 = detailed (prior eval, encodings)
    int verbose_level = 0;
    bool use_nn_value = true;
    bool use_heuristic_rollout = false;
    double heuristic_temperature = 6.0;
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
    // Leaf value cache: value in [-1,1] from the perspective of the player to move at that state.
    std::unordered_map<TTKey, double, TTKeyHash> V;
    std::uint64_t _root_key = 0;
    std::unordered_map<MoveKey, double, MoveKeyHash> root_priors;

    players_ext_internal::PyGNNModules py_mods;

    py::object model_override = py::none();
    std::string model_device = "cpu";

    py::object policy_model_override = py::none();
    std::string policy_model_device = "cpu";

    double dirichlet_alpha = 0.0;
    double dirichlet_epsilon = 0.0;
    double temperature = 0.0;
    int temperature_moves = 0;
    double prior_mix_uniform = 0.04;
    double prior_scale = 1.0;
    double rock_prior_bonus_connected = 1.5;
    double rock_prior_bonus_disconnected = 0.06;
    double stick_between_opp_rocks_bonus = 0.4;

    // Optional linear calibration applied to NN leaf values: v' = a*v + b
    bool value_calibration_enabled = false;
    double value_calibration_a = 1.0;
    double value_calibration_b = 0.0;

    void set_prior_params(double mix_uniform, double scale)
    {
        prior_mix_uniform = mix_uniform;
        prior_scale = scale;
    }

        std::unordered_map<std::uint64_t, py::object> enc_cache;
    static constexpr size_t ENC_CACHE_MAX = 4096;

    double total_encode_time = 0.0;
    double total_model_time = 0.0;

    // Profiling counters for verifying NN usage.
    std::size_t prior_model_calls = 0;
    std::size_t prior_model_batch_items = 0;
    std::size_t value_model_calls = 0;
    std::size_t value_model_batch_items = 0;

    // Separate counters for policy-head priors.
    std::size_t policy_prior_calls = 0;
    std::size_t policy_prior_items = 0;
    double policy_total_time = 0.0;
};
