#include "mcts.hpp"
#include "alphabeta.hpp"
#include "heuristic_eval.hpp"
#include <pybind11/numpy.h>

PYBIND11_MODULE(players_ext, m)
{
    // Standalone heuristic evaluation (fast C++ implementation)
    m.def("heval_evaluate", &heval_evaluate, py::arg("game"), py::arg("perspective"),
          "Fast heuristic evaluation: returns score from perspective's point of view");
    m.def("heval_score_to_prob", &heval_score_to_prob, py::arg("score"), py::arg("temperature") = 6.0,
          "Convert heuristic score to win probability via sigmoid(score/temperature)");

    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def_readwrite("x", &Move::x)
        .def_readwrite("y", &Move::y)
        .def_readwrite("t", &Move::t)
        .def("__repr__", [](const Move &mv)
             {
            std::ostringstream os;
            os << "players_ext.Move(" << mv.x << ", " << mv.y << ", '" << mv.t << "')";
            return os.str(); });

    py::class_<GameState>(m, "GameState")
        .def(py::init<>())
        .def("get_possible_moves", &GameState::get_possible_moves_for_player, py::arg("player_number"))
        .def("is_move_legal", &GameState::is_move_legal, py::arg("move"), py::arg("player_number"))
        .def("can_apply_move", &GameState::can_apply_move, py::arg("move"), py::arg("player_number"))
        .def("explain_illegal_move", &GameState::explain_illegal_move, py::arg("move"), py::arg("player_number"))
        .def("do_move", &GameState::do_move, py::arg("move"), py::arg("player_number"))
        .def("undo_move", &GameState::undo_move)
        .def("state_key", &GameState::state_key)
        .def("set_current_player0", &GameState::set_current_player0)
        .def_readwrite("current_player", &GameState::current_player)
        .def_readonly("winner", &GameState::winner);

    py::class_<MCTSEngine>(m, "MCTSEngine")
        .def(py::init<int, double>(), py::arg("seed") = 0, py::arg("c_puct") = 1.41421356)
        .def("set_seed", &MCTSEngine::set_seed, py::arg("seed"))
        .def("choose_move", &MCTSEngine::choose_move, py::arg("root"), py::arg("n_rollouts") = 1000)
        .def("set_c_puct", &MCTSEngine::set_c_puct)
        .def("set_verbose", &MCTSEngine::set_verbose, py::arg("verbose"))
        .def("set_verbose_level", &MCTSEngine::set_verbose_level, py::arg("level"))
        .def("set_use_nn_value", &MCTSEngine::set_use_nn_value, py::arg("use_nn_value"))
        .def("set_use_heuristic_rollout", &MCTSEngine::set_use_heuristic_rollout, py::arg("use_heuristic_rollout"))
        .def("set_heuristic_temperature", &MCTSEngine::set_heuristic_temperature, py::arg("temperature"))
        .def("set_progressive_widening", &MCTSEngine::set_progressive_widening)
        .def("set_rave_k", &MCTSEngine::set_rave_k)
        .def("set_prior_params", &MCTSEngine::py_set_prior_params)
        .def("set_prior_eval_cap", &MCTSEngine::set_prior_eval_cap)
        .def("set_max_sim_depth", &MCTSEngine::set_max_sim_depth)
        .def("clear_root_priors", &MCTSEngine::clear_root_priors)
        .def("set_exploration", &MCTSEngine::set_exploration, py::arg("dirichlet_alpha"), py::arg("dirichlet_epsilon"), py::arg("temperature"), py::arg("temperature_moves"))
        .def("set_model_checkpoint", &MCTSEngine::set_model_checkpoint, py::arg("path"), py::arg("device") = "cpu")
        .def("set_policy_checkpoint", &MCTSEngine::set_policy_checkpoint, py::arg("path"), py::arg("device") = "cpu")
        .def("set_value_calibration", &MCTSEngine::py_set_value_calibration, py::arg("a"), py::arg("b"), py::arg("enabled") = true)
        .def("reset_search", &MCTSEngine::reset_search)
        .def("get_root_priors", &MCTSEngine::get_root_priors_py, py::arg("root"))
        .def("get_root_values", &MCTSEngine::get_root_values_py, py::arg("root"))
        .def("set_root_priors", &MCTSEngine::set_root_priors_py)
        .def("get_current_root_key", &MCTSEngine::get_current_root_key)
        .def("get_root_visit_stats", &MCTSEngine::get_root_visit_stats_py, py::arg("root"))
        .def("clear_stats", &MCTSEngine::clear_stats)
        .def("get_profile_stats", &MCTSEngine::get_profile_stats)
        .def("advance_root", &MCTSEngine::advance_root, py::arg("game"))
        .def("prune_tables", &MCTSEngine::prune_tables, py::arg("max_states"));

    py::class_<AlphaBetaEngine>(m, "AlphaBetaEngine")
        .def(py::init<int, double>(), py::arg("seed") = 0, py::arg("pass_penalty") = 1.2)
        .def("choose_move", &AlphaBetaEngine::choose_move, py::arg("root"), py::arg("depth") = 3, py::arg("move_cap") = 48)
        .def("choose_move_iterative", &AlphaBetaEngine::choose_move_iterative,
             py::arg("root"), py::arg("max_depth") = 20, py::arg("time_limit_ms") = 5000, py::arg("move_cap") = 48)
        .def("set_model_checkpoint", &AlphaBetaEngine::set_model_checkpoint, py::arg("path"), py::arg("device") = "cpu")
        .def("set_use_heuristic", &AlphaBetaEngine::set_use_heuristic, py::arg("use_heuristic"))
        .def("load_native_model", &AlphaBetaEngine::load_native_model, py::arg("path"),
             "Load native C++ NN weights from binary file (no Python/GIL needed for inference)")
        .def("has_native_model", &AlphaBetaEngine::has_native_model)
        .def("native_nn_prob", [](AlphaBetaEngine &self, GameState &g, int perspective) {
            // Return P(perspective wins) directly from native NN
            self.root_player = perspective;
            if (g.winner != -1)
                return (g.winner == perspective) ? 1.0 : 0.0;
            if (g.connected_points.empty() && g.rocks.empty())
                return 0.5;
            NNInput input = self.encode_for_native_nn_public(g);
            float logit = self.native_nn_raw_logit(input);
            float prob = 1.0f / (1.0f + std::exp(-logit));
            if (g.current_player != perspective)
                prob = 1.0f - prob;
            return (double)prob;
        }, py::arg("game"), py::arg("perspective"),
             "Return P(perspective wins) from native NN for the given position")
        .def("native_nn_raw_logit", [](AlphaBetaEngine &self,
                py::array_t<float> node_feats,
                py::array_t<int32_t> edge_src,
                py::array_t<int32_t> edge_dst,
                py::array_t<float> edge_attr,
                py::array_t<float> global_feats,
                int num_nodes) {
            NNInput input;
            input.num_nodes = num_nodes;
            auto nf = node_feats.unchecked<1>();
            input.node_feats.assign(nf.data(0), nf.data(0) + nf.shape(0));
            auto es = edge_src.unchecked<1>();
            input.edge_src.assign(es.data(0), es.data(0) + es.shape(0));
            auto ed = edge_dst.unchecked<1>();
            input.edge_dst.assign(ed.data(0), ed.data(0) + ed.shape(0));
            input.num_edges = (int)input.edge_src.size();
            auto ea = edge_attr.unchecked<1>();
            input.edge_attr.assign(ea.data(0), ea.data(0) + ea.shape(0));
            auto gf = global_feats.unchecked<1>();
            input.global_feats.assign(gf.data(0), gf.data(0) + gf.shape(0));
            return self.native_nn_raw_logit(input);
        }, "Feed raw tensors into native NN forward pass and return raw logit")

        .def("set_nn_ordering_depth", &AlphaBetaEngine::set_nn_ordering_depth, py::arg("min_depth"),
             "Set minimum AB depth at which NN-based move ordering is used (default: 2)")
        .def("get_nn_ordering_depth", &AlphaBetaEngine::get_nn_ordering_depth)
        .def("heuristic_eval", [](AlphaBetaEngine &self, GameState &g, int perspective) {
            self.root_player = perspective;
            return self.heuristic_evaluate(g);
        }, py::arg("game"), py::arg("perspective"))
        .def("choose_move_with_values", [](AlphaBetaEngine &self, const GameState &root, int depth, int move_cap) {
            auto results = self.choose_move_with_values(root, depth, move_cap);
            py::list out;
            for (auto &[m, v] : results) {
                py::dict d;
                d["x"] = m.x;
                d["y"] = m.y;
                d["t"] = std::string(1, m.t);
                d["value"] = v;
                out.append(d);
            }
            return out;
        }, py::arg("root"), py::arg("depth") = 3, py::arg("move_cap") = 48)
        .def("clear_stats", &AlphaBetaEngine::clear_stats)
        .def("get_profile_stats", &AlphaBetaEngine::get_profile_stats)
        .def_readonly("last_depth_completed", &AlphaBetaEngine::last_depth_completed)
        .def_readonly("last_nodes_searched", &AlphaBetaEngine::last_nodes_searched);
}