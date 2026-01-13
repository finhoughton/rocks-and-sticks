#include "mcts.hpp"
#include "alphabeta.hpp"

PYBIND11_MODULE(players_ext, m)
{
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
        .def("explain_illegal_move", &GameState::explain_illegal_move, py::arg("move"), py::arg("player_number"))
        .def("do_move", &GameState::do_move, py::arg("move"), py::arg("player_number"))
        .def("undo_move", &GameState::undo_move)
        .def("state_key", &GameState::state_key)
        .def("set_current_player0", &GameState::set_current_player0)
        .def_readwrite("current_player", &GameState::current_player);

    py::class_<MCTSEngine>(m, "MCTSEngine")
        .def(py::init<int, double>(), py::arg("seed") = 0, py::arg("c_puct") = 1.41421356)
        .def("choose_move", &MCTSEngine::choose_move, py::arg("root"), py::arg("n_rollouts") = 1000)
        .def("set_c_puct", &MCTSEngine::set_c_puct)
        .def("set_verbose", &MCTSEngine::set_verbose, py::arg("verbose"))
        .def("set_progressive_widening", &MCTSEngine::set_progressive_widening)
        .def("set_rave_k", &MCTSEngine::set_rave_k)
        .def("set_prior_eval_cap", &MCTSEngine::set_prior_eval_cap)
        .def("set_max_sim_depth", &MCTSEngine::set_max_sim_depth)
        .def("clear_root_priors", &MCTSEngine::clear_root_priors)
        .def("set_exploration", &MCTSEngine::set_exploration, py::arg("dirichlet_alpha"), py::arg("dirichlet_epsilon"), py::arg("temperature"), py::arg("temperature_moves"))
        .def("set_model_checkpoint", &MCTSEngine::set_model_checkpoint, py::arg("path"), py::arg("device") = "cpu")
        .def("reset_search", &MCTSEngine::reset_search)
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
        .def("set_model_checkpoint", &AlphaBetaEngine::set_model_checkpoint, py::arg("path"), py::arg("device") = "cpu")
        .def("clear_stats", &AlphaBetaEngine::clear_stats)
        .def("get_profile_stats", &AlphaBetaEngine::get_profile_stats);
}