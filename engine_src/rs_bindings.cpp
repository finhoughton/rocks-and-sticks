// Python bindings for the rs engine (module name: rs_engine).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <string>

#include "rs_parallel.hpp"

namespace py = pybind11;
using namespace rs;

namespace
{

const char *TYPE_NAMES[10] = {"N", "E", "NE", "SE", "NW", "SW", "W", "S", "R", "P"};

int type_from_name(const std::string &t)
{
    for (int i = 0; i < 10; ++i)
        if (t == TYPE_NAMES[i])
            return i;
    throw std::invalid_argument("unknown move type: " + t);
}

py::tuple move_to_py(Move m)
{
    int t = move_type(m);
    if (t == T_PASS)
        return py::make_tuple(0, 0, "P");
    return py::make_tuple(x_of(move_idx(m)), y_of(move_idx(m)), TYPE_NAMES[t]);
}

Move move_from_py(int x, int y, const std::string &t)
{
    int ty = type_from_name(t);
    if (ty == T_PASS)
        return MOVE_PASS;
    if (std::abs(x) >= OFF - SAFE || std::abs(y) >= OFF - SAFE)
        throw std::out_of_range("coordinate outside engine board");
    return make_move(idx_of(x, y), ty);
}

class Engine
{
  public:
    State state;
    ParallelSearcher par;
    Searcher &searcher; // the main worker

    explicit Engine(int tt_bits = 21, int threads = 1) : par(tt_bits, threads), searcher(par.main()) { state.set_start(); }

    void set_threads(int n) { par.set_threads(n); }

    void reset() { state.set_start(); }

    // sticks: (x, y, dir name) each listed once; rocks: (x, y, owner)
    void set_position(const std::vector<std::tuple<int, int, std::string>> &sticks,
                      const std::vector<std::tuple<int, int, int>> &rocks, int cp, int turn,
                      std::array<int, 2> scores, std::array<int, 2> num_rocks, int winner)
    {
        state.reset();
        for (auto &st : sticks)
        {
            int d = type_from_name(std::get<2>(st));
            if (d >= 8)
                throw std::invalid_argument("stick needs a direction");
            state.add_stick_raw(idx_of(std::get<0>(st), std::get<1>(st)), d);
        }
        for (auto &r : rocks)
            state.add_rock_raw(idx_of(std::get<0>(r), std::get<1>(r)), std::get<2>(r));
        state.cp = cp;
        state.turn = turn;
        state.scores[0] = scores[0];
        state.scores[1] = scores[1];
        state.num_rocks[0] = num_rocks[0];
        state.num_rocks[1] = num_rocks[1];
        state.winner = winner;
    }

    void do_move(int x, int y, const std::string &t)
    {
        Move m = move_from_py(x, y, t);
        if (!state.move_legal(m, state.cp))
            throw std::invalid_argument("illegal move");
        state.do_move(m);
    }

    void undo_move()
    {
        if (state.hist.empty())
            throw std::runtime_error("nothing to undo");
        state.undo_move();
    }

    bool is_legal(int x, int y, const std::string &t) { return state.move_legal(move_from_py(x, y, t), state.cp); }

    py::list legal_moves(bool dedupe)
    {
        std::vector<Move> mv;
        state.gen_moves(mv, dedupe);
        py::list out;
        for (Move m : mv)
            out.append(move_to_py(m));
        return out;
    }

    // stop(): make a search running in another thread return soon; it and every later search
    // stop at once until clear_stop(). A plain search() clears it first; a background search
    // passes keep_stop=True so that a stop sent before it begins is not lost.
    void stop() { par.request_abort(); }
    void clear_stop() { par.clear_abort(); }

    py::dict search(int time_ms, int max_depth, std::int64_t max_nodes, bool keep_stop)
    {
        if (!keep_stop)
            par.clear_abort();
        SearchLimits lim;
        lim.time_ms = time_ms;
        lim.max_depth = max_depth > 0 ? max_depth : 64;
        lim.max_nodes = max_nodes;
        SearchResult r;
        {
            py::gil_scoped_release release;
            r = par.search(state, lim);
        }
        py::dict d;
        d["move"] = move_to_py(r.best);
        d["score"] = r.score;
        d["depth"] = r.depth;
        d["nodes"] = r.nodes;
        py::list pv;
        for (Move m : r.pv)
            pv.append(move_to_py(m));
        d["pv"] = pv;
        py::list rs_;
        for (auto &pr : r.root_scores)
            rs_.append(py::make_tuple(move_to_py(pr.first), pr.second));
        d["root_scores"] = rs_;
        return d;
    }

    int eval() { return searcher.static_eval(state); }
    int qeval() { return searcher.qsearch_value(state); }

    std::vector<int> features()
    {
        std::vector<int> f(N_FEATS);
        compute_features(state, f.data());
        return f;
    }

    py::list closures()
    {
        std::vector<Closure> cl;
        compute_closures(state, cl);
        py::list out;
        for (auto &c : cl)
            out.append(py::make_tuple(x_of(c.v), y_of(c.v), TYPE_NAMES[c.d], c.area2, c.can[0], c.can[1]));
        return out;
    }

    void set_weights(const std::vector<int> &w)
    {
        if ((int)w.size() != N_FEATS)
            throw std::invalid_argument("expected " + std::to_string(N_FEATS) + " weights");
        Weights nw;
        for (int i = 0; i < N_FEATS; ++i)
            nw.w[i] = w[(size_t)i];
        par.set_weights(nw);
    }

    std::vector<int> get_weights() { return std::vector<int>(searcher.weights.w, searcher.weights.w + N_FEATS); }

    // A linear weights file or a network file (see rs_eval.hpp load_weights_file).
    bool load_weights(const std::string &path)
    {
        Weights w;
        if (!load_weights_file(path, w))
            return false;
        par.set_weights(w);
        return true;
    }

    bool uses_mlp() const { return (bool)searcher.weights.mlp; }

    // The loaded network's output (eval units) for a feature vector; for checking training.
    int mlp_eval(const std::vector<int> &f) const
    {
        if (!searcher.weights.mlp || (int)f.size() != N_FEATS)
            throw std::invalid_argument("needs a loaded network and N_FEATS features");
        return searcher.weights.mlp->eval(f.data());
    }

    void clear() { par.clear(); }
    void set_root_margin(int m) { searcher.root_margin = m; }
    void seed(std::uint64_t sd) { searcher.rng.seed(sd); }
};

} // namespace

// Frozen copies of older engine versions (engine_src/versions/v<n>) are built under another
// module name, rs_engine_v<n>, so that they can still play rated games after the engine changes.
#ifndef RS_MODULE_NAME
#define RS_MODULE_NAME rs_engine
#endif

PYBIND11_MODULE(RS_MODULE_NAME, m)
{
    m.attr("N_FEATS") = N_FEATS;
    py::class_<Engine>(m, "Engine", py::module_local())
        .def(py::init<int, int>(), py::arg("tt_bits") = 21, py::arg("threads") = 1)
        .def("set_threads", &Engine::set_threads)
        .def("reset", &Engine::reset)
        .def("set_position", &Engine::set_position, py::arg("sticks"), py::arg("rocks"), py::arg("cp"), py::arg("turn"),
             py::arg("scores"), py::arg("num_rocks"), py::arg("winner") = -1)
        .def("do_move", &Engine::do_move)
        .def("undo_move", &Engine::undo_move)
        .def("is_legal", &Engine::is_legal)
        .def("legal_moves", &Engine::legal_moves, py::arg("dedupe") = false)
        .def("stop", &Engine::stop)
        .def("clear_stop", &Engine::clear_stop)
        .def("search", &Engine::search, py::arg("time_ms") = 1000, py::arg("max_depth") = 0, py::arg("max_nodes") = 0, py::arg("keep_stop") = false)
        .def("eval", &Engine::eval)
        .def("qeval", &Engine::qeval)
        .def("features", &Engine::features)
        .def("closures", &Engine::closures)
        .def("set_weights", &Engine::set_weights)
        .def("get_weights", &Engine::get_weights)
        .def("load_weights", &Engine::load_weights)
        .def("uses_mlp", &Engine::uses_mlp)
        .def("mlp_eval", &Engine::mlp_eval)
        .def("clear", &Engine::clear)
        .def("set_root_margin", &Engine::set_root_margin)
        .def("seed", &Engine::seed)
        .def_property_readonly("winner", [](const Engine &e) { return e.state.winner; })
        .def_property_readonly("current_player", [](const Engine &e) { return e.state.cp; })
        .def_property_readonly("turn", [](const Engine &e) { return e.state.turn; })
        .def_property_readonly("scores", [](const Engine &e) { return std::array<int, 2>{e.state.scores[0], e.state.scores[1]}; })
        .def_property_readonly("num_rocks", [](const Engine &e) { return std::array<int, 2>{e.state.num_rocks[0], e.state.num_rocks[1]}; });
}
