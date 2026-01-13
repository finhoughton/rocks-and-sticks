#pragma once

#include "gamestate.hpp"

namespace players_ext_internal
{
    struct PyGNNModules
    {
        py::object torch_module;
        py::object pyg_data_module;
        py::object pyg_data_Data;
        py::object pyg_data_Batch;
        py::object gnn_module;
        py::object types_module;
    };

    void ensure_py_gnn_modules(PyGNNModules &mods);

    py::object encode_state_common(
        GameState &g,
        PyGNNModules &mods,
        std::unordered_map<std::uint64_t, py::object> &enc_cache,
        size_t enc_cache_max,
        double *total_encode_time);

    py::list eval_probs_common(
        PyGNNModules &mods,
        const py::object &model_override,
        const std::string &model_device,
        const py::list &encs,
        double *total_model_time,
        size_t *model_calls,
        size_t *model_batch_items);
} // namespace players_ext_internal
