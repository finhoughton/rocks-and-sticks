#include "gnn_helpers.hpp"

namespace players_ext_internal
{
    void ensure_py_gnn_modules(PyGNNModules &mods)
    {
        if (mods.torch_module)
            return;
        py::gil_scoped_acquire gil;
        mods.torch_module = py::module::import("torch");
        mods.pyg_data_module = py::module::import("torch_geometric.data");
        mods.pyg_data_Data = mods.pyg_data_module.attr("Data");
        mods.pyg_data_Batch = mods.pyg_data_module.attr("Batch");
        mods.gnn_module = py::module::import("gnn.model");
        mods.types_module = py::module::import("types");
    }

    py::object encode_state_common(
        GameState &g,
        PyGNNModules &mods,
        std::unordered_map<std::uint64_t, py::object> &enc_cache,
        size_t enc_cache_max,
        double *total_encode_time)
    {
        ensure_py_gnn_modules(mods);

        std::uint64_t enc_key = g.state_key();
        auto it_cache = enc_cache.find(enc_key);
        if (it_cache != enc_cache.end())
            return it_cache->second;

        // Build a deterministic, de-duplicated node list from connected points + rocks.
        g.scratch_nodes.clear();
        g.scratch_nodes.reserve(g.connected_points.size() + g.rocks.size());
        for (auto *p : g.connected_points)
            g.scratch_nodes.push_back(p);
        for (auto *p : g.rocks)
            g.scratch_nodes.push_back(p);
        std::sort(g.scratch_nodes.begin(), g.scratch_nodes.end(), [](Node *a, Node *b)
                  {
				  if (a == b)
					  return false;
				  if (a->x != b->x)
					  return a->x < b->x;
				  return a->y < b->y; });
        g.scratch_nodes.erase(std::unique(g.scratch_nodes.begin(), g.scratch_nodes.end(), [](Node *a, Node *b)
                                          { return a == b || (a->x == b->x && a->y == b->y); }),
                              g.scratch_nodes.end());

        g.scratch_idx_map.clear();
        if (!g.scratch_nodes.empty())
            g.scratch_idx_map.reserve(g.scratch_nodes.size() * 2);
        for (size_t i = 0; i < g.scratch_nodes.size(); ++i)
            g.scratch_idx_map[g.scratch_nodes[i]] = (int)i;

        std::vector<std::vector<double>> node_feats;
        node_feats.reserve(g.scratch_nodes.size());
        std::vector<std::array<long long, 2>> coords;
        coords.reserve(g.scratch_nodes.size());
        for (auto *n : g.scratch_nodes)
        {
            int num_players = GameState::num_players;
            std::vector<double> owner_one_hot(num_players + 1, 0.0);
            int owner_idx = (n->rocked_by >= 0) ? (n->rocked_by + 1) : 0;
            if (owner_idx >= 0 && owner_idx < (int)owner_one_hot.size())
                owner_one_hot[owner_idx] = 1.0;
            int neighbour_count = 0;
            for (int d = 0; d < 8; ++d)
                if (n->neighbours[d])
                    neighbour_count++;
            double deg = (double)neighbour_count / 8.0;
            double is_leaf = (neighbour_count == 1) ? 1.0 : 0.0;
            double x = (double)n->x;
            double y = (double)n->y;
            double r2 = x * x + y * y;
            std::vector<double> feats = owner_one_hot;
            feats.push_back(deg);
            feats.push_back(is_leaf);
            feats.push_back(x);
            feats.push_back(y);
            feats.push_back(r2);
            node_feats.push_back(std::move(feats));
            coords.push_back({n->x, n->y});
        }

        std::vector<long long> srcs;
        std::vector<long long> dsts;
        std::vector<std::vector<double>> edge_attrs;
        for (size_t i = 0; i < g.scratch_nodes.size(); ++i)
        {
            Node *p = g.scratch_nodes[i];
            for (int d = 0; d < 8; ++d)
            {
                Node *q = p->neighbours[d];
                if (!q)
                    continue;
                auto it_idx = g.scratch_idx_map.find(q);
                if (it_idx == g.scratch_idx_map.end())
                    continue;
                int j = it_idx->second;
                double dx = double(q->x - p->x);
                double dy = double(q->y - p->y);
                double is_diag = (std::abs(dx) == 1.0 && std::abs(dy) == 1.0) ? 1.0 : 0.0;
                double orth = 1.0 - is_diag;
                srcs.push_back((long long)i);
                dsts.push_back((long long)j);
                edge_attrs.push_back({orth, is_diag});
            }
        }

        py::gil_scoped_acquire gil;
        auto enc_start = std::chrono::high_resolution_clock::now();

        py::object x_tensor = mods.torch_module.attr("tensor")(py::cast(node_feats));
        py::object edge_index;
        if (!srcs.empty())
        {
            std::vector<std::vector<long long>> ei = {srcs, dsts};
            edge_index = mods.torch_module.attr("tensor")(py::cast(ei));
        }
        else
        {
            edge_index = mods.torch_module.attr("empty")(py::make_tuple(2, 0));
        }
        py::object edge_attr = edge_attrs.empty() ? mods.torch_module.attr("empty")(py::make_tuple(0, 2)) : mods.torch_module.attr("tensor")(py::cast(edge_attrs));
        py::object batch = mods.torch_module.attr("zeros")(py::make_tuple((py::int_)g.scratch_nodes.size())).attr("to")(mods.torch_module.attr("long"));

        double turn = (double)g.turn_number;
        std::vector<double> cur_one_hot(GameState::num_players, 0.0);
        if (g.current_player >= 0 && g.current_player < GameState::num_players)
            cur_one_hot[g.current_player] = 1.0;
        std::vector<double> scores;
        for (auto s : g.players_scores)
            scores.push_back((double)s);
        std::vector<double> rocks_left;
        for (auto r : g.num_rocks)
            rocks_left.push_back((double)r);
        std::vector<double> rocks_placed(GameState::num_players, 0.0);
        for (auto *n : g.scratch_nodes)
            if (n->rocked_by != -1)
                rocks_placed[n->rocked_by] += 1.0;
        double max_r2 = 0.0;
        for (auto &pr : coords)
        {
            double rr2 = double(pr[0] * pr[0] + pr[1] * pr[1]);
            if (rr2 > max_r2)
                max_r2 = rr2;
        }
        std::vector<double> global_feats;
        global_feats.push_back(turn);
        for (double v : cur_one_hot)
            global_feats.push_back(v);
        for (double v : scores)
            global_feats.push_back(v);
        for (double v : rocks_left)
            global_feats.push_back(v);
        for (double v : rocks_placed)
            global_feats.push_back(v);
        global_feats.push_back(max_r2);
        py::object global_tensor = mods.torch_module.attr("tensor")(py::cast(global_feats)).attr("unsqueeze")(0);

        py::object data = mods.pyg_data_Data("x"_a = x_tensor, "edge_index"_a = edge_index, "edge_attr"_a = edge_attr, "batch"_a = batch, "global_feats"_a = global_tensor);
        data.attr("node_coords") = mods.torch_module.attr("tensor")(py::cast(coords));
        py::object enc_obj = mods.types_module.attr("SimpleNamespace")("data"_a = data);

        auto enc_end = std::chrono::high_resolution_clock::now();
        if (total_encode_time)
            *total_encode_time += std::chrono::duration<double>(enc_end - enc_start).count();

        if (enc_cache.size() > enc_cache_max)
            enc_cache.clear();
        enc_cache[enc_key] = enc_obj;
        return enc_obj;
    }

    py::list eval_probs_common(
        PyGNNModules &mods,
        const py::object &model_override,
        const std::string &model_device,
        const py::list &encs,
        double *total_model_time,
        size_t *model_calls,
        size_t *model_batch_items)
    {
        ensure_py_gnn_modules(mods);
        py::gil_scoped_acquire gil;
        auto model_start = std::chrono::high_resolution_clock::now();

        if (model_calls)
            *model_calls += 1;
        if (model_batch_items)
            *model_batch_items += (size_t)py::len(encs);

        py::list out;
        if (!model_override.is_none())
        {
            py::list datas;
            for (auto item : encs)
            {
                py::object enc_obj = py::cast<py::object>(item);
                datas.append(enc_obj.attr("data"));
            }
            py::object batch = mods.pyg_data_Batch.attr("from_data_list")(datas);
            if (!model_device.empty())
                batch = batch.attr("to")(py::cast(model_device));

            py::object no_grad = mods.torch_module.attr("no_grad")();
            no_grad.attr("__enter__")();
            py::object logits = model_override(batch);
            no_grad.attr("__exit__")(py::none(), py::none(), py::none());
            py::object probs = mods.torch_module.attr("sigmoid")(logits).attr("detach")().attr("cpu")();
            out = py::cast<py::list>(probs.attr("tolist")());
        }
        else
        {
            py::object py_probs = mods.gnn_module.attr("evaluate_encodings")(encs);
            out = py::cast<py::list>(py_probs);
        }

        auto model_end = std::chrono::high_resolution_clock::now();
        if (total_model_time)
            *total_model_time += std::chrono::duration<double>(model_end - model_start).count();
        return out;
    }
} // namespace players_ext_internal
