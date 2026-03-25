#include "envs/bsm_gm.h"

namespace py = pybind11;

BSM_GM::BSM_GM(std::string filename, int instance_number, int w, double min_fr) : BSM_ENV(filename, instance_number, w) {}

BSM_GM::BSM_GM(clpState* s0, int w, double timelimit, std::chrono::steady_clock::time_point start_time) : BSM_ENV(s0, w, timelimit, start_time) {}

void BSM_GM::transition_bsm(std::vector<std::vector<int>> selected_indexes_lists) {
    BSM_ENV::transition(selected_indexes_lists);
    if (batch_items.size() == 0) {
        for (State *s : current_nodes) delete s;
        current_nodes.clear();
    }
}

void BSM_GM::transition_greedy(std::vector<int> selected_indexes) {
    // 1. Aplicar transiciones
    for (size_t n = 0; n < batch_items.size(); ++n) {
        int block_idx = selected_indexes[n];
        if (block_idx == -1) continue;
        
        int block_id = block_index_to_id[block_idx];

        std::list<Action*> actions;
        batch_items[n].current->get_actions(actions);

        for (auto a : actions) {
            clp::clpAction* ca = dynamic_cast<clp::clpAction*>(a);
            if (ca && ca->block.id == block_id) {
                batch_items[n].current->transition(*a);
                break;
            }
        }
        for (auto a : actions) delete a;
    }

    // 2. Filtrar hojas y actualizar resultados
    std::vector<BatchItem> remaining_items; // Para guardar los que no son hojas

    for (auto& item : batch_items) {
        std::list<Action*> next_actions;
        item.current->get_actions(next_actions);

        if (next_actions.empty()) {
            double volume = item.current->get_value();
            
            if (volume > best_volume && get_elapsed_time() <= timelimit) {
                best_volume = volume;
            }

            if (state_actions.find(-volume) == state_actions.end()) {
                state_actions[-volume] = std::make_pair(item.original_node, item.current);
            } else {
                delete item.current;
            }
        } else {
            for (auto a : next_actions) delete a;
            remaining_items.push_back(item); 
        }
    }
    // Al final, batch_items se actualiza con los que no fueron hojas
    batch_items = std::move(remaining_items);

    // 3. Si todos son hojas, generar siguiente nivel
    if (batch_items.empty()) {
        std::list<State*> next_states = get_next_states(state_actions);
        for (State *s : current_nodes) delete s;
        current_nodes.clear();

        for (State *s : next_states)
        {
            clpState* s_copy = dynamic_cast<clpState *>(s);
            std::list<Action *> actions;
            s_copy->get_actions(actions);

            if (actions.size() > 0 && get_elapsed_time() <= timelimit) {
                current_nodes.push_back(dynamic_cast<clpState *>(s));
            } else {
                delete s;
            }

            for (auto a : actions) delete a;
        }

        if (current_nodes.empty()) {
            final_time = get_elapsed_time();
        }
    }
}

py::dict BSM_GM::get_batch_dict_bsm() {
    return BSM_ENV::get_batch_dict();
}

py::dict BSM_GM::get_batch_dict_greedy() {
    update_batches_greedy(); 
    py::dict d;
    d["act_blocks"] = action_blocks_gr;
    d["act_feats"] = action_features_gr;
    d["pl_blocks"] = placed_blocks_gr;
    d["pl_feats"] = placed_features_gr;
    d["sp_feats"] = space_features_gr;
    return d;
}

bool BSM_GM::is_greedy_finished() {
    return batch_items.empty();
}

bool BSM_GM::is_bsm_finished() {
    return BSM_ENV::is_finished();
}

// Helper para extraer los punteros 'current' de batch_items
std::vector<clpState*> BSM_GM::get_greedy_nodes_vec() {
    std::vector<clpState*> nodes;
    nodes.reserve(batch_items.size());
    for (auto& item : batch_items) nodes.push_back(item.current);
    return nodes;
}

void BSM_GM::update_batches_greedy() {
    std::vector<clp::clpState*> nodes = get_greedy_nodes_vec();

    EnvUtilsPython::get_actions_data_batch(nodes, vcs, w, block_id_to_index, action_blocks_gr, action_features_gr);
    EnvUtilsPython::get_placed_data_batch(nodes, block_id_to_index, placed_blocks_gr, placed_features_gr);
    EnvUtilsPython::get_space_features_batch(nodes, space_features_gr);
}

void register_bsm_gm(py::module &m) {
    py::class_<BSM_GM>(m, "BSM_GM")
        .def(py::init<std::string, int, int, double>(), 
                py::arg("filename"), 
                py::arg("instance_number"), 
                py::arg("w"),
                py::arg("min_fr"))
        .def_readwrite("best_volume", &BSM_GM::best_volume)
        .def_readwrite("final_time", &BSM_GM::final_time)
        .def_readwrite("w", &BSM_GM::w)

        .def("get_block_features", &BSM_GM::get_block_features)

        .def("get_batch_dict_bsm", &BSM_GM::get_batch_dict_bsm)
        .def("get_batch_dict_greedy", &BSM_GM::get_batch_dict_greedy)

        .def("transition_bsm", &BSM_GM::transition_bsm, py::arg("selected_ids_lists"))
        .def("transition_greedy", &BSM_GM::transition_greedy, py::arg("selected_ids"))
        .def("is_bsm_finished", &BSM_GM::is_bsm_finished)
        .def("is_greedy_finished", &BSM_GM::is_greedy_finished);
}