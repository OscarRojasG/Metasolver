#include "envs/bsm_gm.h"

namespace py = pybind11;

BSM_GM::BSM_GM(clpState* s0, int w, int max_blocks, int max_actions, int max_pblocks, double timelimit, std::chrono::steady_clock::time_point start_time) : 
    ENV(s0, 999999.9, std::chrono::steady_clock::now()), 
    w(w), max_blocks(max_blocks), max_actions(max_actions), max_pblocks(max_pblocks) {

    clp::clpState* root_copy = dynamic_cast<clp::clpState*>(s0->clone());
    current_states.push_back(root_copy);

    current_depth_expand = 1;
    current_depth_greedy = 1;
}

BSM_GM::BSM_GM(std::string filename, int instance_number, int w, int max_blocks, int max_actions, int max_pblocks, double min_fr) 
    : BSM_GM(new_state(filename, instance_number, min_fr, 10000, clpState::BR), w, max_blocks, max_actions, max_pblocks) {}

BSM_GM::~BSM_GM() {}

py::tuple BSM_GM::get_enc_data() {
    return TensorEncoder::get_enc_data(current_states[0], id_to_idx);
}

py::tuple BSM_GM::get_dec_data_batch_expand() {
    return TensorEncoder::get_dec_data_batch(current_states, vcs, id_to_idx, max_actions, current_depth_expand);
}

py::tuple BSM_GM::get_dec_data_batch_greedy() {
    // Extraemos solo los estados "hijos" para la evaluación greedy
    std::vector<clpState*> nodes;
    nodes.reserve(succ_states.size());
    for (auto& item : succ_states) {
        nodes.push_back(item.second);
    }
    return TensorEncoder::get_dec_data_batch(nodes, vcs, id_to_idx, max_actions, current_depth_greedy);
}

void BSM_GM::expand(std::vector<std::vector<int>> selected_indexes_lists) { 
    int i = 0;

    // Crear mapa inverso para traducir índices del tensor a IDs del simulador
    std::map<int, int> idx_to_id;
    for (const auto& pair : id_to_idx) {
        idx_to_id[pair.second] = pair.first;
    }

    for (auto s : current_states) {
        for (auto block_idx : selected_indexes_lists[i]) {
            std::list<Action*> actions;
            s->get_actions(actions);

            int block_id = idx_to_id.at(block_idx);

            for (auto a : actions) {
                clp::clpAction* ca = dynamic_cast<clp::clpAction*>(a);
                
                if (ca && ca->block.id == block_id) {
                    clpState* state_copy = dynamic_cast<clpState*>(s->clone());
                    state_copy->transition(*a);

                    list<Action*> child_actions;
                    state_copy->get_actions(child_actions);
                    
                    if (child_actions.size() > 0) {
                        succ_states.push_back({s, state_copy});
                    } else {
                        delete state_copy;
                    }
                    
                    for (auto action_ptr : child_actions) delete action_ptr;
                    break;
                }
            }

            for (auto action_ptr : actions) delete action_ptr;
        }
        i++;
    }

    if (succ_states.size() == 0) {
        for (State *s : current_states) delete s;
        current_states.clear();
    } else {
        current_depth_greedy = current_depth_expand + 1;
    }
}

void BSM_GM::greedy_step(std::vector<int> selected_indexes) {
    std::map<int, int> idx_to_id;
    for (const auto& pair : id_to_idx) {
        idx_to_id[pair.second] = pair.first;
    }

    // 1. Aplicar transiciones
    for (size_t n = 0; n < succ_states.size(); ++n) {
        int block_idx = selected_indexes[n];
        if (block_idx == -1) continue;
        
        int block_id = idx_to_id.at(block_idx);

        std::list<Action*> actions;
        succ_states[n].second->get_actions(actions);

        for (auto a : actions) {
            clp::clpAction* ca = dynamic_cast<clp::clpAction*>(a);
            if (ca && ca->block.id == block_id) {
                succ_states[n].second->transition(*a);
                break;
            }
        }
        for (auto a : actions) delete a;
    }

    // 2. Filtrar hojas y actualizar resultados
    std::vector<std::pair<clp::clpState *, clp::clpState *>> remaining_items;

    for (auto& item : succ_states) {
        std::list<Action*> next_actions;
        item.second->get_actions(next_actions);

        if (next_actions.empty()) {
            double volume = item.second->get_value();
            
            if (volume > best_volume && get_elapsed_time() <= timelimit) {
                best_state = dynamic_cast<clpState*> (item.second->clone());
                best_volume = volume;
            }

            if (evals.find(-volume) == evals.end()) {
                evals[-volume] = std::make_pair(item.first, item.second);
            } else {
                delete item.second;
            }
        } else {
            for (auto a : next_actions) delete a;
            remaining_items.push_back(item); 
        }
    }
    succ_states = std::move(remaining_items);

    // 3. Si todos son hojas, generar siguiente nivel
    if (succ_states.empty()) {
        std::list<State*> next_states = EnvUtils::get_next_states(evals, w);
        for (State *s : current_states) delete s;
        current_states.clear();

        for (State *s : next_states)
        {
            clpState* s_copy = dynamic_cast<clpState *>(s);
            std::list<Action *> actions;
            s_copy->get_actions(actions);

            if (actions.size() > 0 && get_elapsed_time() <= timelimit) {
                current_states.push_back(dynamic_cast<clpState *>(s));
            } else {
                delete s;
            }

            for (auto a : actions) delete a;
        }
        current_depth_expand++;
        
    } else {
        current_depth_greedy++;
    }
}

bool BSM_GM::is_finished() {
    bool finished = current_states.empty();
    if (finished) final_time = get_elapsed_time();
    return finished;
}

bool BSM_GM::is_greedy_finished() {
    return succ_states.empty();
}

void register_bsm_gm(py::module &m) {
    py::class_<BSM_GM, ENV>(m, "BSM_GM", py::module_local())
        .def(py::init<std::string, int, int, int, int, int, double>(), 
             py::arg("filename"), py::arg("instance_number"), py::arg("w"),
             py::arg("max_blocks"), py::arg("max_actions"), py::arg("max_pblocks"), py::arg("min_fr"))
        .def_readwrite("best_volume", &BSM_GM::best_volume)
        .def_readwrite("final_time", &BSM_GM::final_time)
        .def_readwrite("w", &BSM_GM::w)

        .def("get_block_data", &ENV::get_block_data)
        
        // Nuevas funciones
        .def("get_enc_data", &BSM_GM::get_enc_data)
        .def("get_dec_data_batch_expand", &BSM_GM::get_dec_data_batch_expand)
        .def("get_dec_data_batch_greedy", &BSM_GM::get_dec_data_batch_greedy)

        .def("expand", &BSM_GM::expand, py::arg("selected_indexes_lists"))
        .def("greedy_step", &BSM_GM::greedy_step, py::arg("selected_indexes"))
        .def("is_greedy_finished", &BSM_GM::is_greedy_finished)
        .def("is_finished", &BSM_GM::is_finished);
}