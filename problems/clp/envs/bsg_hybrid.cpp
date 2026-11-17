#include "bsg_hybrid.h"

namespace py = pybind11;

BSG_Hybrid::BSG_Hybrid(std::string filename, int instance_number, int w, 
                 int max_blocks, int max_actions, int max_pblocks, double min_fr) 
    : BSG_Hybrid(new_state(filename, instance_number, min_fr, 10000, clpState::BR), 
              w, max_blocks, max_actions, max_pblocks) {}

BSG_Hybrid::BSG_Hybrid(clpState* s0, int w, int max_blocks, int max_actions, int max_pblocks, double timelimit) 
    : ENV(s0, timelimit, metasolver::cpu_clock::now()), 
      max_blocks(max_blocks), max_actions(max_actions), max_pblocks(max_pblocks), w(w) {
    
    this->gr = new Greedy(vcs);
    
    clp::clpState* root_copy = dynamic_cast<clp::clpState*>(s0->clone());
    current_states.push_back(root_copy);

    this->current_depth = 1;
}

BSG_Hybrid::~BSG_Hybrid() { delete gr; }

py::tuple BSG_Hybrid::get_enc_data() {
    return TensorEncoder::get_enc_data(current_states[0], id_to_idx);
}

py::tuple BSG_Hybrid::get_dec_data_batch() {
    return TensorEncoder::get_dec_data_batch(current_states, vcs, id_to_idx, max_actions, current_depth);
}

void BSG_Hybrid::transition(const std::vector<std::vector<int>>& selected_indexes_lists) { 
    int i = 0;
    
    std::map<int, int> idx_to_id;
    for (const auto& pair : id_to_idx) {
        idx_to_id[pair.second] = pair.first;
    }

    for (auto s : current_states) {
        for (auto block_idx : selected_indexes_lists[i]) {
            if (get_elapsed_time() > timelimit) break;

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
                        // Estado terminal: se evalúa aquí, ya que no pasa por eval_succ_states
                        double volume = state_copy->get_value();
                        if (volume > best_volume) {
                            delete best_state;
                            best_state = state_copy;
                            best_volume = volume;
                        } else {
                            delete state_copy;
                        }
                    }
                    
                    for (auto action_ptr : child_actions) delete action_ptr;
                    break;
                }
            }

            for (auto action_ptr : actions) delete action_ptr;
        }
        i++;
    }

    std::map<double, std::pair<State *, State *>> evals = eval_succ_states();
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

    succ_states.clear();
    current_depth++;
}

std::map<double, std::pair<State *, State *>> BSG_Hybrid::eval_succ_states() {
    std::map<double, std::pair<State *, State *>> evals;
    for (auto& item : succ_states) {
        // Sin tiempo: el hijo se descarta sin evaluar
        if (get_elapsed_time() > timelimit) {
            delete item.second;
            continue;
        }

        double volume = gr->run(*item.second, timelimit, start_time);
            
        // Actualizar el mejor volumen global
        if (volume > best_volume) {
            delete best_state;
            best_state = dynamic_cast<clpState*> (item.second->clone());
            best_volume = volume;
        }

        if (evals.find(-volume) == evals.end()) {
            evals[-volume] = std::make_pair(item.first, item.second);
        } else {
            delete item.second;
        }
    }

    return evals;
}

bool BSG_Hybrid::is_finished() {
    // Al alcanzar el tiempo límite se descartan los nodos pendientes
    if (get_elapsed_time() > timelimit) {
        for (auto s : current_states) delete s;
        current_states.clear();
    }

    bool finished = current_states.empty();
    if (finished) final_time = get_elapsed_time();
    return finished;
}

void BSG_Hybrid::reset() {
    for (auto s : current_states) delete s;
    current_states.clear();

    current_states.push_back(dynamic_cast<clpState*>(s0->clone()));
    current_depth = 1;
}

void register_bsg_hybrid(py::module &m) {
    py::class_<BSG_Hybrid, ENV>(m, "BSG_Hybrid", py::module_local())
        .def(py::init<std::string, int, int, int, int, int, double>(), 
             py::arg("filename"), py::arg("instance_number"), py::arg("w"), 
             py::arg("max_blocks"), py::arg("max_actions"), py::arg("max_pblocks"), py::arg("min_fr"))
        .def_readwrite("best_volume", &BSG_Hybrid::best_volume)
        .def_readwrite("final_time", &BSG_Hybrid::final_time)
        .def_readwrite("w", &BSG_Hybrid::w)
        .def_readwrite("timelimit", &BSG_Hybrid::timelimit)
        .def_readwrite("max_actions", &BSG_Hybrid::max_actions)

        .def("get_enc_data", &BSG_Hybrid::get_enc_data)
        .def("get_dec_data_batch", &BSG_Hybrid::get_dec_data_batch)
        .def("transition", &BSG_Hybrid::transition, py::arg("selected_indexes_lists"))
        .def("is_finished", &BSG_Hybrid::is_finished)
        .def("reset", &BSG_Hybrid::reset)
        .def("get_elapsed_time", &BSG_Hybrid::get_elapsed_time);
}