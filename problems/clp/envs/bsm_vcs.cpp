#include "bsm_vcs.h"

namespace py = pybind11;

BSM_VCS::BSM_VCS(std::string filename, int instance_number, int w, 
                 int max_blocks, int max_actions, int max_pblocks, double min_fr) 
    : BSM_VCS(new_state(filename, instance_number, min_fr, 10000, clpState::BR), 
              w, max_blocks, max_actions, max_pblocks) {}

BSM_VCS::BSM_VCS(clpState* s0, int w, int max_blocks, int max_actions, int max_pblocks, double timelimit) 
    : ENV(s0, timelimit, std::chrono::steady_clock::now()), 
      max_blocks(max_blocks), max_actions(max_actions), max_pblocks(max_pblocks), w(w) {
    
    this->gr = new Greedy(vcs);
    
    clp::clpState* root_copy = dynamic_cast<clp::clpState*>(s0->clone());
    current_states.push_back(root_copy);

    this->current_depth = 1;
}

BSM_VCS::~BSM_VCS() { delete gr; }

py::tuple BSM_VCS::get_enc_data() {
    return TensorEncoder::get_enc_data(current_states[0], id_to_idx);
}

py::tuple BSM_VCS::get_dec_data_batch() {
    return TensorEncoder::get_dec_data_batch(current_states, vcs, id_to_idx, max_actions, current_depth);
}

void BSM_VCS::transition(const std::vector<std::vector<int>>& selected_indexes_lists) { 
    int i = 0;
    
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

std::map<double, std::pair<State *, State *>> BSM_VCS::eval_succ_states() {
    std::map<double, std::pair<State *, State *>> evals;
    for (auto& item : succ_states) {
        std::list<Action*> next_actions;
        double volume = gr->run(*item.second);
            
        // Actualizar el mejor volumen global
        if (volume > best_volume && get_elapsed_time() <= timelimit) {
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

bool BSM_VCS::is_finished() {
    bool finished = current_states.empty();
    if (finished) final_time = get_elapsed_time();
    return finished;
}

void register_bsm_vcs(py::module &m) {
    py::class_<BSM_VCS, ENV>(m, "BSM_VCS", py::module_local())
        .def(py::init<std::string, int, int, int, int, int, double>(), 
             py::arg("filename"), py::arg("instance_number"), py::arg("w"), 
             py::arg("max_blocks"), py::arg("max_actions"), py::arg("max_pblocks"), py::arg("min_fr"))
        .def_readwrite("best_volume", &BSM_VCS::best_volume)
        .def_readwrite("final_time", &BSM_VCS::final_time)
        .def_readwrite("w", &BSM_VCS::w)

        .def("get_enc_data", &BSM_VCS::get_enc_data)
        .def("get_dec_data_batch", &BSM_VCS::get_dec_data_batch)
        .def("transition", &BSM_VCS::transition, py::arg("selected_indexes_lists"))
        .def("is_finished", &BSM_VCS::is_finished);
}