#include "bsm_vcs.h"

namespace py = pybind11;

BSM_VCS::BSM_VCS(clpState* s0, int w, double timelimit, std::chrono::steady_clock::time_point start_time) : 
    ENV(s0, 999999.9, std::chrono::steady_clock::now()) {
    this->gr = new Greedy(vcs);
    this->w = w;

    clp::clpState* root_copy = dynamic_cast<clp::clpState*>(s0->clone());
    current_states.push_back(root_copy);
    update();
}

BSM_VCS::BSM_VCS(std::string filename, int instance_number, int w, double min_fr) 
    : BSM_VCS(new_state(filename, instance_number, min_fr, 10000, clpState::BR), w) {}

BSM_VCS::~BSM_VCS() {
    delete gr;
}

void BSM_VCS::transition(std::vector<std::vector<int>> selected_indexes_lists) { 
    int i = 0;

    for (auto s : current_states) {
        for (auto block_idx : selected_indexes_lists[i]) {
            std::list<Action*> actions;
            s->get_actions(actions);

            int block_id = block_data->get_block_index_to_id().at(block_idx);

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

    update();
    succ_states.clear();
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

void BSM_VCS::update() {
    BatchData batch_data(*block_data, current_states, vcs, w*w);
    action_data = batch_data.get_batch_action_features();
    placed_data = batch_data.get_batch_placed_features();
    space_data = batch_data.get_batch_space_features();
}

bool BSM_VCS::is_finished() {
    bool finished = current_states.empty();
    if (finished) final_time = get_elapsed_time();
    return finished;
}

void register_bsm_vcs(py::module &m) {
    py::class_<BSM_VCS, ENV>(m, "BSM_VCS")
        .def(py::init<std::string, int, int, double>(), 
             py::arg("filename"), 
             py::arg("instance_number"),
             py::arg("w"),
             py::arg("min_fr"))
        .def_readwrite("best_volume", &BSM_VCS::best_volume)
        .def_readwrite("final_time", &BSM_VCS::final_time)
        .def_readwrite("w", &BSM_VCS::w)

        .def("get_block_data", &ENV::get_block_data)
        .def("get_action_data_batch", &BSM_VCS::get_action_data_batch)
        .def("get_pblock_data_batch", &BSM_VCS::get_pblock_data_batch)
        .def("get_space_data_batch", &BSM_VCS::get_space_data_batch)

        .def("transition", &BSM_VCS::transition, py::arg("selected_indexes_lists"))
        .def("is_finished", &BSM_VCS::is_finished);
}