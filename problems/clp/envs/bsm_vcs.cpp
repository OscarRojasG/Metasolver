#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "bsm_vcs.h"

namespace py = pybind11;

BSM_VCS::BSM_VCS(std::string filename, int instance_number, int w, double min_fr) : BSM_VCS(new_state(filename, instance_number, min_fr, 10000, clpState::BR), w) {}

BSM_VCS::BSM_VCS(clpState* s0, int w, double timelimit, std::chrono::steady_clock::time_point start_time) : BSM_ENV(s0, w, timelimit, start_time) {
    gr = new Greedy(vcs);
}

BSM_VCS::~BSM_VCS() {
    delete gr;
}

void BSM_VCS::transition(std::vector<std::vector<int>> selected_indexes_lists) {
    BSM_ENV::transition(selected_indexes_lists);

    std::map<double, std::pair<State *, State *>> state_actions;
    for (auto& item : batch_items) {
        if (get_elapsed_time() > timelimit) return;

        std::list<Action*> next_actions;
        double volume = gr->run(*item.current);
            
        // Actualizar el mejor volumen global
        if (volume > best_volume) {
            best_volume = volume;
        }

        if (state_actions.find(-volume) == state_actions.end()) {
            state_actions[-volume] = std::make_pair(item.original_node, item.current);
        }
    }

    std::list<State*> next_states = get_next_states(state_actions);
    for (State *s : next_states)
    {
        clpState* s_copy = dynamic_cast<clpState *>(s);
        std::list<Action *> actions;
        s_copy->get_actions(actions);

        if (actions.size() > 0) {
            current_nodes.push_back(dynamic_cast<clpState *>(s));
        }
    }
}

void register_bsm_vcs(py::module &m) {
    py::class_<BSM_VCS>(m, "BSM_VCS")
        .def(py::init<std::string, int, int, double>(), 
                py::arg("filename"), 
                py::arg("instance_number"),
                py::arg("w"),
                py::arg("min_fr"))
        .def_readwrite("best_volume", &BSM_VCS::best_volume)
        .def_readwrite("w", &BSM_VCS::w)

        .def("get_block_features", &BSM_VCS::get_block_features)
        .def("get_batch_dict", &BSM_VCS::get_batch_dict)
        .def("transition", &BSM_VCS::transition, py::arg("selected_ids_lists"))
        .def("is_finished", &BSM_VCS::is_finished);
}