#include "envs/greedy_model.h"

namespace py = pybind11;

GreedyModel::GreedyModel(clpState* s0, int w) : ENV(s0, 999999.9, std::chrono::steady_clock::now()) {
    current_node = dynamic_cast<clp::clpState*>(s0->clone());

    volume = 0.0;
    this->w = w;

    update();
}

GreedyModel::GreedyModel(std::string filename, int instance_number, int w, double min_fr) 
    : GreedyModel(new_state(filename, instance_number, min_fr, 10000, clpState::BR), w) {}

void GreedyModel::transition(int selected_index) {
    std::list<Action*> actions;
    current_node->get_actions(actions);

    int block_id = block_data->get_block_index_to_id().at(selected_index);

    for (auto a : actions) {
        clp::clpAction* ca = dynamic_cast<clp::clpAction*>(a);
        
        if (ca && ca->block.id == block_id) {
            current_node->transition(*a);
            volume = current_node->get_value();
            best_state = current_node;

            list<Action*> child_actions;
            current_node->get_actions(child_actions);
            
            if (child_actions.size() == 0) {
                final_time = get_elapsed_time();
                completed = true;
            } else {
                update();
            }
            
            // Liberar la lista de acciones para evitar memory leaks
            for (auto action_ptr : actions) delete action_ptr;
            for (auto action_ptr : child_actions) delete action_ptr;
            break;
        }
    }
}

void GreedyModel::update() {
    StateData state_data(*block_data, current_node, vcs, w*w);
    action_data = state_data.get_action_features();
    placed_data = state_data.get_placed_features();
    space_data = state_data.get_space_features();
}

void register_greedy_model(py::module &m) {
    py::class_<GreedyModel, ENV>(m, "GreedyModel")
        .def(py::init<std::string, int, int, double>(), 
                py::arg("filename"), 
                py::arg("instance_number"),
                py::arg("w"),
                py::arg("min_fr"))
        .def_readwrite("volume", &GreedyModel::volume)
        .def_readwrite("final_time", &GreedyModel::final_time)
        .def_readwrite("w", &GreedyModel::w)

        .def("get_block_data", &ENV::get_block_data)
        .def("get_box_data", &ENV::get_box_data)
        .def("get_boxes_per_block", &ENV::get_boxes_per_block)
        .def("get_action_data", &GreedyModel::get_action_data)
        .def("get_pblock_data", &GreedyModel::get_pblock_data)
        .def("get_space_data", &GreedyModel::get_space_data)

        .def("transition", &GreedyModel::transition, py::arg("selected_index"))
        .def("is_finished", &GreedyModel::is_finished)

        .def("get_path", &ENV::get_path);
}