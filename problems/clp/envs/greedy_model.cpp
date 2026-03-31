#include "envs/greedy_model.h"

namespace py = pybind11;

GreedyModel::GreedyModel(clpState* s0, int w) : ENV(s0, 999999.9, std::chrono::steady_clock::now()) {
    current_node = dynamic_cast<clp::clpState*>(s0->clone());

    volume = 0.0;
    this->w = w;

    // VCS y parámetros
    double alpha = 4.0, beta = 1.0, gamma = 0.2, delta = 1.0, p = 0.04;
    double r = 0.0;
    vcs = new VCS_Function(s0->nb_left_boxes, *s0->cont, alpha, beta, gamma, p, delta, 0.0, r);

    EnvUtilsPython::get_blocks_data(s0, block_features, block_id_to_index, block_index_to_id);
}

GreedyModel::GreedyModel(std::string filename, int instance_number, int w, double min_fr) 
    : GreedyModel(new_state(filename, instance_number, min_fr, 10000, clpState::BR), w) {}

py::array_t<float> GreedyModel::get_block_features() {
    return block_features;
}

py::dict GreedyModel::get_dict() {
    update(); 
    py::dict d;
    d["act_blocks"] = action_blocks;
    d["act_feats"] = action_features;
    d["pl_blocks"] = placed_blocks;
    d["pl_feats"] = placed_features;
    d["sp_feats"] = space_features;
    d["biases"] = biases;
    return d;
}

void GreedyModel::transition(int selected_index) {
    int i = 0;

    std::list<Action*> actions;
    current_node->get_actions(actions);

    int block_id = block_index_to_id[selected_index];

    for (auto a : actions) {
        clp::clpAction* ca = dynamic_cast<clp::clpAction*>(a);
        
        if (ca && ca->block.id == block_id) {
            current_node->transition(*a);
            volume = current_node->get_value();

            list<Action*> child_actions;
            current_node->get_actions(child_actions);
            
            if (child_actions.size() == 0) {
                final_time = get_elapsed_time();
                completed = true;
            }
            
            // Liberar la lista de acciones para evitar memory leaks
            for (auto action_ptr : actions) delete action_ptr;
            for (auto action_ptr : child_actions) delete action_ptr;
            break;
        }
        i++;
    }
}

bool GreedyModel::is_finished() {
    return completed;
}

void GreedyModel::update() {
    EnvUtilsPython::get_actions_data(current_node, vcs, w, block_id_to_index, action_blocks, action_features, biases);
    EnvUtilsPython::get_placed_data(current_node, block_id_to_index, placed_blocks, placed_features);
    EnvUtilsPython::get_space_features(current_node, space_features);
}

void register_greedy_model(py::module &m) {
    py::class_<GreedyModel>(m, "GreedyModel")
        .def(py::init<std::string, int, int, double>(), 
                py::arg("filename"), 
                py::arg("instance_number"),
                py::arg("w"),
                py::arg("min_fr"))
        .def_readwrite("volume", &GreedyModel::volume)
        .def_readwrite("final_time", &GreedyModel::final_time)
        .def_readwrite("w", &GreedyModel::w)

        .def("get_block_features", &GreedyModel::get_block_features)
        .def("get_dict", &GreedyModel::get_dict)
        .def("transition", &GreedyModel::transition, py::arg("selected_index"))
        .def("is_finished", &GreedyModel::is_finished);
}