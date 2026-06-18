#include "envs/greedy_model.h"

namespace py = pybind11;

GreedyModel::GreedyModel(clpState* s0, int w, int max_blocks, int max_actions, int max_pblocks) 
    : ENV(s0, 999999.9, std::chrono::steady_clock::now()), 
      max_blocks(max_blocks), max_actions(max_actions), max_pblocks(max_pblocks) {
    
    current_node = dynamic_cast<clp::clpState*>(s0->clone());
    volume = 0.0;
    this->w = w;
}

GreedyModel::GreedyModel(std::string filename, int instance_number, int w, 
    int max_blocks, int max_actions, int max_pblocks, double min_fr) 
: GreedyModel(new_state(filename, instance_number, min_fr, 10000, clpState::BR), 
w, max_blocks, max_actions, max_pblocks) {}

py::tuple GreedyModel::get_enc_data() {
    // La interfaz del encoder es directa
    return TensorEncoder::get_enc_data(current_node, max_blocks, id_to_idx);
}

py::tuple GreedyModel::get_dec_data() {
    // Delegamos la creación de los 3 tensores al encoder
    return TensorEncoder::get_dec_data(current_node, vcs, id_to_idx, max_actions, max_pblocks);
}

void GreedyModel::transition(int selected_index) {
    std::list<Action*> actions;
    current_node->get_actions(actions);

    // Como bloqueamos la creación de BlockData, ahora usamos la lista de bloques válidos directamente.
    // selected_index es la posición en el tensor, por lo que iteramos para encontrar el ID real.
    int block_id = -1;
    for (const auto& pair : id_to_idx) {
        if (pair.second == selected_index) {
            block_id = pair.first;
            break;
        }
    }

    for (auto a : actions) {
        clp::clpAction* ca = dynamic_cast<clp::clpAction*>(a);
        
        if (ca && ca->block.id == block_id) {
            current_node->transition(*a);
            volume = current_node->get_value();
            best_state = current_node;

            std::list<Action*> child_actions;
            current_node->get_actions(child_actions);
            
            if (child_actions.size() == 0) {
                final_time = get_elapsed_time();
                completed = true;
            }
            
            for (auto action_ptr : actions) delete action_ptr;
            for (auto action_ptr : child_actions) delete action_ptr;
            break;
        }
    }
}

void register_greedy_model(py::module &m) {
    py::class_<GreedyModel, ENV>(m, "GreedyModel", py::module_local())
        // 1. Constructor actualizado con los límites de los tensores
        .def(py::init<std::string, int, int, int, int, int, double>(), 
                py::arg("filename"), 
                py::arg("instance_number"),
                py::arg("w"),
                py::arg("max_blocks"),
                py::arg("max_actions"),
                py::arg("max_pblocks"),
                py::arg("min_fr"))
        
        // Atributos públicos
        .def_readwrite("volume", &GreedyModel::volume)
        .def_readwrite("final_time", &GreedyModel::final_time)
        .def_readwrite("w", &GreedyModel::w)

        // 2. --- NUEVOS MÉTODOS CERO-COPIAS ---
        .def("get_enc_data", &GreedyModel::get_enc_data)
        .def("get_dec_data", &GreedyModel::get_dec_data)

        // Métodos de control
        .def("transition", &GreedyModel::transition, py::arg("selected_index"))
        .def("is_finished", &GreedyModel::is_finished)

        .def("get_path", &ENV::get_path);
}