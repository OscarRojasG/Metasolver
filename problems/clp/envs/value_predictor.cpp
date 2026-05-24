#include "envs/value_predictor.h"

namespace py = pybind11;

ValuePredictor::ValuePredictor(clpState* s0, int w) : ENV(s0, 999999.9, std::chrono::steady_clock::now()) {
    this->w = w;
    
    clp::clpState* root_copy = dynamic_cast<clp::clpState*>(s0->clone());
    current_states.push_back(root_copy);
    update();
}

ValuePredictor::ValuePredictor(std::string filename, int instance_number, int w, double min_fr) 
    : ValuePredictor(new_state(filename, instance_number, min_fr, 10000, clpState::BR), w) {}

bool ValuePredictor::is_finished() {
    return completed;
}

void ValuePredictor::expand(const std::vector<std::vector<int>>& selected_blocks) {
    succ_states.clear();

    for (size_t i = 0; i < current_states.size(); ++i) {
        clp::clpState* current_state = current_states[i];
        const std::vector<int>& chosen_block_ids = selected_blocks[i];

        for (int block_id : chosen_block_ids) {
            int target_block_id = block_index_to_id[block_id];

            // 1. Creamos el clon del hijo primero
            clp::clpState* child_state = dynamic_cast<clp::clpState*>(current_state->clone());
            
            // 2. Extraemos las acciones directamente desde este hijo
            std::list<Action*> child_actions;
            child_state->get_actions(child_actions);

            bool action_applied = false;

            for (auto* action : child_actions) {
                clp::clpAction* ca = dynamic_cast<clp::clpAction*>(action);
                
                if (ca && ca->block.id == target_block_id) {
                    // Transición segura
                    child_state->transition(*action);

                    // --- BINGO: NUEVA VALIDACIÓN DE ACCIONES FUTURAS ---
                    std::list<Action*> future_actions;
                    child_state->get_actions(future_actions);
                    bool has_future_actions = !future_actions.empty();
                    
                    // Limpiamos los punteros de la consulta de prueba
                    for (auto* fa : future_actions) delete fa; 

                    // Actualizamos el volumen óptimo siempre (incluso si es terminal)
                    double current_vol = child_state->get_value();
                    if (current_vol > best_volume) best_volume = current_vol;

                    if (has_future_actions) {
                        succ_states.push_back(child_state);
                        action_applied = true; // Se mantiene vivo en succ_states
                    }
                    break;
                }
            }

            // Limpieza de la lista de acciones del clon
            for (auto* action : child_actions) delete action;

            // Si Python mandó un ID inválido O si el estado se quedó sin acciones futuras, lo destruimos
            if (!action_applied) {
                delete child_state;
            }
        }
    }

    // Los estados padres se liberan solo después de generar todos los hijos
    for (auto* state : current_states) delete state;
    current_states.clear();

    // Si la poda de validación vació por completo succ_states, el juego termina naturalmente.
    if (succ_states.empty()) {
        completed = true;
        final_time = get_elapsed_time();
    } else {
        update_succ();
    }
}

void ValuePredictor::prune(const std::vector<int>& selected_states) {
    for (int idx : selected_states) {
        if (idx >= 0 && idx < static_cast<int>(succ_states.size())) {
            // Ya no llamamos a get_actions() aquí. 
            // Simplemente pasamos el estado intacto a la siguiente generación.
            current_states.push_back(succ_states[idx]);
            
            // Lo marcamos como nullptr para que no sea destruido en la limpieza
            succ_states[idx] = nullptr;
        }
    }

    // Limpiamos los estados que NO fueron seleccionados por el Value Model
    for (auto* state : succ_states) {
        if (state != nullptr) delete state;
    }
    succ_states.clear();

    if (current_states.empty()) {
        completed = true;
        final_time = get_elapsed_time();
    } else {
        update();
    }
}

void ValuePredictor::update() {
    // Limpiamos los buffers de datos vectoriales previos
    action_data.clear();
    placed_data.clear();
    space_data.clear();

    // Re-calculamos las representaciones vectoriales para cada uno de los estados vivos en el haz
    for (auto* state : current_states) {
        action_data.push_back(EnvUtilsPython::get_actions_data(state, vcs, block_id_to_index, w * w));
        placed_data.push_back(EnvUtilsPython::get_placed_data(state, block_id_to_index));
        space_data.push_back(EnvUtilsPython::get_space_data(state));
    }
}

void ValuePredictor::update_succ() {
    succ_action_data.clear();
    succ_placed_data.clear();
    succ_space_data.clear();

    for (auto* state : succ_states) {
        succ_action_data.push_back(EnvUtilsPython::get_actions_data(state, vcs, block_id_to_index, w * w));
        succ_placed_data.push_back(EnvUtilsPython::get_placed_data(state, block_id_to_index));
        succ_space_data.push_back(EnvUtilsPython::get_space_data(state));
    }
}

// Registro del nuevo entorno para Pybind11
void register_value_predictor(py::module &m) {
    py::class_<ValuePredictor>(m, "ValuePredictor")
        .def(py::init<std::string, int, int, double>(), 
             py::arg("filename"), 
             py::arg("instance_number"),
             py::arg("w"),
             py::arg("min_fr"))
        .def_readwrite("best_volume", &ValuePredictor::best_volume)
        .def_readwrite("final_time", &ValuePredictor::final_time)
        .def_readwrite("w", &ValuePredictor::w)

        .def("get_block_data", &ValuePredictor::get_block_data)
        .def("get_action_data_batch", &ValuePredictor::get_action_data_batch)
        .def("get_pblock_data_batch", &ValuePredictor::get_pblock_data_batch)
        .def("get_space_data_batch", &ValuePredictor::get_space_data_batch)

        .def("get_succ_action_data_batch", &ValuePredictor::get_succ_action_data_batch)
        .def("get_succ_pblock_data_batch", &ValuePredictor::get_succ_pblock_data_batch)
        .def("get_succ_space_data_batch",  &ValuePredictor::get_succ_space_data_batch)

        .def("expand", &ValuePredictor::expand, py::arg("selected_blocks"))
        .def("prune", &ValuePredictor::prune, py::arg("selected_states"))
        .def("is_finished", &ValuePredictor::is_finished);
}