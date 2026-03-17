
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "clpState.h"
#include "BSG.h"
#include "Greedy.h"
#include "VCS_Function.h"
#include "BlockMetrics.h"
#include "envs/env_utils_python.h"
#include <random> // Para los números aleatorios

namespace py = pybind11;

struct BatchItem {
	clpState* original_node;
	clpState* current;
};

class BSM_ENV {
public:
    int w;
    double best_volume;

    BSM_ENV(std::string filename, int instance_number, int w) {
        double min_fr = 0.98;
        best_volume = 0.0;
        this->w = w;

        clpState::Format f = clpState::BR;
        s0 = new_state(filename, instance_number, min_fr, 10000, f);
        current_nodes.push_back(dynamic_cast<clp::clpState*>(s0->clone()));

        // VCS y parámetros
        double alpha = 4.0, beta = 1.0, gamma = 0.2, delta = 1.0, p = 0.04;
        double r = 0.0;
        vcs = new VCS_Function(s0->nb_left_boxes, *s0->cont, alpha, beta, gamma, p, delta, 0.0, r);

        EnvUtilsPython::get_blocks_data(s0, block_features, block_id_to_index, block_index_to_id);
        update_full_batch_bsm();
    }

    py::array_t<float> get_block_features() {
        return block_features;
    }

    py::array_t<int32_t> get_action_blocks_batch_bsm() {
        return action_blocks_bsm;
    }

    py::array_t<int32_t> get_action_blocks_batch_greedy() {
        return action_blocks_gr;
    }

    py::array_t<float> get_action_features_batch_bsm() {
        return action_features_bsm;
    }

    py::array_t<float> get_action_features_batch_greedy() {
        return action_features_gr;
    }

    py::array_t<int32_t> get_placed_blocks_batch_bsm() {
        return placed_blocks_bsm;
    }

    py::array_t<int32_t> get_placed_blocks_batch_greedy() {
        return placed_blocks_gr;
    }

    py::array_t<float> get_placed_features_batch_bsm() {
        return placed_features_bsm;
    }

    py::array_t<float> get_placed_features_batch_greedy() {
        return placed_features_gr;
    }

    py::array_t<float> get_space_features_batch_bsm() {
        return space_features_bsm;
    }

    py::array_t<float> get_space_features_batch_greedy() {
        return space_features_gr;
    }

    py::dict get_full_batch_bsm() {
        update_full_batch_bsm(); 
        py::dict d;
        d["act_blocks"] = action_blocks_bsm;
        d["act_feats"] = action_features_bsm;
        d["pl_blocks"] = placed_blocks_bsm;
        d["pl_feats"] = placed_features_bsm;
        d["sp_feats"] = space_features_bsm;
        return d;
    }

    py::dict get_full_batch_greedy() {
        update_full_batch_greedy(); 
        py::dict d;
        d["act_blocks"] = action_blocks_gr;
        d["act_feats"] = action_features_gr;
        d["pl_blocks"] = placed_blocks_gr;
        d["pl_feats"] = placed_features_gr;
        d["sp_feats"] = space_features_gr;
        return d;
    }

    void transition_bsm(std::vector<std::vector<int>> selected_indexes_lists) {
        // Limpiar el batch anterior para evitar leaks o datos sucios
        batch_items.clear();
        
        int i = 0;
        // Iteramos sobre los nodos actuales (current_nodes es std::list)
        for (auto s : current_nodes) {
            for (auto block_idx : selected_indexes_lists[i]) {
                std::list<Action*> actions;
                s->get_actions(actions);

                int block_id = block_index_to_id[block_idx];

                for (auto a : actions) {
                    clp::clpAction* ca = dynamic_cast<clp::clpAction*>(a);
                    
                    if (ca && ca->block.id == block_id) {
                        clpState* state_copy = dynamic_cast<clpState*>(s->clone());
                        state_copy->transition(*a);

                        list<Action*> child_actions;
                        state_copy->get_actions(child_actions);
                        
                        if (child_actions.size() > 0) {
                            batch_items.push_back({s, state_copy});
                        }
                        
                        // Liberar la lista de acciones para evitar memory leaks
                        for (auto action_ptr : actions) delete action_ptr;
                        for (auto action_ptr : child_actions) delete action_ptr;
                        break;
                    }
                }
            }
            i++;
        }

        // Vaciar current_nodes
        // transition_greedy se encargará de reconstruir la lista
        current_nodes.clear();
    }

    void transition_greedy(std::vector<int> selected_indexes) {
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
                // ¡Es una hoja! 
                double volume = item.current->get_value();
                
                // Actualizar el mejor volumen global
                if (volume > best_volume) {
                    best_volume = volume;
                }

                if (state_actions.find(-volume) == state_actions.end()) {
                    state_actions[-volume] = std::make_pair(item.original_node, item.current);
                }
            } else {
                // Todavía tiene acciones: limpiar memoria y conservar para el siguiente paso
                for (auto a : next_actions) delete a;
                remaining_items.push_back(item); 
            }
        }
        // Al final, batch_items se actualiza con los que no fueron hojas
        batch_items = std::move(remaining_items);

        if (batch_items.empty()) {
			list<State*> next_states = get_next_states(state_actions);
			for (State *s : next_states)
			{
				clpState* s_copy = dynamic_cast<clpState *>(s);
				list<Action *> actions;
				s_copy->get_actions(actions);

				if (actions.size() > 0) {
					current_nodes.push_back(dynamic_cast<clpState *>(s));
				}
			}
        }
    }

    bool is_greedy_finished() {
        return batch_items.empty();
    }

    bool is_bsm_finished() {
        return current_nodes.empty();
    }


private:
    VCS_Function *vcs;
    clpState* s0;
	std::list<clpState *> current_nodes;
    vector<BatchItem> batch_items;
    map<double, pair<State *, State *>> state_actions;

    map<int, int> block_id_to_index;
    std::vector<int> block_index_to_id;

    py::array_t<float> block_features;

    py::array_t<int32_t> action_blocks_bsm;
    py::array_t<float> action_features_bsm;
    py::array_t<int32_t> placed_blocks_bsm;
    py::array_t<float> placed_features_bsm;
    py::array_t<float> space_features_bsm;

    py::array_t<int32_t> action_blocks_gr;
    py::array_t<float> action_features_gr;
    py::array_t<int32_t> placed_blocks_gr;
    py::array_t<float> placed_features_gr;
    py::array_t<float> space_features_gr;

    template<class map_container>
    list<State*> get_next_states(map_container& sorted_states) {
        list<State*> nextS;
        typename map_container::iterator state_action=sorted_states.begin();

        int k = 0;
        while(state_action!=sorted_states.end()) {
            State* s = state_action->second.first;
            State* final_state=state_action->second.second;
            Action* a = (s) ? s->next_action(*final_state) : NULL;

            if (nextS.size() < w && a) {
                State* p=s;
                s=s->clone();
                state_action->second.first=s;
                s->transition(*a);
                nextS.push_back(s);
                p->add_children(s);
            }
            else state_action->second.first=NULL;

            if (k >= w) {
                delete final_state;
                state_action=sorted_states.erase(state_action);
            }
            else state_action++;

            if (a) delete a;
            k++;
        }

        return nextS;
    }

    // Helper para extraer los punteros de current_nodes (std::list)
    std::vector<clpState*> get_current_nodes_vec() {
        return {current_nodes.begin(), current_nodes.end()};
    }

    // Helper para extraer los punteros 'current' de batch_items
    std::vector<clpState*> get_greedy_nodes_vec() {
        std::vector<clpState*> nodes;
        nodes.reserve(batch_items.size());
        for (auto& item : batch_items) nodes.push_back(item.current);
        return nodes;
    }

    void update_full_batch_bsm() {
        std::vector<clp::clpState*> nodes = get_current_nodes_vec();

        EnvUtilsPython::get_actions_data_batch(nodes, vcs, w, block_id_to_index, action_blocks_bsm, action_features_bsm);
        EnvUtilsPython::get_placed_data_batch(nodes, block_id_to_index, placed_blocks_bsm, placed_features_bsm);
        EnvUtilsPython::get_space_features_batch(nodes, space_features_bsm);
    }

    void update_full_batch_greedy() {
        std::vector<clp::clpState*> nodes = get_greedy_nodes_vec();

        EnvUtilsPython::get_actions_data_batch(nodes, vcs, w, block_id_to_index, action_blocks_gr, action_features_gr);
        EnvUtilsPython::get_placed_data_batch(nodes, block_id_to_index, placed_blocks_gr, placed_features_gr);
        EnvUtilsPython::get_space_features_batch(nodes, space_features_gr);
    }
};

// Definición del módulo Pybind11
PYBIND11_MODULE(bsm_engine, m) {
    py::class_<BSM_ENV>(m, "BSM_ENV")
        .def(py::init<std::string, int, int>(), 
                py::arg("filename"), 
                py::arg("instance_number"), 
                py::arg("w"))
        .def_readwrite("best_volume", &BSM_ENV::best_volume)

        // Bloques
        .def("get_block_features", &BSM_ENV::get_block_features)

        // Fase BSM (Basada en current_nodes)
        .def("get_action_blocks_batch_bsm", &BSM_ENV::get_action_blocks_batch_bsm)
        .def("get_action_features_batch_bsm", &BSM_ENV::get_action_features_batch_bsm)
        .def("get_placed_blocks_batch_bsm", &BSM_ENV::get_placed_blocks_batch_bsm)
        .def("get_placed_features_batch_bsm", &BSM_ENV::get_placed_features_batch_bsm)
        .def("get_space_features_batch_bsm", &BSM_ENV::get_space_features_batch_bsm)
        .def("get_full_batch_bsm", &BSM_ENV::get_full_batch_bsm)

        // Fase Greedy (Basada en batch_items del rollout)
        .def("get_action_blocks_batch_greedy", &BSM_ENV::get_action_blocks_batch_greedy)
        .def("get_action_features_batch_greedy", &BSM_ENV::get_action_features_batch_greedy)
        .def("get_placed_blocks_batch_greedy", &BSM_ENV::get_placed_blocks_batch_greedy)
        .def("get_placed_features_batch_greedy", &BSM_ENV::get_placed_features_batch_greedy)
        .def("get_space_features_batch_greedy", &BSM_ENV::get_space_features_batch_greedy)
        .def("get_full_batch_greedy", &BSM_ENV::get_full_batch_greedy)

        // Métodos de control de flujo
        .def("transition_bsm", &BSM_ENV::transition_bsm, py::arg("selected_ids_lists"))
        .def("transition_greedy", &BSM_ENV::transition_greedy, py::arg("selected_ids"))
        .def("is_bsm_finished", &BSM_ENV::is_bsm_finished)
        .def("is_greedy_finished", &BSM_ENV::is_greedy_finished);
}
