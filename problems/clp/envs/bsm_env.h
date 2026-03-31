#ifndef BSM_ENV_H
#define BSM_ENV_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "clpState.h"
#include "BSG.h"
#include "Greedy.h"
#include "VCS_Function.h"
#include "BlockMetrics.h"
#include "envs/env_utils_python.h"
#include "envs/env.h"

namespace py = pybind11;

struct BatchItem {
	clpState* original_node;
	clpState* current;
};

class BSM_ENV : public ENV {
public:
    int w;
    double best_volume;

    BSM_ENV(clpState* s0, int w, double timelimit=99999.9, std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now()) : ENV(s0, timelimit, start_time) {
        current_nodes.push_back(dynamic_cast<clp::clpState*>(s0->clone()));

        best_volume = 0.0;
        this->w = w;

        // VCS y parámetros
        double alpha = 4.0, beta = 1.0, gamma = 0.2, delta = 1.0, p = 0.04;
        double r = 0.0;
        vcs = new VCS_Function(s0->nb_left_boxes, *s0->cont, alpha, beta, gamma, p, delta, 0.0, r);

        EnvUtilsPython::get_blocks_data(s0, block_features, block_id_to_index, block_index_to_id);
    }

    BSM_ENV(std::string filename, int instance_number, int w, double min_fr=0.98, double timelimit=99999.9, std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now()) 
        : BSM_ENV(new_state(filename, instance_number, min_fr, 10000, clpState::BR), w, timelimit, start_time) {}

    virtual ~BSM_ENV() {
        delete vcs;
    }

    py::array_t<float> get_block_features() {
        return block_features;
    }

    py::dict get_batch_dict() {
        update_batches(); 
        py::dict d;
        d["act_blocks"] = action_blocks;
        d["act_feats"] = action_features;
        d["pl_blocks"] = placed_blocks;
        d["pl_feats"] = placed_features;
        d["sp_feats"] = space_features;
        d["biases"] = biases;
        return d;
    }

    virtual void transition(std::vector<std::vector<int>> selected_indexes_lists) { 
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
    }

    bool is_finished() {
        return current_nodes.empty();
    }

protected:
    std::list<clpState *> current_nodes;
    vector<BatchItem> batch_items;
    VCS_Function *vcs;

    map<int, int> block_id_to_index;
    std::vector<int> block_index_to_id;

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
                s=s->clone();
                s->transition(*a);
                nextS.push_back(s);
            }

            delete final_state;
            state_action=sorted_states.erase(state_action);

            if (a) delete a;
            k++;
        }

        return nextS;
    }

private:
    py::array_t<float> block_features;
    py::array_t<int32_t> action_blocks;
    py::array_t<float> action_features;
    py::array_t<int32_t> placed_blocks;
    py::array_t<float> placed_features;
    py::array_t<float> space_features;
    py::array_t<float> biases;

    // Helper para extraer los punteros de current_nodes (std::list)
    std::vector<clpState*> get_current_nodes_vec() {
        return {current_nodes.begin(), current_nodes.end()};
    }

    void update_batches() {
        std::vector<clp::clpState*> nodes = get_current_nodes_vec();

        EnvUtilsPython::get_actions_data_batch(nodes, vcs, w, block_id_to_index, action_blocks, action_features, biases);
        EnvUtilsPython::get_placed_data_batch(nodes, block_id_to_index, placed_blocks, placed_features);
        EnvUtilsPython::get_space_features_batch(nodes, space_features);
    }
};

#endif