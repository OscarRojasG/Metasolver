#ifndef ENV_UTILS2_H
#define ENV_UTILS2_H

#include <vector>
#include <list>
#include <map>
#include "clpState.h"
#include "VCS_Function.h"
#include "Greedy.h"

class EnvUtils {
public:
    static const int N_BLOCK_FEATURES = 4;
    static const int N_ACTION_FEATURES = 4;
    static const int N_PLACED_FEATURES = 4;
    static const int N_SPACE_FEATURES = 6;

    static void get_blocks_data(clpState* s0, std::vector<float>& out_features, std::map<int, int>& block_id_to_index, std::vector<int>& block_index_to_id) {
        out_features.clear();
        block_id_to_index.clear();
        block_index_to_id.clear();

        out_features.reserve(s0->valid_blocks.size() * N_BLOCK_FEATURES);

        int count = 0;
        for (const Block* block : s0->valid_blocks) {          
            // Insertamos consecutivamente las 4 características dinámicamente
            out_features.push_back((float)block->getL() / (float)s0->cont->getL());
            out_features.push_back((float)block->getW() / (float)s0->cont->getW());
            out_features.push_back((float)block->getH() / (float)s0->cont->getH());
            out_features.push_back((float)block->n_boxes);

            block_id_to_index[block->id] = count;
            block_index_to_id.push_back(block->id);
            count++;
        }
    }

    static std::multimap<double, Action*> get_ranked_actions(clpState* state, VCS_Function* vcs, int num_actions) {
        clpState* state_copy = dynamic_cast<clpState*>(state->clone());
        
        std::list<Action*> actions;
        state_copy->get_actions(actions);
        
        std::multimap<double, Action*> ranked_actions;
        for (auto a : actions) {
            double eval = vcs->eval_action(*state_copy, *a);
            if (eval > 0 && (ranked_actions.size() < num_actions || ranked_actions.begin()->first < eval)) {
                ranked_actions.insert({eval, a});
                if (ranked_actions.size() > num_actions) {
                    delete ranked_actions.begin()->second;
                    ranked_actions.erase(ranked_actions.begin());
                }
            } else { delete a; }
        }
        
        delete state_copy;
        return ranked_actions;
    }

    static void get_actions_data(clpState* state, VCS_Function *vcs, std::map<int, int>& block_id_to_index,
                                 std::vector<float>& out_features, int num_actions) {     
        std::multimap<double, Action*> ranked_actions = get_ranked_actions(state, vcs, num_actions);

        out_features.clear();
        out_features.reserve(ranked_actions.size() * N_ACTION_FEATURES);

        // Iteramos en reversa (de mejor a peor evaluación)
        for (auto it = ranked_actions.rbegin(); it != ranked_actions.rend(); ++it) {
            auto ca = static_cast<clp::clpAction*>(it->second);
            
            out_features.push_back((float)block_id_to_index[ca->block.id]);
            out_features.push_back((float)ca->metrics.vcs);
            out_features.push_back((float)ca->metrics.loss);
            out_features.push_back((float)ca->metrics.cs);

            delete it->second;
        }
    }

    static void get_placed_data(clpState* state, std::map<int, int>& block_id_to_index, std::vector<float>& out_features) {
        out_features.clear(); 

        std::list<const Action*> path = state->get_path();
        
        // 1. Perspectiva del ESPACIO ACTUAL (Tope del stack)
        clp::Space current_space = state->cont->spaces->top();
        const bool* current_anchors = current_space.get_anchor();
        
        double cL = (double)state->cont->getL();
        double cW = (double)state->cont->getW();
        double cH = (double)state->cont->getH();

        out_features.reserve(path.size() * N_PLACED_FEATURES);

        for (const Action* a : path) {
            auto ca = static_cast<const clp::clpAction*>(a);
            
            // Coordenada global absoluta que calcula el motor para esa acción pasada
            Vector3 coords = ca->space.get_location(ca->block);

            // 2. Reflejamos TODA la escena usando la perspectiva del espacio actual
            double bx = current_anchors[0] ? (coords.getX() + ca->block.getL()) * -1 + cL : coords.getX();
            double by = current_anchors[1] ? (coords.getY() + ca->block.getW()) * -1 + cW : coords.getY();
            double bz = current_anchors[2] ? (coords.getZ() + ca->block.getH()) * -1 + cH : coords.getZ();

            // 3. Insertamos las variables correspondientes a cada eje (CORREGIDO)
            out_features.push_back((float)block_id_to_index[ca->block.id]);
            out_features.push_back((float)(bx / cL));
            out_features.push_back((float)(by / cW)); // Asignado correctamente by
            out_features.push_back((float)(bz / cH)); // Asignado correctamente bz
        }
    }

    static void get_space_data(clpState* state, std::vector<float>& out_features) {
        out_features.clear();

        clp::Space space = state->cont->spaces->top();
        const bool* anchors = space.get_anchor();
        const Vector3 corner = space.get_corner();

        float cL = (float)state->cont->getL();
        float cW = (float)state->cont->getW();
        float cH = (float)state->cont->getH();

        out_features.reserve(N_SPACE_FEATURES);

        out_features.push_back((float)(anchors[0] ? corner.getX() * -1 + cL : corner.getX()) / cL);
        out_features.push_back((float)(anchors[1] ? corner.getY() * -1 + cW : corner.getY()) / cW);
        out_features.push_back((float)(anchors[2] ? corner.getZ() * -1 + cH : corner.getZ()) / cH);
        
        out_features.push_back((float)space.getL() / cL);
        out_features.push_back((float)space.getW() / cW);
        out_features.push_back((float)space.getH() / cH);
    }

    static double get_volume_ratio(clpState* state) {
        return state->cont->getOccupiedVolume() / state->cont->getVolume();
    }

    static std::vector<std::vector<float>> get_actions_data_batch(const std::vector<clpState*>& states, VCS_Function* vcs, std::map<int, int>& block_id_to_index, int num_actions) {
        std::vector<std::vector<float>> batch_features;
        batch_features.reserve(states.size());

        for (clpState* state : states) {
            std::vector<float> state_features;
            
            // 2. Extraemos las características enviando el multimap por referencia
            get_actions_data(state, vcs, block_id_to_index, state_features, num_actions);
            
            batch_features.push_back(std::move(state_features));
        }

        return batch_features;
    }

    static void get_actions_greedy_eval(clpState* state, VCS_Function* vcs,
                                        const std::multimap<double, Action*>& ranked_actions,
                                        std::vector<float>& out_greedy_values) {
        out_greedy_values.clear();
        out_greedy_values.reserve(ranked_actions.size());

        // Instanciamos la estrategia de búsqueda Greedy usando tu framework
        SearchStrategy* gr = new Greedy(vcs);

        for (auto it = ranked_actions.rbegin(); it != ranked_actions.rend(); ++it) {
            // Clonamos el estado para simular la transición sin alterar el entorno actual
            clpState* s_copy = static_cast<clpState*>(state->clone());
            s_copy->transition(*it->second);
            
            // Evaluamos el estado resultante con el rollout Greedy
            double value = gr->run(*s_copy);
            out_greedy_values.push_back((float)value);

            delete s_copy;
        }

        delete gr; // Limpieza de la estrategia
    }

    static std::vector<std::vector<float>> get_placed_data_batch(const std::vector<clpState*>& states, std::map<int, int>& block_id_to_index) {
        std::vector<std::vector<float>> batch_features;
        batch_features.reserve(states.size());

        for (clpState* state : states) {
            std::vector<float> state_features; // Nace vacío, tamaño 0
            
            // La función lo llenará con el tamaño exacto que necesite
            get_placed_data(state, block_id_to_index, state_features);
            
            batch_features.push_back(std::move(state_features));
        }

        return batch_features;
    }

    static std::vector<std::vector<float>> get_space_data_batch(const std::vector<clpState*>& states) {
        std::vector<std::vector<float>> batch_spaces;
        batch_spaces.reserve(states.size());

        for (clpState* state : states) {
            std::vector<float> space_features;
            get_space_data(state, space_features);
            batch_spaces.push_back(std::move(space_features));
        }

        return batch_spaces;
    }
};

#endif