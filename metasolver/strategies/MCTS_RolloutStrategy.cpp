#include "MCTS_RolloutStrategy.h"
#include <vector>

namespace metasolver {

// Función para recuperar el histórico desde Python si es necesario
std::vector<std::vector<double>> MCTS_RolloutStrategy::get_rollout_history() const {
    return rollout_history;
}

list<State*> MCTS_RolloutStrategy::next(list<State*>& S) {
    list<State*> next_generation;

    if (S.empty() || get_time() > timelimit) return next_generation;

    State* current_state = S.front();

    list<Action*> rcl_actions;
    get_best_actions(*current_state, rcl_actions, w);

    if (rcl_actions.empty()) return next_generation;

    Action* best_action_overall = nullptr;
    double best_average_volume = -1.0;

    // 1. Iterar sobre las acciones candidatas
    for (Action* act : rcl_actions) {
        if (get_time() > timelimit) break;

        // Clonación segura para la simulación
        State* child_state = current_state->clone();
        child_state->transition(*act);

        double total_vol = 0.0;
        int successful_rollouts = 0;

        // 2. Rollout
        for (int n = 0; n < num_rollouts; ++n) {
            State* rollout = child_state->clone();
            
            while (true) {
                list<Action*> acts;
                rollout->get_actions(acts);
                if (acts.empty()) break;
                
                // Elegir aleatoriamente
                std::vector<Action*> v(acts.begin(), acts.end());
                Action* choice = v[rand() % v.size()];
                
                rollout->transition(*choice);
                
                // Limpieza de acciones locales
                for (Action* a : acts) delete a;
            }
            
            total_vol += rollout->get_value();
            successful_rollouts++;
            delete rollout;
        }

        double avg = (successful_rollouts > 0) ? (total_vol / successful_rollouts) : 0.0;

        if (avg > best_average_volume) {
            best_average_volume = avg;
            best_action_overall = act;
        }
        delete child_state;
    }

    // 3. Crear la nueva generación
    if (best_action_overall != nullptr) {
        State* next_state = current_state->clone();
        next_state->transition(*best_action_overall);
        next_generation.push_back(next_state);
    }

    // 4. Limpieza final de acciones candidatas
    for (Action* act : rcl_actions) delete act;

    return next_generation;
}

} /* namespace metasolver */