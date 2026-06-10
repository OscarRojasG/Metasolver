#include "GuidedRolloutStrategy.h"
#include <algorithm>
#include <iostream>

namespace metasolver {

double GuidedRolloutStrategy::perform_rollout(State& s) {
    double total_value = 0.0;

    // Ejecutar N rollouts (tu parámetro num_rollouts)
    for (int n = 0; n < num_rollouts; ++n) {
        // Clonamos el estado inicial de la simulación
        State* sim_state = s.clone();

        // Simulación guiada por VCS hasta el final
        while (true) {
            list<Action*> best_local_actions;
            // Usamos el mismo evaluador (VCS) para decidir el paso local
            get_best_actions(*sim_state, best_local_actions, 1);
            
            if (best_local_actions.empty()) break;

            // Transicionar con la mejor acción guiada
            sim_state->transition(*best_local_actions.front());
            
            // Limpiar acciones locales después de usarlas
            for (Action* a : best_local_actions) delete a;
        }

        // Acumular valor
        total_value += sim_state->get_value();
        
        // Limpiar memoria de la simulación
        delete sim_state;
    }

    return total_value / num_rollouts;
}

void GuidedRolloutStrategy::clean(list<State*>& S) {
    while(!S.empty()){ 
        delete S.front(); 
        S.pop_front(); 
    }
}

} /* namespace metasolver */