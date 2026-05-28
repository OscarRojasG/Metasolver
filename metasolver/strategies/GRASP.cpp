/*
 * GRASP.cpp - Ordenamiento y Selección Basados Estrictamente en SearchStrategy
 */

#include "GRASP.h"
#include <map>
#include <vector>
#include <cstdlib>
#include <algorithm>

namespace metasolver {

list<State*> GRASP::next(list<State*>& S) {
    list<State*> next_generation;

    if (S.empty() || get_time() > timelimit) return next_generation;

    State* current_state = S.front();

    // 1. Generamos las 'w' mejores acciones usando el evaluador local rápido (VCS)
    list<Action*> rcl_actions;
    get_best_actions(*current_state, rcl_actions, w);

    if (rcl_actions.empty()) {
        return next_generation;
    }

    std::vector<Action*> actions_vector(rcl_actions.begin(), rcl_actions.end());
    
    // Estructura local para emparejar la acción con el valor de su simulación futura
    struct CandidatePair {
        Action* original_action;
        double macro_value;
    };
    std::vector<CandidatePair> evaluated_candidates;

    // 2. Ejecutamos el rollout para cada acción candidata
    for (Action* act : actions_vector) {
        if (get_time() > timelimit) break;

        // Clonamos y aplicamos la acción para evaluar esta rama
        State* candidate_state = current_state->clone();
        candidate_state->transition(*act);

        // Corremos el algoritmo Greedy externo hasta el final
        double solution_value = evaluator_strategy.run(*candidate_state, timelimit, begin_time);

        // Almacenamos el par
        CandidatePair cp;
        cp.original_action = act;
        cp.macro_value = solution_value;
        evaluated_candidates.push_back(cp);

        // Liberamos la simulación, conservando la acción original
        delete candidate_state;
    }

    int n_candidates = evaluated_candidates.size();
    if (n_candidates == 0) {
        for (Action* act : actions_vector) delete act;
        return next_generation;
    }

    // 3. ORDENAMOS las acciones de MAYOR a PEOR en función del valor de la SearchStrategy
    std::sort(evaluated_candidates.begin(), evaluated_candidates.end(), 
              [](const CandidatePair& a, const CandidatePair& b) {
                  return a.macro_value > b.macro_value;
              });

    // 4. SELECCIÓN GEOMÉTRICA sobre el nuevo ordenamiento macro
    // Índice 0 (Solución macro más alta): 50% de probabilidad
    // Índice 1 (Segunda solución macro más alta): 25% de probabilidad, etc.
    int chosen_idx = 0;
    double r = (double)rand() / RAND_MAX;
    double current_probability = 0.5;

    for (int i = 0; i < n_candidates; ++i) {
        if (r < current_probability || i == n_candidates - 1) {
            chosen_idx = i;
            break;
        }
        r -= current_probability;
        current_probability *= 0.5;
    }

    // Rescatamos la acción ganadora real elegida por el potencial a futuro
    Action* selected_action = evaluated_candidates[chosen_idx].original_action;
    double final_chosen_value = evaluated_candidates[chosen_idx].macro_value;

    // 5. Aplicamos la acción ganadora oficial sobre un clon definitivo para el pipeline
    State* next_state = current_state->clone();
    next_state->transition(*selected_action);

    // Actualización adaptativa del récord global basado en la cota de la estrategia
    if (final_chosen_value > get_best_value()) {
        if (best_state) delete best_state;
        best_state = next_state->clone();
        
        cout << "[GRASP_LookAhead_Verified] Nueva mejor solución macro encontrada (" << get_time() << "): " 
             << final_chosen_value << endl;
    }

    next_generation.push_back(next_state);

    // 6. LIMPIEZA RIGUROSA DE MEMORIA
    // Destruimos todas las acciones candidatas, ya que la transición ya se consumó en next_state
    for (Action* act : actions_vector) {
        delete act;
    }

    return next_generation;
}

} /* namespace metasolver */