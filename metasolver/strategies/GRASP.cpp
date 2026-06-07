/*
 * GRASP.cpp - Corrección de actualización de best_state (Estado Terminal Completo)
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
    
    // Estructura local para emparejar la acción con el valor macro final de su simulación
    struct CandidatePair {
        Action* original_action;
        double macro_value;
    };
    std::vector<CandidatePair> evaluated_candidates;

    // Guardamos las mejores soluciones terminales de cada rollout para no perderlas
    // Llave: valor macro, Valor: puntero al clon del estado terminal completo
    std::multimap<double, State*, std::greater<double>> terminal_states_pool;

    // 2. Ejecutamos el rollout para cada acción candidata
    for (Action* act : actions_vector) {
        if (get_time() > timelimit) break;

        State* candidate_state = current_state->clone();
        candidate_state->transition(*act);

        // Corremos el algoritmo Greedy externo hasta llegar al estado terminal
        double solution_value = evaluator_strategy.run(*candidate_state, timelimit, begin_time);

        // EXTRAEMOS EL MEJOR ESTADO TERMINAL: Conseguimos el clon del contenedor completamente lleno 
        // desde el best_state interno de la estrategia que acaba de terminar.
        const State* internal_terminal = evaluator_strategy.get_best_state();
        State* terminal_clone = internal_terminal ? internal_terminal->clone() : candidate_state->clone();

        CandidatePair cp;
        cp.original_action = act;
        cp.macro_value = solution_value;
        evaluated_candidates.push_back(cp);

        // Almacenamos el estado terminal indexado por su volumen
        terminal_states_pool.insert(std::make_pair(solution_value, terminal_clone));

        delete candidate_state;
    }

    int n_candidates = evaluated_candidates.size();
    if (n_candidates == 0) {
        for (Action* act : actions_vector) delete act;
        return next_generation;
    }

    // 3. ORDENAMOS las acciones candidatas de MAYOR a PEOR en función de su rollout
    std::sort(evaluated_candidates.begin(), evaluated_candidates.end(), 
              [](const CandidatePair& a, const CandidatePair& b) {
                  return a.macro_value > b.macro_value;
              });

    // 4. SELECCIÓN GEOMÉTRICA PARAMETRIZADA (Ej: 40%, 20%, 10%...)
    int chosen_idx = 0;
    double r = (double)rand() / RAND_MAX;
    
    // 0.4 = 40% -> 24% -> 14.4% -> 8.6%...
    double p_select = 0.3; 
    double current_probability = p_select;

    for (int i = 0; i < n_candidates; ++i) {
        // Si el número aleatorio cae en el rango o es el último candidato disponible, se selecciona
        if (r < current_probability || i == n_candidates - 1) {
            chosen_idx = i;
            break;
        }
        r -= current_probability;
        current_probability *= (1.0 - p_select);
    }

    Action* selected_action = evaluated_candidates[chosen_idx].original_action;
    double final_chosen_value = evaluated_candidates[chosen_idx].macro_value;

    // 5. El estado real que avanza en el camino oficial de este arranque GRASP es el hijo intermedio
    State* next_state = current_state->clone();
    next_state->transition(*selected_action);
    next_generation.push_back(next_state);

    // 6. ACTUALIZACIÓN ADAPTATIVA CRUCIAL:
    // Comparamos contra el valor del mejor récord global real.
    if (final_chosen_value > get_best_value()) {
        if (best_state) delete best_state;
        
        // Buscamos en el pool el estado terminal completo correspondiente al valor ganador
        auto pool_it = terminal_states_pool.find(final_chosen_value);
        if (pool_it != terminal_states_pool.end()) {
            best_state = pool_it->second->clone(); // Guardamos la solución terminal COMPLETA yllena
        } else {
            best_state = next_state->clone();
        }
        
        cout << "[GRASP_LookAhead_Fixed] NUEVO RÉCORD GLOBAL ENCONTRADO (" << get_time() << "): " 
             << get_best_value() << endl;
    }

    // 7. LIMPIEZA ABSOLUTA DE MEMORIA RAM
    for (auto& pair : terminal_states_pool) {
        delete pair.second;
    }
    for (Action* act : actions_vector) {
        delete act;
    }

    return next_generation;
}

} /* namespace metasolver */