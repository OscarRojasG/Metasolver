#include "MCTS_RolloutStrategy.h"
#include <vector>
#include <cmath>
#include <algorithm>

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

    std::vector<Action*> actions_vector(rcl_actions.begin(), rcl_actions.end());
    std::vector<double> current_step_volumes;

    Action* best_action_overall = nullptr;

    for (Action* act : actions_vector) {
        if (get_time() > timelimit) break;

        State* child_state = current_state->clone();
        child_state->transition(*act);

        double total_rollout_volume = 0.0;
        int successful_rollouts = 0;

        for (int n = 0; n < num_rollouts; ++n) {
            if (get_time() > timelimit) break;
            State* rollout_state = child_state->clone();
            while (true) {
                list<Action*> local_actions;
                rollout_state->get_actions(local_actions);
                if (local_actions.empty()) break;

                Action* chosen_sim_action = nullptr;
                if ((rand() % 100) < 20) {
                    list<Action*> best_actions;
                    get_best_actions(*rollout_state, best_actions, 1);
                    if (!best_actions.empty()) chosen_sim_action = best_actions.front();
                }

                if (chosen_sim_action == nullptr) {
                    std::vector<Action*> sim_vector(local_actions.begin(), local_actions.end());
                    chosen_sim_action = sim_vector[rand() % sim_vector.size()];
                    for (Action* a : sim_vector) if (a != chosen_sim_action) delete a;
                } else {
                    for (Action* a : local_actions) if (a != chosen_sim_action) delete a;
                }
                rollout_state->transition(*chosen_sim_action);
                delete chosen_sim_action;
            }
            total_rollout_volume += rollout_state->get_value();
            successful_rollouts++;
            delete rollout_state;
        }

        double average_volume = (successful_rollouts > 0) ? (total_rollout_volume / successful_rollouts) : 0.0;
        current_step_volumes.push_back(average_volume);
        delete child_state;
    }

    // --- SELECCIÓN GEOMÉTRICA ÚNICA ---
    if (!current_step_volumes.empty()) {
        rollout_history.push_back(current_step_volumes);

        std::vector<std::pair<int, double>> indexed_volumes;
        for (size_t i = 0; i < current_step_volumes.size(); ++i) {
            indexed_volumes.push_back({(int)i, current_step_volumes[i]});
        }

        std::sort(indexed_volumes.begin(), indexed_volumes.end(), 
                  [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                      return a.second > b.second;
                  });

        double p = 0.2;
        double r = (double)rand() / RAND_MAX;
        double cumulative_p = 0.0;
        int selected_idx = indexed_volumes.back().first;

        for (size_t i = 0; i < indexed_volumes.size(); ++i) {
            double prob_i = p * std::pow(1.0 - p, i);
            cumulative_p += prob_i;
            if (r <= cumulative_p) {
                selected_idx = indexed_volumes[i].first;
                break;
            }
        }
        best_action_overall = actions_vector[selected_idx];
    }

    if (best_action_overall != nullptr) {
        State* next_state = current_state->clone();
        next_state->transition(*best_action_overall);
        next_generation.push_back(next_state);

        if (next_state->get_value() > get_best_value()) {
            if (best_state) delete best_state;
            best_state = next_state->clone();
        }
    }

    for (Action* act : actions_vector) delete act;
    return next_generation;
}

} /* namespace metasolver */