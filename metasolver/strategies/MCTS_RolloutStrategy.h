/*
 * MCTS_RolloutStrategy.h
 */

#ifndef STRATEGIES_MCTS_ROLLOUT_STRATEGY_H_
#define STRATEGIES_MCTS_ROLLOUT_STRATEGY_H_

#include "../SearchStrategy.h"
#include <vector>
#include <cstdlib>

namespace metasolver {

class MCTS_RolloutStrategy : public SearchStrategy {
public:
    /**
     * Constructor
     * @param evl El evaluador VCS para filtrar las mejores 'w' acciones iniciales.
     * @param w El límite de candidatos a expandir en el paso constructivo actual.
     * @param num_rollouts Cantidad 'N' de simulaciones completamente aleatorias por cada hijo.
     */
    MCTS_RolloutStrategy(ActionEvaluator* evl, int w, int num_rollouts) 
        : SearchStrategy(evl), w(w), num_rollouts(num_rollouts), mcts_best_state(nullptr) {}

    virtual ~MCTS_RolloutStrategy() = default;

    /**
     * Ejecuta una iteración constructiva basada en proyecciones promedio de N Rollouts aleatorios.
     */
    virtual list<State*> next(list<State*>& S) override;

    /**
     * Evita que la clase base destruya prematuramente el estado raíz oficial.
     */
    virtual void clean(list<State*>& S) override {
        if (S.size() <= 1) {
            S.clear();
            return;
        }
        while(!S.empty()){ 
            delete S.front(); 
            S.pop_front(); 
        }
    }

    std::vector<std::vector<double>> get_rollout_history() const;
    std::vector<std::vector<double>> get_rollout_std_history() const;

    State *mcts_best_state;

private:
    int w;
    int num_rollouts; // Parámetro 'N'
    std::vector<std::vector<double>> rollout_history;
    std::vector<std::vector<double>> rollout_std_history;
};

} /* namespace metasolver */

#endif /* STRATEGIES_MCTS_ROLLOUT_STRATEGY_H_ */