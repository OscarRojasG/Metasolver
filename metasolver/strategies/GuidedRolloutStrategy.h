#ifndef STRATEGIES_GUIDED_ROLLOUT_STRATEGY_H_
#define STRATEGIES_GUIDED_ROLLOUT_STRATEGY_H_

#include "../SearchStrategy.h"
#include "ActionEvaluator.h" // Asegúrate de que este sea el header de tu VCS
#include <vector>

namespace metasolver {

class GuidedRolloutStrategy : public SearchStrategy {
public:
    /**
     * @param evl Evaluador VCS para guiar la selección.
     * @param w Límite de mejores candidatos a considerar.
     * @param num_rollouts Cantidad de simulaciones por cada acción candidata.
     */
    GuidedRolloutStrategy(ActionEvaluator* evl, int w, int num_rollouts)
        : SearchStrategy(evl), w(w), num_rollouts(num_rollouts) {}

    virtual ~GuidedRolloutStrategy() = default;
    
    // Método para limpiar sin destruir el estado raíz si no es necesario
    virtual void clean(list<State*>& S) override;

    double perform_rollout(State& s);

private:
    int w;
    int num_rollouts;
};

} /* namespace metasolver */

#endif