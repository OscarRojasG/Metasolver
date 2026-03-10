#ifndef STRATEGIES_BSM_H_
#define STRATEGIES_BSM_H_

#include "BSG.h"

namespace metasolver {

class BSM : public BSG {
public:
    BSM(
        ActionEvaluator* evl,
        SearchStrategy& greedy,
        int beams,
        double p_elite = 0.0,
        int max_level_size = 0,
        bool plot = false
    )
    : BSG(evl, greedy, beams, p_elite, max_level_size, plot) {}

    void set_best_actions(const std::list<Action*>& actions) {
        external_best_actions = actions;
    }

protected:
    virtual void obtain_best_actions(
        State& state,
        std::list<Action*>& best_actions,
        int w
    ) override {
        best_actions = external_best_actions;
    }

private:
    std::list<Action*> external_best_actions;
};

} // namespace metasolver

#endif