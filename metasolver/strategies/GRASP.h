/*
 * GRASP.h
 */

#ifndef STRATEGIES_GRASP_H_
#define STRATEGIES_GRASP_H_

#include "../SearchStrategy.h"
#include <vector>
#include <cstdlib>

namespace metasolver {

class GRASP : public SearchStrategy {
public:
    /**
     * Constructor desacoplado al estilo BSG
     * @param evl Evaluador de acciones locales para armar la RCL (ej: VCS puro o redes neuronales).
     * @param evaluator_strategy Estrategia subyacente para evaluar el potencial completo del estado resultante.
     * @param w Tamaño de la Restricted Candidate List (RCL).
     */
    GRASP(ActionEvaluator* evl, SearchStrategy& evaluator_strategy, int w) 
        : SearchStrategy(evl), evaluator_strategy(evaluator_strategy), w(w) {}

    virtual list<State*> next(list<State*>& S) override;

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

    void set_threshold(int new_w) { w = new_w; }
    int get_threshold() const { return w; }

private:
    SearchStrategy& evaluator_strategy; // Estrategia para rollouts / evaluaciones macro
    int w; 
};

} /* namespace metasolver */

#endif /* STRATEGIES_GRASP_H_ */