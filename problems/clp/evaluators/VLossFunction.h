#ifndef HEURISTICS_VLOSSFUNCTION_H_
#define HEURISTICS_VLOSSFUNCTION_H_

#include "../../metasolver/ActionEvaluator.h"
#include "../objects2/Block.h"
#include <map>
#include <set>

using namespace metasolver;

namespace clp {

class VLossFunction : public ActionEvaluator {
public:
    // Constructor con el mapa determinista
    VLossFunction(const std::map<const BoxShape*, int, BoxShapeComparator>& nb_boxes, Vector3& dims, double r=0.0);

    virtual ~VLossFunction();

    virtual double eval_action(const State& , const Action& )=0;

protected:
    // Función Loss con mapa determinista
    double Loss(const std::map<const BoxShape*, int, BoxShapeComparator>& nb_boxes, const Block& block, const Space& free_space);

    /**
     * Returns an upper bound of the capacity that we can fill of the dimension with the
     * leaving boxes. Ahora usa el set determinista para iterar siempre en el mismo orden.
     */
    long compute_maxX(const std::map<const BoxShape*, int, BoxShapeComparator>& nb_boxes, const Block& block, 
                      long& lossX, long resX, long* mX, std::set<const BoxShape*, BoxShapeComparator>* listX);

    /**
     * Solve the knapsack problem
     */
    void solveKnapsack(const std::map<const BoxShape*, int, BoxShapeComparator>& nb_boxes, Vector3& dims);

private:

    /**
     * Genera los vectores mX y listX con orden determinista por ID de caja.
     */
    void compute_mX(const std::map<const BoxShape*, int, BoxShapeComparator>& nb_boxes, int X, long *mX, 
                    std::set<const BoxShape*, BoxShapeComparator>* listX, int dim);

    // Arrays para las soluciones del knapsack
    long *mL, *mW, *mH;
    
    // Arrays de sets que ahora usan el comparador por ID en lugar de dirección de memoria
    std::set<const BoxShape*, BoxShapeComparator> *listL, *listW, *listH;
};

} /* namespace clp */

#endif /* HEURISTICS_VLOSSFUNCTION_H_ */