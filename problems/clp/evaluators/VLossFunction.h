/*
 * VLossFunction.h
 *
 *  Created on: 2 jun. 2017
 *      Author: iaraya
 */

#include "../../metasolver/ActionEvaluator.h"
#include "../objects2/Block.h"

#ifndef HEURISTICS_VLOSSFUNCTION_H_
#define HEURISTICS_VLOSSFUNCTION_H_

using namespace metasolver;

namespace clp {

class VLossFunction : public ActionEvaluator{
public:
	VLossFunction(const std::map<const BoxShape*, int>& nb_boxes, Vector3& dims, double r=0.0);

	virtual ~VLossFunction();

	virtual double eval_action(const State& , const Action& )=0;


protected:
	double Loss(const std::map<const BoxShape*, int>& nb_boxes, const Block& block, const Space& free_space);

	/**
	 * Returns an upper bound of the capacity that we can fill of the dimension with the
	 * leaving boxes
	 */
	long compute_maxX(const std::map<const BoxShape*, int>& nb_boxes, const Block& block, long& lossX, long resX, long* mX, std::set<const BoxShape*>* listX);

	/**
	 * Solve the knapsack problem
	 * Generates the vectors mL, mW and mH.
	 * It also generates the box lists listL, listW and listH with the boxes that
	 * we can put for each capacity of the dimensions
	 * @param dims
	 */
	void solveKnapsack(const std::map<const BoxShape*, int>& nb_boxes, Vector3& dims);


private:

	/**
	 * This function generates the mX vectors with the solutions of the knapsack problem, and
	 * the vector listX with the lists of boxes we can put for each capacity
	 * @param X max length of the dimension
	 * @param mX vector with the knapsack solutions for each capacity
	 * @param listX vector of lists of boxes
	 * @param dim current dimension (0:L, 1:W, 2:H)
	 */
	void compute_mX(const std::map<const BoxShape*, int>& nb_boxes, int X, long *mX, std::set<const BoxShape*>* listX,  int dim);


    //for the knapsack solutions
    long *mL, *mW, *mH;
    std::set<const BoxShape*> *listL, *listW, *listH;
};

} /* namespace clp */

#endif /* HEURISTICS_VLOSSFUNCTION_H_ */
