/*
 * DoubleEffort.h
 *
 *  Created on: 6 jul. 2017
 *      Author: iaraya
 */

#ifndef STRATEGIES_DOUBLEEFFORT_H_
#define STRATEGIES_DOUBLEEFFORT_H_
#include "SearchStrategy.h"
#include "BSG.h"
#include <iostream>
#include <list>

using namespace std;

namespace metasolver {

class DoubleEffort : public SearchStrategy {
public:
	DoubleEffort(BSG& bsg, int max_w=999999) : bsg(bsg) {
		this->max_w = max_w;
	};

	virtual list<State*> next(list<State*>& S){
		State& s= **S.begin();

		bsg.run(*s.clone(), timelimit, begin_time);


		if(get_best_value() < bsg.get_best_value()){
			best_state=bsg.get_best_state()->clone();
			cout << "[DoubleEffort] new best_solution_found ("<< get_time() <<"): " << get_best_value() << endl;
		}

		bsg.double_effort();
		if (bsg.beams > max_w) {
			S.clear();
			return S;
		}

		return S;
	}

private:
	BSG& bsg;
	int max_w;
};

} /* namespace clp */

#endif /* STRATEGIES_DOUBLEEFFORT_H_ */
