#ifndef BSM_VCS_H
#define BSM_VCS_H

#include "clpState.h"
#include "BSG.h"
#include "Greedy.h"
#include "VCS_Function.h"
#include "envs/bsm_env.h"

class BSM_VCS : public BSM_ENV {
public:
    SearchStrategy *gr;

    BSM_VCS(std::string filename, int instance_number, int w);

    BSM_VCS(clpState* s0, int w, double timelimit = 99999.9, std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now());

    virtual ~BSM_VCS();

    void transition(std::vector<std::vector<int>> selected_indexes_lists) override;
    
};

#endif