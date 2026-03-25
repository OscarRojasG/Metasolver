#ifndef ENV_H
#define ENV_H

#include "clpState.h"
#include "BSG.h"
#include "Greedy.h"
#include "VCS_Function.h"
#include "BlockMetrics.h"
#include <chrono>

class ENV {
public:
    double timelimit;
    double final_time = 0;
    clpState* s0;

    ENV(clpState* s0, double timelimit=99999.9, std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now()) {
        this->s0 = s0;
        this->timelimit = timelimit;
        this->start_time = start_time;
    }

protected:
    std::chrono::steady_clock::time_point start_time;
    
    double get_elapsed_time() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time).count();
    }
};

#endif