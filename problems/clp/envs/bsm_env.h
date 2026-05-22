#ifndef BSM_ENV_H
#define BSM_ENV_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "clpState.h"
#include "BSG.h"
#include "Greedy.h"
#include "VCS_Function.h"
#include "BlockMetrics.h"
#include "envs/env_utils_python.h"
#include "envs/env.h"

namespace py = pybind11;

struct BatchItem {
	clpState* original_node;
	clpState* current;
};

class BSM_ENV : public ENV {
public:
    int w;

    BSM_ENV(clpState* s0, int w, double timelimit=99999.9, std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now()) : ENV(s0, timelimit, start_time) {
        this->w = w;
    }

    BSM_ENV(std::string filename, int instance_number, int w, double min_fr, double timelimit=99999.9, std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now()) 
        : BSM_ENV(new_state(filename, instance_number, min_fr, 10000, clpState::BR), w, timelimit, start_time) {}

    virtual ~BSM_ENV() {
        delete vcs;
    }
};

#endif