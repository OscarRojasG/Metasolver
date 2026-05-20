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

        // VCS y parámetros
        double alpha = 4.0, beta = 1.0, gamma = 0.2, delta = 1.0, p = 0.04;
        double r = 0.0;
        vcs = new VCS_Function(s0->nb_left_boxes, *s0->cont, alpha, beta, gamma, p, delta, 0.0, r);

        block_data = EnvUtilsPython::get_blocks_data(s0, block_id_to_index, block_index_to_id);
    }

    BSM_ENV(std::string filename, int instance_number, int w, double min_fr, double timelimit=99999.9, std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now()) 
        : BSM_ENV(new_state(filename, instance_number, min_fr, 10000, clpState::BR), w, timelimit, start_time) {}

    virtual ~BSM_ENV() {
        delete vcs;
    }

protected:
    VCS_Function *vcs;

    std::vector<float> block_data;
    map<int, int> block_id_to_index;
    std::vector<int> block_index_to_id;
};

#endif