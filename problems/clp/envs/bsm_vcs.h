#ifndef BSM_VCS_H
#define BSM_VCS_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "clpState.h"
#include "BSG.h"
#include "Greedy.h"
#include "VCS_Function.h"
#include "env_utils.h"
#include "envs/env.h"

class BSM_VCS : public ENV {
public:
    int w;

    BSM_VCS(std::string filename, int instance_number, int w, double min_fr=1);

    BSM_VCS(clpState* s0, int w, double timelimit = 99999.9, std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now());

    virtual ~BSM_VCS();

    void transition(std::vector<std::vector<int>> selected_indexes_lists);

    std::vector<std::vector<float>> get_action_data_batch() { return action_data; }
    std::vector<std::vector<float>> get_pblock_data_batch() { return placed_data; }
    std::vector<std::vector<float>> get_space_data_batch()  { return space_data;  }

    bool is_finished();

private:
    std::vector<clpState *> current_states;
    std::vector<std::pair<clpState *, clpState *>> succ_states;

    std::vector<std::vector<float>> action_data;
    std::vector<std::vector<float>> placed_data;
    std::vector<std::vector<float>> space_data;

    SearchStrategy *gr;

    bool completed = false;

    std::map<double, std::pair<State *, State *>> eval_succ_states();
    void update();
};

#endif