#ifndef ValuePredictor_H
#define ValuePredictor_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <string>
#include <vector>
#include <map>
#include "clpState.h"
#include "BSG.h"
#include "Greedy.h"
#include "VCS_Function.h"
#include "envs/env_utils_python.h"
#include "envs/bsm_env.h"

namespace py = pybind11;

class ValuePredictor : public BSM_ENV {
public:
    ValuePredictor(std::string filename, int instance_number, int w, double min_fr);

    ValuePredictor(clpState* s0, int w, double timelimit=99999.9, std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now());
    
    void expand(const std::vector<std::vector<int>>& selected_blocks);

    void prune(const std::vector<int>& selected_states);

    std::vector<float> get_block_data() { return block_data; }

    std::vector<std::vector<float>> get_action_data_batch() { return action_data; }
    std::vector<std::vector<float>> get_pblock_data_batch() { return placed_data; }
    std::vector<std::vector<float>> get_space_data_batch()  { return space_data;  }

    std::vector<std::vector<float>> get_succ_action_data_batch() { return succ_action_data; }
    std::vector<std::vector<float>> get_succ_pblock_data_batch() { return succ_placed_data; }
    std::vector<std::vector<float>> get_succ_space_data_batch()  { return succ_space_data;  }

    bool is_finished();

private:
    std::vector<clp::clpState*> current_states;
    std::vector<clp::clpState*> succ_states;

    std::vector<std::vector<float>> action_data;
    std::vector<std::vector<float>> placed_data;
    std::vector<std::vector<float>> space_data;

    std::vector<std::vector<float>> succ_action_data;
    std::vector<std::vector<float>> succ_placed_data;
    std::vector<std::vector<float>> succ_space_data;

    bool completed = false;

    void update();
    void update_succ();
};

#endif