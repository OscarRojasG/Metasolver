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
#include "envs/env.h"

namespace py = pybind11;

class ValuePredictor : public ENV {
public:
    ValuePredictor(std::string filename, int instance_number, int w, double min_fr=1);

    ValuePredictor(clpState* s0, int w);
    
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

    int w;

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