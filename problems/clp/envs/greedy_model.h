#ifndef GREEDY_MODEL_H
#define GREEDY_MODEL_H

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

class GreedyModel : public ENV {
public:
    int w;
    double volume;

    GreedyModel(std::string filename, int instance_number, int w, double min_fr=1);

    GreedyModel(clpState* s0, int w);

    std::vector<float> get_block_data();

    std::vector<float> get_action_data();

    std::vector<float> get_pblock_data();

    std::vector<float> get_space_data();

    void transition(int selected_index);

    bool is_finished();

private:
    clpState* current_node;
    VCS_Function *vcs;

    map<int, int> block_id_to_index;
    std::vector<int> block_index_to_id;

    std::vector<float> block_data;
    std::vector<float> action_data;
    std::vector<float> placed_data;
    std::vector<float> space_data;

    bool completed = false;

    void update();
};

#endif