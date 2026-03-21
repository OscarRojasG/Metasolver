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
#include "envs/bsm_env.h"

namespace py = pybind11;

class GreedyModel : public ENV {
public:
    int w;
    double volume;

    GreedyModel(std::string filename, int instance_number, int w, double min_fr=1);

    GreedyModel(clpState* s0, int w);

    py::array_t<float> get_block_features();

    void transition(int selected_index);

    py::dict get_dict();

    bool is_finished();

private:
    clpState* current_node;
    VCS_Function *vcs;

    map<int, int> block_id_to_index;
    std::vector<int> block_index_to_id;

    py::array_t<float> block_features;
    py::array_t<int32_t> action_blocks;
    py::array_t<float> action_features;
    py::array_t<int32_t> placed_blocks;
    py::array_t<float> placed_features;
    py::array_t<float> space_features;

    bool completed = false;

    void update();
};

#endif