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
#include "data/state_data.h"
#include "envs/env.h"

namespace py = pybind11;

class GreedyModel : public ENV {
public:
    int w;
    double volume;

    GreedyModel(std::string filename, int instance_number, int w, double min_fr=1);

    GreedyModel(clpState* s0, int w);

    void transition(int selected_index);

    const std::vector<float> get_action_data() const { return action_data; }

    const std::vector<float> get_pblock_data() const { return placed_data; }

    const std::vector<float> get_space_data() const { return space_data; }

    const bool is_finished() const { return completed; }

private:
    clpState* current_node;

    std::vector<float> action_data;
    std::vector<float> placed_data;
    std::vector<float> space_data;

    bool completed = false;

    void update();
};

#endif