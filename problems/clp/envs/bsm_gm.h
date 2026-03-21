#ifndef BSM_GM_H
#define BSM_GM_H

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

class BSM_GM : public BSM_ENV {
public:
    BSM_GM(std::string filename, int instance_number, int w, double min_fr=1);

    BSM_GM(clpState* s0, int w, double timelimit=99999.9, std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now());

    void transition_bsm(std::vector<std::vector<int>> selected_indexes_lists);

    void transition_greedy(std::vector<int> selected_indexes);

    py::dict get_batch_dict_bsm();

    py::dict get_batch_dict_greedy();

    bool is_greedy_finished();

    bool is_bsm_finished();

private:
    std::map<double, std::pair<State *, State *>> state_actions;

    py::array_t<int32_t> action_blocks_gr;
    py::array_t<float> action_features_gr;
    py::array_t<int32_t> placed_blocks_gr;
    py::array_t<float> placed_features_gr;
    py::array_t<float> space_features_gr;

    std::vector<clpState*> get_greedy_nodes_vec();

    void update_batches_greedy();
};

#endif