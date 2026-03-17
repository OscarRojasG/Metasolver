#ifndef ENV_UTILS_PYTHON_H
#define ENV_UTILS_PYTHON_H

#include <vector>
#include <list>
#include <map>
#include "env_utils.h"
#include <pybind11/numpy.h>

namespace py = pybind11;

class EnvUtilsPython {
public:
    static void get_blocks_data(clpState* s0, py::array_t<float>& block_features, std::map<int, int>& block_id_to_index, std::vector<int>& block_index_to_id) {
        size_t n = s0->valid_blocks.size();
        block_features = py::array_t<float>({ (long)n, 8L });
        EnvUtils::get_blocks_data(s0, block_features.mutable_data(), block_id_to_index, block_index_to_id);
    }

    static void get_actions_data_batch(const std::vector<clp::clpState*>& states, VCS_Function* vcs, int w, map<int, int>& block_id_to_index, py::array_t<int32_t>& action_blocks_batch, py::array_t<float>& action_features_batch) {
        size_t n = states.size();
        size_t limit = (size_t)(w * w);

        action_blocks_batch = py::array_t<int32_t>({ (long)n, (long)limit });
        action_features_batch = py::array_t<float>({ (long)n, (long)limit, 2L });

        int32_t* p_blocks = action_blocks_batch.mutable_data();
        float* p_features = action_features_batch.mutable_data();

        EnvUtils::get_actions_data_batch(states, vcs, w, block_id_to_index, p_blocks, p_features);
    }

    static void get_placed_data_batch(const std::vector<clp::clpState*>& states, map<int, int>& block_id_to_index, py::array_t<int32_t>& placed_blocks_batch, py::array_t<float>& placed_features_batch) {
        size_t n = states.size();
        size_t padding = 64;

        placed_blocks_batch = py::array_t<int32_t>({ (long)n, (long)padding });
        placed_features_batch = py::array_t<float>({ (long)n, (long)padding, 4L });

        int32_t* p_blocks = placed_blocks_batch.mutable_data();
        float* p_features = placed_features_batch.mutable_data();

        EnvUtils::get_placed_data_batch(states, block_id_to_index, p_blocks, p_features);
    }

    static void get_space_features_batch(const std::vector<clp::clpState*>& states, py::array_t<float>& space_features_batch) {
        size_t n = states.size();
        space_features_batch = py::array_t<float>({ (long)n, 6L });
        EnvUtils::get_space_features_batch(states, space_features_batch.mutable_data());
    }
};

#endif