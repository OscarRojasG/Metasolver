#ifndef ENV_UTILS2_PYTHON_H
#define ENV_UTILS2_PYTHON_H

#include <vector>
#include <map>
#include "env_utils.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class EnvUtilsPython {
public:
    // Retorna una lista con las features de los bloques disponibles (Tamaño: N x 5)
    static std::vector<float> get_blocks_data(clpState* s0, std::map<int, int>& block_id_to_index, std::vector<int>& block_index_to_id) {
        std::vector<float> out_features;
        EnvUtils::get_blocks_data(s0, out_features, block_id_to_index, block_index_to_id);
        return out_features;
    }

    // Retorna una lista de listas (Jagged/Ragged Array) con las mejores acciones encontradas por estado
    static std::vector<std::vector<float>> get_actions_data_batch(const std::vector<clpState*>& states, VCS_Function* vcs, std::map<int, int>& block_id_to_index, int num_actions) {
        // Llama directamente a la función batch que hicimos con vectores dinámicos
        return EnvUtils::get_actions_data_batch(states, vcs, block_id_to_index, num_actions);
    }

    // Retorna una lista de listas con los bloques ya colocados (path) de cada estado
    static std::vector<std::vector<float>> get_placed_data_batch(const std::vector<clpState*>& states, std::map<int, int>& block_id_to_index) {
        return EnvUtils::get_placed_data_batch(states, block_id_to_index);
    }

    // Retorna una lista de listas con las características de los espacios (EMS) de cada estado
    static std::vector<std::vector<float>> get_space_data_batch(const std::vector<clpState*>& states) {
        return EnvUtils::get_space_data_batch(states);
    }

    // --- Versiones individuales por si las necesitas en tu bucle de step ---
    
    static std::vector<float> get_actions_data(clpState* state, VCS_Function* vcs, std::map<int, int>& block_id_to_index, int num_actions) {
        std::vector<float> out_features;
        EnvUtils::get_actions_data(state, vcs, block_id_to_index, out_features, num_actions);
        return out_features;
    }

    static std::vector<float> get_placed_data(clpState* state, std::map<int, int>& block_id_to_index) {
        std::vector<float> out_features;
        EnvUtils::get_placed_data(state, block_id_to_index, out_features);
        return out_features;
    }

    static std::vector<float> get_space_data(clpState* state) {
        std::vector<float> out_features;
        EnvUtils::get_space_data(state, out_features);
        return out_features;
    }
};

#endif