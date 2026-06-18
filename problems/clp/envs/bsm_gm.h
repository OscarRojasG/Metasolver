#ifndef BSM_GM_H
#define BSM_GM_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <map>
#include "clpState.h"
#include "BSG.h"
#include "Greedy.h"
#include "VCS_Function.h"
#include "env_utils.h"
#include "envs/env.h"
#include "data/tensor_encoder.h"

class BSM_GM : public ENV {
public:
    int w;
    int max_blocks, max_actions, max_pblocks; // Nuevos límites de tensores

    BSM_GM(std::string filename, int instance_number, int w, int max_blocks, int max_actions, int max_pblocks, double min_fr=1);
    BSM_GM(clpState* s0, int w, int max_blocks, int max_actions, int max_pblocks, double timelimit = 99999.9, std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now());

    virtual ~BSM_GM();

    void expand(std::vector<std::vector<int>> selected_indexes_lists);
    void greedy_step(std::vector<int> selected_indexes);

    // Nuevas funciones de tensores unificadas
    py::tuple get_enc_data();
    py::tuple get_dec_data_batch_expand();
    py::tuple get_dec_data_batch_greedy();

    bool is_finished();
    bool is_greedy_finished();

private:
    std::vector<clpState *> current_states;
    std::vector<std::pair<clpState *, clpState *>> succ_states;
    std::map<double, std::pair<State *, State *>> evals;
    
    // Mapeo necesario para TensorEncoder y las transiciones
    std::map<int, int> id_to_idx;
    int current_depth_expand;
    int current_depth_greedy;

    bool completed = false;
};

#endif