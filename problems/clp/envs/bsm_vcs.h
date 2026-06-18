#ifndef BSM_VCS_H
#define BSM_VCS_H

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

class BSM_VCS : public ENV {
public:
    int w;
    int max_blocks, max_actions, max_pblocks;

    BSM_VCS(std::string filename, int instance_number, int w, int max_blocks, int max_actions, int max_pblocks, double min_fr=1);
    BSM_VCS(clpState* s0, int w, int max_blocks, int max_actions, int max_pblocks, double timelimit = 99999.9);
    
    virtual ~BSM_VCS();

    void transition(const std::vector<std::vector<int>>& selected_indexes_lists);
    bool is_finished();

    // Nueva interfaz directa
    py::tuple get_enc_data();
    py::tuple get_dec_data_batch();

private:
    std::vector<clpState *> current_states;
    std::vector<std::pair<clpState *, clpState *>> succ_states;
    SearchStrategy *gr;
    bool completed = false;
    int current_depth;
    
    // Mapeo necesario para la transición
    std::map<int, int> id_to_idx;

    std::map<double, std::pair<State *, State *>> eval_succ_states();
};
#endif