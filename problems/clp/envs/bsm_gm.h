#ifndef BSM_GM_H
#define BSM_GM_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "clpState.h"
#include "BSG.h"
#include "Greedy.h"
#include "VCS_Function.h"
#include "env_utils.h"
#include "envs/env.h"

class BSM_GM : public ENV {
public:
    int w;

    BSM_GM(std::string filename, int instance_number, int w, double min_fr=1);

    BSM_GM(clpState* s0, int w, double timelimit = 99999.9, std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now());

    virtual ~BSM_GM();

    void expand(std::vector<std::vector<int>> selected_indexes_lists);
    void greedy_step(std::vector<int> selected_indexes);

    std::vector<std::vector<float>> get_action_data_batch_expand() { return action_data_expand; }
    std::vector<std::vector<float>> get_pblock_data_batch_expand() { return placed_data_expand; }
    std::vector<std::vector<float>> get_space_data_batch_expand()  { return space_data_expand;  }

    std::vector<std::vector<float>> get_action_data_batch_greedy() { return action_data_greedy; }
    std::vector<std::vector<float>> get_pblock_data_batch_greedy() { return placed_data_greedy; }
    std::vector<std::vector<float>> get_space_data_batch_greedy()  { return space_data_greedy;  }    

    bool is_finished();
    bool is_greedy_finished();

private:
    std::vector<clpState *> current_states;
    std::vector<std::pair<clpState *, clpState *>> succ_states;
    std::map<double, std::pair<State *, State *>> evals;

    std::vector<std::vector<float>> action_data_expand;
    std::vector<std::vector<float>> placed_data_expand;
    std::vector<std::vector<float>> space_data_expand;

    std::vector<std::vector<float>> action_data_greedy;
    std::vector<std::vector<float>> placed_data_greedy;
    std::vector<std::vector<float>> space_data_greedy;

    bool completed = false;

    void update_batch_expand();
    void update_batch_greedy();
};

#endif