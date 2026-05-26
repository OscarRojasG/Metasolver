#ifndef ENV_H
#define ENV_H

#include "clpState.h"
#include "BSG.h"
#include "Greedy.h"
#include "VCS_Function.h"
#include <chrono>
#include "data/block_data.h"
#include "data/batch_data.h"

class ENV {
public:
    double timelimit;
    double final_time = 0;

    double best_volume = 0;
    clpState* best_state = NULL;

    clpState* s0;

    ENV(clpState* s0, double timelimit=99999.9, std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now()) {
        this->s0 = s0;
        this->timelimit = timelimit;
        this->start_time = start_time;

        // VCS y parámetros
        double alpha = 4.0, beta = 1.0, gamma = 0.2, delta = 1.0, p = 0.04;
        double r = 0.0;
        vcs = new VCS_Function(s0->nb_left_boxes, *s0->cont, alpha, beta, gamma, p, delta, 0.0, r);

        block_data = new BlockData(s0);
    }

    std::vector<float> get_block_data() {
        return block_data->get_block_features();
    }

    double get_path_length() {
        return best_state->get_path().size();
    }

    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<float>>> get_path() {
        clpState* s = dynamic_cast<clpState*> (s0->clone());
        list<const Action*>& actions = best_state->get_path();

        std::vector<clpState*> path_nodes;
        path_nodes.push_back(s0);

        for(auto action:actions) {
			const clpAction* a = dynamic_cast<const clpAction*> (action);
            s->transition(*a);

            clpState* s_copy = dynamic_cast<clpState*> (s->clone());
            path_nodes.push_back(s_copy);
        }

        BatchData batch_data(*block_data, path_nodes, vcs, 1);
        std::vector<std::vector<float>> placed_data = batch_data.get_batch_placed_features();
        std::vector<std::vector<float>> space_data = batch_data.get_batch_space_features();

        return { placed_data, space_data };
    }

protected:
    std::chrono::steady_clock::time_point start_time;

    BlockData *block_data;
    VCS_Function *vcs;
    
    double get_elapsed_time() {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time).count();
    }
};

#endif