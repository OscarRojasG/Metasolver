#ifndef BATCH_DATA_H
#define BATCH_DATA_H

#include <vector>
#include <list>
#include <map>
#include <utility>
#include "clpState.h"
#include "VCS_Function.h"
#include "Greedy.h"
#include "block_data.h"
#include "state_data.h"

class BatchData {
    private:
        std::vector<std::vector<float>> batch_action_features;
        std::vector<std::vector<float>> batch_placed_features;
        std::vector<std::vector<float>> batch_space_features;
        std::vector<double> batch_volume_ratios;
    
    public:
        BatchData(BlockData& block_data, const std::vector<clpState*>& states, VCS_Function* vcs, int num_actions) {
            batch_action_features.reserve(states.size());
            batch_placed_features.reserve(states.size());
            batch_space_features.reserve(states.size());
            batch_volume_ratios.reserve(states.size());
    
            for (clpState* state : states) {
                StateData state_data(block_data, state, vcs, num_actions);
    
                batch_action_features.push_back(std::move(const_cast<std::vector<float>&>(state_data.get_action_features())));
                batch_placed_features.push_back(std::move(const_cast<std::vector<float>&>(state_data.get_placed_features())));
                batch_space_features.push_back(std::move(const_cast<std::vector<float>&>(state_data.get_space_features())));
                batch_volume_ratios.push_back(state_data.get_volume_ratio());
            }
        }
    
        const std::vector<std::vector<float>>& get_batch_action_features() const { return batch_action_features; }
        const std::vector<std::vector<float>>& get_batch_placed_features() const { return batch_placed_features; }
        const std::vector<std::vector<float>>& get_batch_space_features() const { return batch_space_features; }
        const std::vector<double>& get_batch_volume_ratios() const { return batch_volume_ratios; }
    };
    
#endif