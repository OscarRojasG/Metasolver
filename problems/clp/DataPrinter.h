#include <iostream>
#include <vector>
#include <list>
#include "VCS_Function.h"
#include "clpState.h"
#include "data/block_data.h"
#include "data/state_data.h"

using namespace std;
using namespace metasolver;

class DataPrinter {

public:
    static void printBlocks(const BlockData& block_data) {
        const std::vector<float>& features = block_data.get_block_features();

        for (size_t i = 0; i < features.size(); i += BlockData::N_BLOCK_FEATURES) {
            for (int j = 0; j < BlockData::N_BLOCK_FEATURES; ++j) {
                cout << features[i + j] << (j == BlockData::N_BLOCK_FEATURES - 1 ? "" : " ");
            }
            cout << "\n";
        }
    }

    static void printActions(const StateData& state_data) {
        const std::vector<float>& features = state_data.get_action_features();

        for (size_t i = 0; i < features.size(); i += StateData::N_ACTION_FEATURES) {
            for (int j = 0; j < StateData::N_ACTION_FEATURES; ++j) {
                cout << features[i + j] << (j == StateData::N_ACTION_FEATURES - 1 ? "" : " ");
            }
            cout << "\n";
        }
    }

    static void printGreedy(StateData& state_data) {
        std::vector<float> values = state_data.get_greedy_values();
        
        for (size_t i = 0; i < values.size(); i++) {
            cout << values[i] << "\n";
        }
    }

    static void printPlaced(const StateData& state_data) {
        const std::vector<float>& features = state_data.get_placed_features();

        for (size_t i = 0; i < features.size(); i += StateData::N_PLACED_FEATURES) {
            for (int j = 0; j < StateData::N_PLACED_FEATURES; ++j) {
                cout << features[i + j] << (j == StateData::N_PLACED_FEATURES - 1 ? "" : " ");
            }
            cout << "\n";
        }
    }

    static void printSpace(const StateData& state_data) {
        const std::vector<float>& features = state_data.get_space_features();

        if (!features.empty()) {
            for (size_t j = 0; j < features.size(); ++j) {
                cout << features[j] << (j == features.size() - 1 ? "" : " ");
            }
            cout << "\n";
        }
    }

    static void printVolume(const StateData& state_data) {
        cout << state_data.get_volume_ratio() << "\n";
    }

    static void printBlockIndex(const BlockData& block_data, int block_id) {
        cout << block_data.get_block_id_to_index().at(block_id) << "\n";
    }
};