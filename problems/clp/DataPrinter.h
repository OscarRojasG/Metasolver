#include <list>
#include "VCS_Function.h"
#include "envs/env_utils.h"
#include "clpState.h"

using namespace std;
using namespace metasolver;

class DataPrinter {
private:
    clpState* state;
    map<int, int> block_id_to_index;
    vector<int> block_index_to_id;

    VCS_Function *vcs;
    int w;

    int num_blocks;
    int num_actions;
    int num_placed;

    vector<float> block_features;
    vector<int> action_blocks;
    vector<float> action_features;
    vector<int> placed_blocks;
    vector<float> placed_features;
    vector<float> space_features;

    void get_blocks() {
        num_blocks = static_cast<int>(state->valid_blocks.size());
        
        block_features.assign(num_blocks * EnvUtils::N_BLOCK_FEATURES, 0.0f);
        
        EnvUtils::get_blocks_data(state, block_features.data(), block_id_to_index, block_index_to_id);
    }
    
    void get_actions() {
        list<Action *> actions;
        state->get_actions(actions);
        num_actions = min(w * w, static_cast<int>(actions.size()));

        for (auto a : actions) delete a;

        action_blocks.assign(num_actions, 0);
        action_features.assign(num_actions * EnvUtils::N_ACTION_FEATURES, 0.0f);

        EnvUtils::get_actions_data(state, vcs, w, block_id_to_index, action_blocks.data(), action_features.data(), num_actions);
    }

    void get_placed_data() {
        num_placed = static_cast<int>(state->get_path().size());

        placed_blocks.assign(num_placed, 0);
        placed_features.assign(num_placed * EnvUtils::N_PLACED_FEATURES, 0.0f);

        EnvUtils::get_placed_data(state, block_id_to_index, placed_blocks.data(), placed_features.data(), num_placed);
    }

    void get_space_features() {
        space_features.assign(EnvUtils::N_SPACE_FEATURES, 0.0f);
        
        EnvUtils::get_space_features(state, space_features.data());
    }

public:
    DataPrinter(clpState* s, VCS_Function* vcs, int w) {
        state = s;
        this->vcs = vcs;
        this->w = w;

        get_blocks();
    }

    void print_blocks() {
        for (int i = 0; i < num_blocks; i++) {
            for (int j = 0; j < EnvUtils::N_BLOCK_FEATURES; j++) {
                cout << block_features[i * EnvUtils::N_BLOCK_FEATURES + j] << " ";
            }
            cout << endl;
        }
    }

    void print_actions() {
        get_actions();

        for (int i = 0; i < num_actions; i++) {
            cout << action_blocks[i] << " ";
            for (int j = 0; j < EnvUtils::N_ACTION_FEATURES; j++) {
                cout << action_features[i * (EnvUtils::N_ACTION_FEATURES) + j] << " ";
            }
            cout << endl;
        }
    }

    void print_placed() {
        get_placed_data();

        for (int i = 0; i < num_placed; i++) {
            cout << placed_blocks[i] << " ";
            for (int j = 0; j < EnvUtils::N_PLACED_FEATURES; j++) {
                cout << placed_features[i * EnvUtils::N_PLACED_FEATURES + j] << " ";
            }
            cout << endl;
        }
    }

    void print_space() {
        get_space_features();

        for (int i = 0; i < EnvUtils::N_SPACE_FEATURES; i++) {
            cout << space_features[i] << " ";
        }
        cout << endl;
    }

    void print_volume() {
        cout << state->cont->getOccupiedVolume() / state->cont->getVolume() << endl;
    }

    void print_block_index(int id) {
        cout << block_id_to_index[id] << endl;
    }
};