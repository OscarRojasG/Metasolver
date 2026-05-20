#include <list>
#include "VCS_Function.h"
#include "envs/env_utils.h"
#include "clpState.h"

using namespace std;
using namespace metasolver;

class DataPrinter {
private:
    clpState* state;
    VCS_Function* vcs;
    int w;
    
    // Estructuras para el mapeo de IDs requeridas por las funciones de EnvUtils
    std::map<int, int> block_id_to_index;
    std::vector<int> block_index_to_id;

public:
    DataPrinter(clpState* state, VCS_Function* vcs, int w) : state(state), vcs(vcs), w(w) {}

    void printBlocks() {
        std::vector<float> features;
        // get_blocks_data llena el vector y además inicializa las estructuras de mapeo de IDs
        EnvUtils::get_blocks_data(state, features, block_id_to_index, block_index_to_id);

        for (size_t i = 0; i < features.size(); i += EnvUtils::N_BLOCK_FEATURES) {
            for (int j = 0; j < EnvUtils::N_BLOCK_FEATURES; ++j) {
                cout << features[i + j] << (j == EnvUtils::N_BLOCK_FEATURES - 1 ? "" : " ");
            }
            cout << "\n";
        }
    }

    void printActions() {
        std::vector<float> features;
        EnvUtils::get_actions_data(state, vcs, block_id_to_index, features, w*w);

        for (size_t i = 0; i < features.size(); i += EnvUtils::N_ACTION_FEATURES) {
            for (int j = 0; j < EnvUtils::N_ACTION_FEATURES; ++j) {
                cout << features[i + j] << (j == EnvUtils::N_ACTION_FEATURES - 1 ? "" : " ");
            }
            cout << "\n";
        }
    }

    void printGreedy() {
        double max_greedy_value = -1e9; 
        SearchStrategy* gr = new Greedy(vcs);

        clpState* s_copy = dynamic_cast<clpState*>(state->clone()); 
        double value = gr->run(*s_copy);
        cout << value << "\n";

        delete gr;
    }

    void printPlaced() {
        std::vector<float> features;
        EnvUtils::get_placed_data(state, block_id_to_index, features);

        for (size_t i = 0; i < features.size(); i += EnvUtils::N_PLACED_FEATURES) {
            for (int j = 0; j < EnvUtils::N_PLACED_FEATURES; ++j) {
                cout << features[i + j] << (j == EnvUtils::N_PLACED_FEATURES - 1 ? "" : " ");
            }
            cout << "\n";
        }
    }

    void printSpace() {
        std::vector<float> features;
        EnvUtils::get_space_data(state, features);

        for (int j = 0; j < EnvUtils::N_SPACE_FEATURES; ++j) {
            cout << features[j] << (j == EnvUtils::N_SPACE_FEATURES - 1 ? "" : " ");
        }
        cout << "\n";
    }

    void printVolume() {
        cout << EnvUtils::get_volume_ratio(state) << "\n";
    }

    void printBlockIndex(int block_id) {
        cout << block_id_to_index[block_id] << "\n";
    }
};