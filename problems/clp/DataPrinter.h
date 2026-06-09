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
    static void printBoxes(const BlockData& block_data) {
        const std::vector<float>& features = block_data.get_box_features();

        for (size_t i = 0; i < features.size(); i += BlockData::N_BOX_FEATURES) {
            for (int j = 0; j < BlockData::N_BOX_FEATURES; ++j) {
                cout << features[i + j] << (j == BlockData::N_BOX_FEATURES - 1 ? "" : " ");
            }
            cout << "\n";
        }
    }

    static void printBlocks(const BlockData& block_data) {
        const std::vector<float>& features = block_data.get_block_features();
        const auto& boxes_map = block_data.get_block_to_boxes_info();
        const std::vector<int>& index_to_id = block_data.get_block_index_to_id();
    
        int block_idx = 0; 
    
        // Recorremos las features de 4 en 4 (N_BLOCK_FEATURES)
        for (size_t i = 0; i < features.size(); i += BlockData::N_BLOCK_FEATURES) {
            
            // 1. Imprimir las 4 características base
            for (int j = 0; j < BlockData::N_BLOCK_FEATURES; ++j) {
                cout << features[i + j] << " ";
            }
    
            // 2. Obtener el ID del bloque usando el índice actual
            int current_block_id = index_to_id[block_idx];
    
            // 3. Buscar las cajas en el mapa y imprimirlas
            auto it = boxes_map.find(current_block_id);
            if (it != boxes_map.end()) {
                for (const auto& pair : it->second) {
                    // pair.first es el ID correlativo de la caja, pair.second es la cantidad
                    cout << pair.first << " " << pair.second << " ";
                }
            }
    
            cout << "\n";
            block_idx++; // Pasamos al siguiente índice
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