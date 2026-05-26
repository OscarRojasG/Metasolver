#ifndef BLOCK_DATA_H
#define BLOCK_DATA_H

#include <vector>
#include <list>
#include <map>
#include "clpState.h"

class BlockData {
    private:
        std::vector<float> block_features;
        std::map<int, int> block_id_to_index;
        std::vector<int> block_index_to_id;
    
    public:
        static const int N_BLOCK_FEATURES = 4;

        BlockData(clpState* s0) {
            block_features.reserve(s0->valid_blocks.size() * N_BLOCK_FEATURES);
    
            int count = 0;
            for (const Block* block : s0->valid_blocks) {          
                block_features.push_back((float)block->getL() / (float)s0->cont->getL());
                block_features.push_back((float)block->getW() / (float)s0->cont->getW());
                block_features.push_back((float)block->getH() / (float)s0->cont->getH());
                block_features.push_back((float)block->n_boxes);
    
                block_id_to_index[block->id] = count;
                block_index_to_id.push_back(block->id);
                count++;
            }
        }
    
        const std::vector<float>& get_block_features() const { return block_features; }
        const std::map<int, int>& get_block_id_to_index() const { return block_id_to_index; }
        const std::vector<int>& get_block_index_to_id() const { return block_index_to_id; }
    };

#endif