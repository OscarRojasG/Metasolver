#ifndef BLOCK_DATA_H
#define BLOCK_DATA_H

#include <vector>
#include <list>
#include <map>
#include "clpState.h"

class BlockData {
    private:
        std::vector<float> block_features;
        std::vector<float> box_features;
        std::map<int, int> block_id_to_index;
        std::vector<int> block_index_to_id;

        std::map<int, std::vector<std::pair<int, int>>> block_to_boxes_info;
    
    public:
        static const int N_BLOCK_FEATURES = 4;
        static const int N_BOX_FEATURES = 4;

        BlockData(clpState* s0) {
            block_features.reserve(s0->valid_blocks.size() * N_BLOCK_FEATURES);

            double cL = (double)s0->cont->getL();
            double cW = (double)s0->cont->getW();
            double cH = (double)s0->cont->getH();
            double md = max({cL, cW, cH});

            // 1. Mapa auxiliar para mapear BoxShape a índice correlativo
            std::map<const clp::BoxShape*, int> shape_to_index;

            // 2. Procesamos cajas para llenar box_features y mapear índices
            int box_index = 0;
            std::map<const clp::BoxShape *, int, clp::BoxShapeComparator> boxes = s0->nb_left_boxes;

            for (const auto& [shape, count] : boxes) {
                box_features.push_back((float)shape->getL(BoxShape::Orientation::LWH) / md);
                box_features.push_back((float)shape->getW(BoxShape::Orientation::LWH) / md);
                box_features.push_back((float)shape->getH(BoxShape::Orientation::LWH) / md);
                box_features.push_back((float)count);
                
                shape_to_index[shape] = box_index;
                box_index++;
            }

            // 3. Procesamos bloques, llenamos block_features y el nuevo mapa
            int block_count = 0;
            for (const Block* block : s0->valid_blocks) {          
                block_features.push_back((float)block->getL() / md);
                block_features.push_back((float)block->getW() / md);
                block_features.push_back((float)block->getH() / md);
                block_features.push_back((float)block->n_boxes);

                // Creamos el vector temporal para este bloque
                std::vector<std::pair<int, int>> boxes_in_block;

                for (const auto& [shape, count] : block->nb_boxes) {
                    int correlative_id = shape_to_index[shape];

                    // Guardamos en la lista del bloque
                    boxes_in_block.push_back({correlative_id, count});
                }
                
                // Asignamos la lista al mapa del bloque
                block_to_boxes_info[block->id] = boxes_in_block;
            
                block_id_to_index[block->id] = block_count;
                block_index_to_id.push_back(block->id);
                block_count++;
            }
        }
    
        const std::vector<float>& get_block_features() const { return block_features; }
        const std::map<int, std::vector<std::pair<int, int>>>& get_block_to_boxes_info() const { 
            return block_to_boxes_info; 
        }
        const std::vector<float>& get_box_features() const { return box_features; }
        const std::map<int, int>& get_block_id_to_index() const { return block_id_to_index; }
        const std::vector<int>& get_block_index_to_id() const { return block_index_to_id; }
    };

#endif