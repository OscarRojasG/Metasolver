#ifndef TENSOR_ENCODER_H
#define TENSOR_ENCODER_H

#include <vector>
#include <list>
#include <map>
#include <algorithm>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "clpState.h"
#include "VCS_Function.h"

namespace py = pybind11;

namespace TensorEncoder {

    inline void populate_blocks(clpState* state, int max_blocks, float* ptr, std::map<int, int>* out_id_to_idx) {
        std::fill(ptr, ptr + (max_blocks * 5), -1.0f);

        float cL = static_cast<float>(state->cont->getL());
        float cW = static_cast<float>(state->cont->getW());
        float cH = static_cast<float>(state->cont->getH());
        float md = std::max({cL, std::max(cW, cH)});

        int count = 0;
        if (out_id_to_idx) out_id_to_idx->clear(); 

        for (const auto* block : state->valid_blocks) {
            if (count >= max_blocks) break;

            float l = static_cast<float>(block->getL()) / md;
            float w = static_cast<float>(block->getW()) / md;
            float h = static_cast<float>(block->getH()) / md;
            
            int offset = count * 5;
            ptr[offset + 0] = l;
            ptr[offset + 1] = w;
            ptr[offset + 2] = h;
            ptr[offset + 3] = l * w * h; 
            ptr[offset + 4] = 1.0f / static_cast<float>(block->n_boxes);

            if (out_id_to_idx) (*out_id_to_idx)[block->id] = count;
            count++;
        }
    }
    
    inline void populate_actions(clpState* state, VCS_Function* vcs, const std::map<int, int>& id_to_idx, 
                                 int max_actions, int* ptr_blocks, float* ptr_features) {
        // 1. Llenar con padding
        std::fill(ptr_blocks, ptr_blocks + max_actions, -1);
        std::fill(ptr_features, ptr_features + (max_actions * 2), -1.0f);

        std::list<Action*> actions;
        state->get_actions(actions);
        std::multimap<double, Action*> ranked_actions;

        for (auto a : actions) {
            double eval = vcs->eval_action(*state, *a);
            if (eval > 0 && (ranked_actions.size() < max_actions || ranked_actions.begin()->first < eval)) {
                ranked_actions.insert({eval, a});
                if (ranked_actions.size() > max_actions) {
                    delete ranked_actions.begin()->second;
                    ranked_actions.erase(ranked_actions.begin());
                }
            } else { delete a; }
        }

        int count = 0;
        for (auto it = ranked_actions.rbegin(); it != ranked_actions.rend(); ++it) {
            if (count >= max_actions) { delete it->second; continue; }
            
            auto ca = static_cast<clp::clpAction*>(it->second);
            
            ptr_blocks[count] = id_to_idx.at(ca->block.id);
            ptr_features[(count * 2) + 0] = std::max(0.0f, static_cast<float>(ca->metrics.loss));
            ptr_features[(count * 2) + 1] = static_cast<float>(ca->metrics.cs);
            
            delete it->second;
            count++;
        }
    }

    inline void populate_placed(clpState* state, int max_pblocks, float* ptr_placed) {
        // 1. Llenar con padding
        std::fill(ptr_placed, ptr_placed + (max_pblocks * 6), -1.0f);

        if (state->cont->spaces->size() == 0) return;

        double cL = (double)state->cont->getL();
        double cW = (double)state->cont->getW();
        double cH = (double)state->cont->getH();
        double md = std::max({cL, std::max(cW, cH)});

        clp::Space current_space = state->cont->spaces->top();
        const bool* anchors = current_space.get_anchor();
        const Vector3 corner = current_space.get_corner();
        
        double sx = anchors[0] ? corner.getX() * -1 + cL : corner.getX();
        double sy = anchors[1] ? corner.getY() * -1 + cW : corner.getY();
        double sz = anchors[2] ? corner.getZ() * -1 + cH : corner.getZ();

        std::list<const Action*> path = state->get_path();
        int count = 0;
        
        for (const Action* a : path) {
            if (count >= max_pblocks) break;

            auto ca = static_cast<const clp::clpAction*>(a);
            Vector3 coords = ca->space.get_location(ca->block);

            double bx1 = anchors[0] ? (coords.getX() + ca->block.getL()) * -1 + cL : coords.getX();
            double by1 = anchors[1] ? (coords.getY() + ca->block.getW()) * -1 + cW : coords.getY();
            double bz1 = anchors[2] ? (coords.getZ() + ca->block.getH()) * -1 + cH : coords.getZ();

            int offset = count * 6;
            ptr_placed[offset + 0] = static_cast<float>((bx1 - sx) / md);
            ptr_placed[offset + 1] = static_cast<float>((by1 - sy) / md);
            ptr_placed[offset + 2] = static_cast<float>((bz1 - sz) / md);
            ptr_placed[offset + 3] = static_cast<float>(((bx1 + ca->block.getL()) - sx) / md);
            ptr_placed[offset + 4] = static_cast<float>(((by1 + ca->block.getW()) - sy) / md);
            ptr_placed[offset + 5] = static_cast<float>(((bz1 + ca->block.getH()) - sz) / md);

            count++;
        }
    }

    // ========================================================================
    // 1. CODIFICADOR DE BLOQUES (Equivalente a enc_2_vec)
    // ========================================================================
    inline py::array_t<float> encode_blocks(clpState* state, int max_blocks, std::map<int, int>& out_id_to_idx) {
        py::array_t<float> block_features({max_blocks, 5});
        float* ptr = static_cast<float*>(block_features.request().ptr);
        std::fill(ptr, ptr + (max_blocks * 5), -1.0f);

        float cL = static_cast<float>(state->cont->getL());
        float cW = static_cast<float>(state->cont->getW());
        float cH = static_cast<float>(state->cont->getH());
        float md = std::max({cL, std::max(cW, cH)});

        int count = 0;
        out_id_to_idx.clear(); // Asegurar que el mapa esté limpio

        for (const auto* block : state->valid_blocks) {
            if (count >= max_blocks) break;

            float l = static_cast<float>(block->getL()) / md;
            float w = static_cast<float>(block->getW()) / md;
            float h = static_cast<float>(block->getH()) / md;
            float vol = l * w * h; 
            float inv_n = 1.0f / static_cast<float>(block->n_boxes);

            int offset = count * 5;
            ptr[offset + 0] = l;
            ptr[offset + 1] = w;
            ptr[offset + 2] = h;
            ptr[offset + 3] = vol;
            ptr[offset + 4] = inv_n;

            out_id_to_idx[block->id] = count;
            count++;
        }
        return block_features;
    }

    // ========================================================================
    // 2. CODIFICADOR DE ACCIONES (Rama izquierda de dec_2_vec)
    // ========================================================================
    inline std::pair<py::array_t<int>, py::array_t<float>> encode_actions(
        clpState* state, VCS_Function* vcs, const std::map<int, int>& id_to_idx, int max_actions) {
        
        py::array_t<int> action_blocks({max_actions});
        py::array_t<float> action_features({max_actions, 2});

        int* ptr_blocks = static_cast<int*>(action_blocks.request().ptr);
        float* ptr_features = static_cast<float*>(action_features.request().ptr);

        std::fill(ptr_blocks, ptr_blocks + max_actions, -1);
        std::fill(ptr_features, ptr_features + (max_actions * 2), -1.0f);

        std::list<Action*> actions;
        state->get_actions(actions);
        std::multimap<double, Action*> ranked_actions;

        for (auto a : actions) {
            double eval = vcs->eval_action(*state, *a);
            if (eval > 0 && (ranked_actions.size() < max_actions || ranked_actions.begin()->first < eval)) {
                ranked_actions.insert({eval, a});
                if (ranked_actions.size() > max_actions) {
                    delete ranked_actions.begin()->second;
                    ranked_actions.erase(ranked_actions.begin());
                }
            } else { 
                delete a;
            }
        }

        int count = 0;
        for (auto it = ranked_actions.rbegin(); it != ranked_actions.rend(); ++it) {
            if (count >= max_actions) {
                delete it->second;
                continue;
            }
            auto ca = static_cast<clp::clpAction*>(it->second);
            
            ptr_blocks[count] = id_to_idx.at(ca->block.id);
            
            float loss = static_cast<float>(ca->metrics.loss);
            float cs = static_cast<float>(ca->metrics.cs);

            ptr_features[(count * 2) + 0] = std::max(0.0f, loss);
            ptr_features[(count * 2) + 1] = cs;
            
            delete it->second;
            count++;
        }

        return {action_blocks, action_features};
    }

    // ========================================================================
    // 3. CODIFICADOR DE GEOMETRÍA (Rama derecha de dec_2_vec)
    // ========================================================================
    inline py::array_t<float> encode_placed(clpState* state, int max_pblocks) {
        py::array_t<float> placed_features({max_pblocks, 6});
        float* ptr = static_cast<float*>(placed_features.request().ptr);
        std::fill(ptr, ptr + (max_pblocks * 6), -1.0f);

        if (state->cont->spaces->size() == 0) return placed_features;

        double cL = (double)state->cont->getL();
        double cW = (double)state->cont->getW();
        double cH = (double)state->cont->getH();
        double md = std::max({cL, std::max(cW, cH)});

        clp::Space current_space = state->cont->spaces->top();
        const bool* anchors = current_space.get_anchor();
        const Vector3 corner = current_space.get_corner();
        
        double sx = anchors[0] ? corner.getX() * -1 + cL : corner.getX();
        double sy = anchors[1] ? corner.getY() * -1 + cW : corner.getY();
        double sz = anchors[2] ? corner.getZ() * -1 + cH : corner.getZ();

        std::list<const Action*> path = state->get_path();
        int count = 0;
        
        for (const Action* a : path) {
            if (count >= max_pblocks) break;

            auto ca = static_cast<const clp::clpAction*>(a);
            Vector3 coords = ca->space.get_location(ca->block);

            double bx1 = anchors[0] ? (coords.getX() + ca->block.getL()) * -1 + cL : coords.getX();
            double by1 = anchors[1] ? (coords.getY() + ca->block.getW()) * -1 + cW : coords.getY();
            double bz1 = anchors[2] ? (coords.getZ() + ca->block.getH()) * -1 + cH : coords.getZ();

            float rel_x1 = static_cast<float>((bx1 - sx) / md);
            float rel_y1 = static_cast<float>((by1 - sy) / md);
            float rel_z1 = static_cast<float>((bz1 - sz) / md);
            
            float rel_x2 = static_cast<float>(((bx1 + ca->block.getL()) - sx) / md);
            float rel_y2 = static_cast<float>(((by1 + ca->block.getW()) - sy) / md);
            float rel_z2 = static_cast<float>(((bz1 + ca->block.getH()) - sz) / md);

            int offset = count * 6;
            ptr[offset + 0] = rel_x1;
            ptr[offset + 1] = rel_y1;
            ptr[offset + 2] = rel_z1;
            ptr[offset + 3] = rel_x2;
            ptr[offset + 4] = rel_y2;
            ptr[offset + 5] = rel_z2;

            count++;
        }
        return placed_features;
    }

    // 2.1 Encoder: Single State
    inline py::tuple get_enc_data(clpState* state, int max_blocks, std::map<int, int>& out_id_to_idx) {
        py::array_t<float> block_features({max_blocks, 5});
        populate_blocks(state, max_blocks, static_cast<float*>(block_features.request().ptr), &out_id_to_idx);
        return py::make_tuple(block_features);
    }

    // 2.2 Decoder: Single State
    inline py::tuple get_dec_data(clpState* state, VCS_Function* vcs, const std::map<int, int>& id_to_idx, 
                                  int max_actions, int max_pblocks) {
        py::array_t<int> action_blocks({max_actions});
        py::array_t<float> action_features({max_actions, 2});
        py::array_t<float> placed_features({max_pblocks, 6});

        populate_actions(state, vcs, id_to_idx, max_actions, 
                         static_cast<int*>(action_blocks.request().ptr), 
                         static_cast<float*>(action_features.request().ptr));
                         
        populate_placed(state, max_pblocks, static_cast<float*>(placed_features.request().ptr));

        return py::make_tuple(action_blocks, action_features, placed_features);
    }

    // 2.4 Decoder: Batch
    inline py::tuple get_dec_data_batch(const std::vector<clpState*>& states, VCS_Function* vcs, 
                                        const std::map<int, int>& id_to_idx, int max_actions, int max_pblocks) {
        int B = states.size();
        
        py::array_t<int> arr_blocks({B, max_actions});
        py::array_t<float> arr_features({B, max_actions, 2});
        py::array_t<float> arr_placed({B, max_pblocks, 6});

        int* ptr_b = static_cast<int*>(arr_blocks.request().ptr);
        float* ptr_f = static_cast<float*>(arr_features.request().ptr);
        float* ptr_p = static_cast<float*>(arr_placed.request().ptr);

        for (int b = 0; b < B; ++b) {
            populate_actions(states[b], vcs, id_to_idx, max_actions, 
                             ptr_b + (b * max_actions), 
                             ptr_f + (b * max_actions * 2));
                             
            populate_placed(states[b], max_pblocks, 
                            ptr_p + (b * max_pblocks * 6));
        }

        return py::make_tuple(arr_blocks, arr_features, arr_placed);
    }

} // namespace TensorEncoder

#endif