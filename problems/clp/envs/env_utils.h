#ifndef ENV_UTILS_H
#define ENV_UTILS_H

#include <vector>
#include <list>
#include <map>
#include "clpState.h"
#include "VCS_Function.h"
#include "BlockMetrics.h"

class EnvUtils {
public:
    static const int N_BLOCK_FEATURES = 8;
    static const int N_ACTION_FEATURES = 2;
    static const int N_PLACED_FEATURES = 4;
    static const int N_SPACE_FEATURES = 6;

    static void get_blocks_data(clpState* s0, float* out_features, std::map<int, int>& block_id_to_index, std::vector<int>& block_index_to_id) {
        block_id_to_index.clear();
        block_index_to_id.clear();

        size_t num_blocks = s0->valid_blocks.size();
        
        block_index_to_id.reserve(num_blocks);

        size_t count = 0;
        for (const Block* block : s0->valid_blocks) {
            BlockMetrics bm(*block, *(s0->cont));
            
            size_t idx = count * 8;
            
            out_features[idx]     = (float)bm.getNormL();
            out_features[idx + 1] = (float)bm.getNormH();
            out_features[idx + 2] = (float)bm.getNormW();
            out_features[idx + 3] = (float)bm.getNormOccupiedVolumeCont();
            out_features[idx + 4] = (float)bm.getBoxesAmountReciprocal();
            out_features[idx + 5] = (float)bm.getNormL() * (float)bm.getNormW();
            out_features[idx + 6] = (float)bm.getNormW() * (float)bm.getNormH();
            out_features[idx + 7] = (float)bm.getNormH() * (float)bm.getNormL();

            block_id_to_index[block->id] = (int)count;
            block_index_to_id.push_back(block->id);
            
            count++;
        }
    }

    static void get_actions_data(clpState* state, VCS_Function* vcs, int w, map<int, int>& block_id_to_index, int* out_blocks, float* out_features, int num_actions) {
        std::list<Action*> actions;
        state->get_actions(actions);
        
        std::multimap<double, Action*> ranked_actions;

        for (auto a : actions) {
            double eval = vcs->eval_action(*state, *a);
            if (eval > 0 && (ranked_actions.size() < num_actions || ranked_actions.begin()->first < eval)) {
                ranked_actions.insert({eval, a});
                if (ranked_actions.size() > num_actions) {
                    delete ranked_actions.begin()->second;
                    ranked_actions.erase(ranked_actions.begin());
                }
            } else { delete a; }
        }

        std::fill(out_blocks, out_blocks + num_actions, -1);
        std::fill(out_features, out_features + (num_actions * 2), -1.0f);

        size_t count = 0;
        for (auto it = ranked_actions.rbegin(); it != ranked_actions.rend(); ++it) {
            clp::clpAction* ca = dynamic_cast<clp::clpAction*>(it->second);
            if (ca) {
                out_blocks[count] = block_id_to_index[ca->block.id];
                
                auto it_m = ca->metrics.begin();
                out_features[count * 2] = (float)(*it_m);
                it_m++;
                out_features[count * 2 + 1] = (float)(*it_m);
                
                count++;
            }
            delete it->second;
        }
    }

    static void get_placed_data(clpState* state, map<int, int>& block_id_to_index, int* out_blocks, float* out_features, int padding) {
        std::list<const Action*> path = state->get_path();
        
        std::fill(out_blocks, out_blocks + padding, -1);
        std::fill(out_features, out_features + (padding * 4), -1.0f);

        clp::Space space = state->cont->spaces->top();
        const bool* anchors = space.get_anchor();
        const Vector3 corner = space.get_corner();

        long sx = anchors[0] ? corner.getX() * -1 + state->cont->getL() : corner.getX();
        long sy = anchors[1] ? corner.getY() * -1 + state->cont->getW() : corner.getY();
        long sz = anchors[2] ? corner.getZ() * -1 + state->cont->getH() : corner.getZ();

        double cL = (double)state->cont->getL();
        double cW = (double)state->cont->getW();
        double cH = (double)state->cont->getH();

        size_t count = 0;
        for (const Action* a : path) {
            if (count >= (size_t)padding) break;

            auto ca = static_cast<const clp::clpAction*>(a);
            Vector3 coords = ca->space.get_location(ca->block);

            double bx = anchors[0] ? (coords.getX() + ca->block.getL()) * -1 + cL : coords.getX();
            double by = anchors[1] ? (coords.getY() + ca->block.getW()) * -1 + cW : coords.getY();
            double bz = anchors[2] ? (coords.getZ() + ca->block.getH()) * -1 + cH : coords.getZ();

            float contact = 1.0f;
            if ((bx + ca->block.getL() < sx) || (bx > sx + space.getL()) ||
                (by + ca->block.getW() < sy) || (by > sy + space.getW()) ||
                (bz + ca->block.getH() < sz) || (bz > sz + space.getH())) {
                contact = 0.0f;
            }

            out_blocks[count] = block_id_to_index[ca->block.id];
            
            size_t f_idx = count * 4;
            out_features[f_idx]     = (float)(bx / cL);
            out_features[f_idx + 1] = (float)(by / cW);
            out_features[f_idx + 2] = (float)(bz / cH);
            out_features[f_idx + 3] = contact;

            count++;
        }
    }

    static void get_space_features(clpState* state, float* out_features) {
        clp::Space space = state->cont->spaces->top();
        const bool* anchors = space.get_anchor();
        const Vector3 corner = space.get_corner();

        float contL = (float)state->cont->getL();
        float contW = (float)state->cont->getW();
        float contH = (float)state->cont->getH();

        out_features[0] = static_cast<float>(anchors[0] ? corner.getX() * -1 + contL : corner.getX()) / contL;
        out_features[1] = static_cast<float>(anchors[1] ? corner.getY() * -1 + contW : corner.getY()) / contW;
        out_features[2] = static_cast<float>(anchors[2] ? corner.getZ() * -1 + contH : corner.getZ()) / contH;
        
        out_features[3] = static_cast<float>(space.getL()) / contL;
        out_features[4] = static_cast<float>(space.getW()) / contW;
        out_features[5] = static_cast<float>(space.getH()) / contH;
    }

    // Retorna el ratio de volumen ocupado (0.0 a 1.0)
    static double get_volume(clpState* state) {
        return state->cont->getOccupiedVolume() / state->cont->getVolume();
    }

    static void get_actions_data_batch(const std::vector<clp::clpState*>& states, VCS_Function* vcs, int w, map<int, int>& block_id_to_index, int* out_blocks_ptr, float* out_features_ptr) {
        size_t num_states = states.size();
        size_t limit = (size_t)(w * w);

        for (size_t i = 0; i < num_states; ++i) {
            int* current_block_ptr = out_blocks_ptr + (i * limit);
            float* current_feature_ptr = out_features_ptr + (i * limit * 2);
            
            get_actions_data(states[i], vcs, w, block_id_to_index, current_block_ptr, current_feature_ptr, limit);
        }
    }

    static void get_placed_data_batch(const std::vector<clp::clpState*>& states, map<int, int>& block_id_to_index, int* out_blocks_ptr, float* out_features_ptr, int padding = 64) {
        size_t num_states = states.size();

        for (size_t i = 0; i < num_states; ++i) {
            int* current_block_ptr = out_blocks_ptr + (i * padding);
            float* current_feature_ptr = out_features_ptr + (i * padding * 4);
            
            get_placed_data(states[i], block_id_to_index, current_block_ptr, current_feature_ptr, padding);
        }
    }

    static void get_space_features_batch(const std::vector<clp::clpState*>& states, float* out_spaces_ptr) {
        size_t num_states = states.size();

        for (size_t i = 0; i < num_states; ++i) {
            get_space_features(states[i], out_spaces_ptr + (i * 6));
        }
    }
};

#endif