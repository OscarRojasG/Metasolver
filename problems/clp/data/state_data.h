#ifndef STATE_DATA_H
#define STATE_DATA_H

#include <vector>
#include <list>
#include <map>
#include <utility>
#include "clpState.h"
#include "VCS_Function.h"
#include "Greedy.h"
#include "block_data.h"

class StateData {
    private:
        std::vector<float> action_features;
        std::vector<float> placed_features;
        std::vector<float> space_features;
        double volume_ratio;
    
        void compute_action_features(clpState* state, VCS_Function* vcs, const std::map<int, int>& id_to_idx, int num_actions) {
            std::multimap<double, Action*> ranked_actions = get_ranked_actions(state, vcs, num_actions);
            action_features.reserve(ranked_actions.size() * N_ACTION_FEATURES);
    
            for (auto it = ranked_actions.rbegin(); it != ranked_actions.rend(); ++it) {
                auto ca = static_cast<clp::clpAction*>(it->second);
                
                action_features.push_back((float)id_to_idx.at(ca->block.id)); 
                action_features.push_back((float)ca->metrics.vcs);
                action_features.push_back((float)ca->metrics.loss);
                action_features.push_back((float)ca->metrics.cs);
                
                delete it->second;
            }
        }
    
        void compute_placed_features(clpState* state, const std::map<int, int>& id_to_idx) {
            std::list<const Action*> path = state->get_path();
            clp::Space current_space = state->cont->spaces->top();
            const bool* current_anchors = current_space.get_anchor();
            
            double cL = (double)state->cont->getL();
            double cW = (double)state->cont->getW();
            double cH = (double)state->cont->getH();
            double md = max({cL, cW, cH});
    
            placed_features.reserve(path.size() * N_PLACED_FEATURES);
    
            for (const Action* a : path) {
                auto ca = static_cast<const clp::clpAction*>(a);
                Vector3 coords = ca->space.get_location(ca->block);
    
                double bx = current_anchors[0] ? (coords.getX() + ca->block.getL()) * -1 + cL : coords.getX();
                double by = current_anchors[1] ? (coords.getY() + ca->block.getW()) * -1 + cW : coords.getY();
                double bz = current_anchors[2] ? (coords.getZ() + ca->block.getH()) * -1 + cH : coords.getZ();

                placed_features.push_back((float)id_to_idx.at(ca->block.id));
                placed_features.push_back((float)(bx / md));
                placed_features.push_back((float)(by / md));
                placed_features.push_back((float)(bz / md));
            }
        }
    
        void compute_space_features(clpState* state) {
            clp::Space space = state->cont->spaces->top();
            const bool* anchors = space.get_anchor();
            const Vector3 corner = space.get_corner();
    
            float cL = (float)state->cont->getL();
            float cW = (float)state->cont->getW();
            float cH = (float)state->cont->getH();
            double md = max({cL, cW, cH});
    
            space_features.reserve(N_SPACE_FEATURES);
    
            space_features.push_back((float)(anchors[0] ? corner.getX() * -1 + cL : corner.getX()) / md);
            space_features.push_back((float)(anchors[1] ? corner.getY() * -1 + cW : corner.getY()) / md);
            space_features.push_back((float)(anchors[2] ? corner.getZ() * -1 + cH : corner.getZ()) / md);
            
            space_features.push_back((float)space.getL() / md);
            space_features.push_back((float)space.getW() / md);
            space_features.push_back((float)space.getH() / md);
        }
    
    public:
        static const int N_ACTION_FEATURES = 4;
        static const int N_PLACED_FEATURES = 4;
        static const int N_SPACE_FEATURES = 6;

        clpState* state;
        VCS_Function* vcs;
        int num_actions;

        StateData(const BlockData& block_data, clpState* s, VCS_Function* vcs, int num_actions) 
            : state(s), vcs(vcs), num_actions(num_actions) {
            
            const std::map<int, int>& id_to_idx = block_data.get_block_id_to_index();
            
            compute_action_features(s, vcs, id_to_idx, num_actions);

            if (state->cont->spaces->size() > 0) {
                compute_placed_features(s, id_to_idx);
                compute_space_features(s);
            }
            volume_ratio = s->cont->getOccupiedVolume() / s->cont->getVolume();
        }
    
        const std::vector<float>& get_action_features() const { return action_features; }
        const std::vector<float>& get_placed_features() const { return placed_features; }
        const std::vector<float>& get_space_features() const { return space_features; }
        double get_volume_ratio() const { return volume_ratio; }

        std::vector<float> get_greedy_values() {
            std::multimap<double, Action*> ranked_actions = get_ranked_actions(state, vcs, num_actions);
            SearchStrategy* gr = new Greedy(vcs);

            std::vector<float> greedy_values;
            greedy_values.reserve(ranked_actions.size());
    
            for (auto it = ranked_actions.rbegin(); it != ranked_actions.rend(); it++) {
                clpState* s_copy = dynamic_cast<clpState*>(state->clone()); 
                s_copy->transition(*it->second);
    
                double value = gr->run(*s_copy);
                greedy_values.push_back((float)value);

                delete s_copy; 
            }

            delete gr; 
            
            for (auto item : ranked_actions) {
                delete item.second;
            }

            return greedy_values;
        }

        std::multimap<double, Action*> get_ranked_actions(clpState* state, VCS_Function* vcs, int num_actions) {       
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
                } else { 
                    delete a;
                }
            }
            return ranked_actions;
        }
    };

#endif