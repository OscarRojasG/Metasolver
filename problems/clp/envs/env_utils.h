#include "clpState.h"

class EnvUtils {
public:
    template<class map_container>
    static list<State*> get_next_states(map_container& sorted_states, int w) {
        list<State*> nextS;
        typename map_container::iterator state_action=sorted_states.begin();

        int k = 0;
        while(state_action!=sorted_states.end()) {
            State* s = state_action->second.first;
            State* final_state=state_action->second.second;
            Action* a = (s) ? s->next_action(*final_state) : NULL;

            if (nextS.size() < w && a) {
                s=s->clone();
                s->transition(*a);
                nextS.push_back(s);
            }

            delete final_state;
            state_action=sorted_states.erase(state_action);

            if (a) delete a;
            k++;
        }

        return nextS;
    }

};