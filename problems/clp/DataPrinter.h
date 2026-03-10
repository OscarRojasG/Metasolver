#pragma once
#include <iostream>
#include <list>
#include <vector>
#include <algorithm>
#include "VCS_Function.h"
#include "BlockMetrics.h"
#include "clpState.h"

using namespace std;
using namespace metasolver;

class DataPrinter {
private:
    clpState* state;  // solo referencia, no copia

public:
    explicit DataPrinter(clpState* s)
        : state(s) {}

    // ======================================================
    //  imprimir acciones con sus métricas
    // ======================================================
    void printActions(VCS_Function* vcs, int w) {
        list<Action*> actions;
        state->get_actions(actions);

        multimap<double, Action*> ranked_actions;

        for (auto a : actions) {
            double eval = vcs->eval_action(*state, *a);
            if (eval > 0 && (ranked_actions.size() < w*w || ranked_actions.begin()->first < eval)) {
                ranked_actions.insert(make_pair(eval, a));
                if (ranked_actions.size() == w*w + 1) {
                    ranked_actions.erase(ranked_actions.begin());
                }
            } else {
                delete a;
            }
        }

        vector<pair<double, Action*>> scored(ranked_actions.begin(), ranked_actions.end());
        reverse(scored.begin(), scored.end()); // de mejor a peor

        int printed = 0;
        for (auto& p : scored) {
            if (printed >= w * w) break;
            clp::clpAction* ca = dynamic_cast<clp::clpAction*>(p.second);
            if (!ca) continue;
            cout << ca->block.id;
            for (auto m : ca->metrics) cout << " " << m;
            cout << endl;
            printed++;
        }
    }

    // ======================================================
    //  imprimir métricas de cada bloque válido
    // ======================================================
    void printBlocks() {
        for (const Block* block : state->valid_blocks) {
            BlockMetrics bm(*block, *(state->cont));
            cout << block->id
                 << " " << bm.getNormL()
                 << " " << bm.getNormH()
                 << " " << bm.getNormW()
                 << " " << bm.getNormOccupiedVolumeCont()
                 << " " << bm.getBoxesAmountReciprocal()
                 << " " << bm.getNormL() * bm.getNormW()
                 << " " << bm.getNormW() * bm.getNormH()
                 << " " << bm.getNormH() * bm.getNormL()
                 << endl;
        }
    }

    // ======================================================
    //  imprimir coordenadas relativas de bloques colocados
    // ======================================================
    void printPlaced() {
        clp::Space space = state->cont->spaces->top();
        const bool* anchors = space.get_anchor();
        const Vector3 corner = space.get_corner();

        long sx = anchors[0] ? corner.getX() * -1 + state->cont->getL(): corner.getX();
        long sy = anchors[1] ? corner.getY() * -1 + state->cont->getW(): corner.getY();
        long sz = anchors[2] ? corner.getZ() * -1 + state->cont->getH(): corner.getZ();

        list<const Action*> actions = state->get_path();
        for (const Action* a : actions) {
            const clp::clpAction* action = static_cast<const clp::clpAction*>(a);
            const Space& sb = action->space;
            const Block& block = action->block;
            Vector3 coords = sb.get_location(block);

            double bx = anchors[0] ? (coords.getX() + block.getL()) * -1 + state->cont->getL(): coords.getX();
            double by = anchors[1] ? (coords.getY() + block.getW()) * -1 + state->cont->getW(): coords.getY();
            double bz = anchors[2] ? (coords.getZ() + block.getH()) * -1 + state->cont->getH(): coords.getZ();

            int contact = 1;
            if ((bx + block.getL() < sx) || (bx > sx + space.getL()) ||
                (by + block.getW() < sy) || (by > sy + space.getW()) ||
                (bz + block.getH() < sz) || (bz > sz + space.getH())) {
                    contact = 0;
            }

            bx /= state->cont->getL();
            by /= state->cont->getW();
            bz /= state->cont->getH();

            cout << block.id << " " << bx << " " << by << " " << bz << " " << contact << endl;
        }
    }

    void printVolume() {
        cout << state->cont->getOccupiedVolume() / state->cont->getVolume() << endl;
    }

    void printSpace() {
        Space space = state->cont->spaces->top();
        const bool* anchors = space.get_anchor();
        const Vector3 corner = space.get_corner();

        double sx = anchors[0] ? corner.getX() * -1 + state->cont->getL(): corner.getX();
        double sy = anchors[1] ? corner.getY() * -1 + state->cont->getW(): corner.getY();
        double sz = anchors[2] ? corner.getZ() * -1 + state->cont->getH(): corner.getZ();

        cout << sx / state->cont->getL() << " "
             << sy / state->cont->getW() << " "
             << sz / state->cont->getH() << " "
             << (double) space.getL() / state->cont->getL() << " "
             << (double) space.getW() / state->cont->getW() << " "
             << (double) space.getH() / state->cont->getH() << endl;
    }
};