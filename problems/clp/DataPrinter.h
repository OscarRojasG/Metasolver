#pragma once
#include <iostream>
#include <list>
#include <vector>
#include <algorithm>
#include "PathBuilder.h"
#include "VCS_Function.h"
#include "BlockMetrics.h"

using namespace std;
using namespace metasolver;

class DataPrinter {
private:
    PathBuilder* pathBuilder;  // solo referencia, no copia

public:
    explicit DataPrinter(PathBuilder* pb)
        : pathBuilder(pb) {}

    // ======================================================
    //  imprimir acciones con sus métricas
    // ======================================================
    void printActions(VCS_Function* vcs, int w) {
        const clpState& s = pathBuilder->getState();

        list<Action*> actions;
        s.get_actions(actions);

        multimap<double, Action*> ranked_actions;

        for (auto a : actions) {
            double eval = vcs->eval_action(s, *a);
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
        for (const Block* block : pathBuilder->getInitialState().valid_blocks) {
            BlockMetrics bm(*block, *(pathBuilder->getInitialState().cont));
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
        const clpState& s = pathBuilder->getState();
        clp::Space space = s.cont->spaces->top();
        const bool* anchors = space.get_anchor();
        const Vector3 corner = space.get_corner();

        long sx = anchors[0] ? corner.getX() * -1 : corner.getX();
        long sy = anchors[1] ? corner.getY() * -1 : corner.getY();
        long sz = anchors[2] ? corner.getZ() * -1 : corner.getZ();

        for (const clpAction* action : pathBuilder->getActions()) {
            const Space& sb = action->space;
            const Block& block = action->block;
            Vector3 coords = sb.get_location(block);

            long bx = anchors[0] ? (coords.getX() + block.getL()) * -1 : coords.getX();
            long by = anchors[1] ? (coords.getY() + block.getW()) * -1 : coords.getY();
            long bz = anchors[2] ? (coords.getZ() + block.getH()) * -1 : coords.getZ();

            double rbx = (double)(bx - sx);
            double rby = (double)(by - sy);
            double rbz = (double)(bz - sz);

            int contact = 1;
            if ((rbx + block.getL() < 0) || (rbx > space.getL()) ||
                (rby + block.getW() < 0) || (rby > space.getW()) ||
                (rbz + block.getH() < 0) || (rbz > space.getH())) {
                    contact = 0;
            }

            rbx /= s.cont->getL();
            rby /= s.cont->getW();
            rbz /= s.cont->getH();

            cout << block.id << " " << rbx << " " << rby << " " << rbz << " " << contact << endl;
        }
    }

    // ======================================================
    //  imprimir coordenadas del corner
    // ======================================================
    /*
    void printCoords() {
        const clpState& s = pathBuilder->getState();
        clp::Space space = s.cont->spaces->top();
        const Vector3 corner = space.get_corner();
        const bool* anchors = space.get_anchor();

        long sx = anchors[0] ? corner.getX() * -1 : corner.getX();
        long sy = anchors[1] ? corner.getY() * -1 : corner.getY();
        long sz = anchors[2] ? corner.getZ() * -1 : corner.getZ();

        cout << sx << " " << sy << " " << sz << endl;
    }
    */

    // ======================================================
    //  imprimir volumen ocupado
    // ======================================================
    void printVolume() {
        const clpState& s = pathBuilder->getState();
        cout << s.cont->getOccupiedVolume() / s.cont->getVolume() << endl;
    }
};