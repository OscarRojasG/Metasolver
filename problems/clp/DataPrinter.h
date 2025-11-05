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

        vector<pair<double, Action*>> scored;
        for (auto a : actions) {
            clp::clpAction* ca = dynamic_cast<clp::clpAction*>(a);
            if (!ca) continue;
            ca->metrics.clear();
            double val = vcs->eval_action(s, *a);
            scored.push_back({val, a});
        }

        sort(scored.begin(), scored.end(),
            [](const pair<double, Action*>& A, const pair<double, Action*>& B) {
                return A.first > B.first;
            });

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
        cout << "END" << endl;
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
                 << endl;
        }
        cout << "END" << endl;
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

            double rbx = (double)(bx - sx) / s.cont->getL();
            double rby = (double)(by - sy) / s.cont->getW();
            double rbz = (double)(bz - sz) / s.cont->getH();

            cout << block.id << " " << rbx << " " << rby << " " << rbz << endl;
        }
        cout << "END" << endl;
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