#pragma once
#include <iostream>
#include <list>
#include <vector>
#include <algorithm>
#include "VCS_Function.h"
#include "BlockMetrics.h"
#include "clpState.h"
#include <cstdint>

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

        // Primero, contamos cuántas acciones válidas tenemos realmente
        uint32_t num_to_print = 0;
        for (auto& p : scored) {
            if (num_to_print >= (uint32_t)(w * w)) break;
            if (dynamic_cast<clp::clpAction*>(p.second)) num_to_print++;
        }

        // 1. Enviar cabecera: número de acciones
        std::fwrite(&num_to_print, sizeof(uint32_t), 1, stdout);

        for (auto& p : scored) {
            if (num_to_print == 0) break; // Seguridad
            clp::clpAction* ca = dynamic_cast<clp::clpAction*>(p.second);
            if (!ca) continue;

            // Suponiendo que ca->metrics tiene un tamaño fijo (por ejemplo, 8)
            // Ajusta el tamaño del array según tus métricas reales
            size_t n_metrics = ca->metrics.size();
            float data[n_metrics + 1]; 
            
            data[0] = static_cast<float>(ca->block.id);
            size_t i = 0;
            for (auto m : ca->metrics) {
                data[i+1] = static_cast<float>(m);
                ++i;
            }

            // 2. Enviar ID + métricas
            std::fwrite(data, sizeof(float), n_metrics + 1, stdout);
            
            if (--num_to_print == 0) break;
        }
        std::fflush(stdout);
    }

    // ======================================================
    //  imprimir métricas de cada bloque válido
    // ======================================================
    void printBlocks() {
        uint32_t num_blocks = static_cast<uint32_t>(state->valid_blocks.size());
        
        // 1. Enviar el total de bloques primero (Header)
        std::fwrite(&num_blocks, sizeof(uint32_t), 1, stdout);

        for (const Block* block : state->valid_blocks) {
            BlockMetrics bm(*block, *(state->cont));
            
            // Creamos un array temporal con los datos del bloque (8 floats + 1 int)
            // OJO: Asegúrate de que los tipos coincidan con lo que esperas en Python
            float data[9]; 
            data[0] = static_cast<float>(block->id);
            data[1] = bm.getNormL();
            data[2] = bm.getNormH();
            data[3] = bm.getNormW();
            data[4] = bm.getNormOccupiedVolumeCont();
            data[5] = bm.getBoxesAmountReciprocal();
            data[6] = bm.getNormL() * bm.getNormW();
            data[7] = bm.getNormW() * bm.getNormH();
            data[8] = bm.getNormH() * bm.getNormL();

            // 2. Enviar el bloque completo de una sola vez
            std::fwrite(data, sizeof(float), 9, stdout);
        }
        // 3. Flush manual al final de TODOS los bloques, no por cada uno
        std::fflush(stdout);
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
        uint32_t num_placed = static_cast<uint32_t>(actions.size());

        // 1. Enviar cuántos bloques hay colocados
        std::fwrite(&num_placed, sizeof(uint32_t), 1, stdout);

        for (const Action* a : actions) {
            const clp::clpAction* action = static_cast<const clp::clpAction*>(a);
            const Space& sb = action->space;
            const Block& block = action->block;
            Vector3 coords = sb.get_location(block);

            double bx = anchors[0] ? (coords.getX() + block.getL()) * -1 + state->cont->getL(): coords.getX();
            double by = anchors[1] ? (coords.getY() + block.getW()) * -1 + state->cont->getW(): coords.getY();
            double bz = anchors[2] ? (coords.getZ() + block.getH()) * -1 + state->cont->getH(): coords.getZ();

            float contact = 1;
            if ((bx + block.getL() < sx) || (bx > sx + space.getL()) ||
                (by + block.getW() < sy) || (by > sy + space.getW()) ||
                (bz + block.getH() < sz) || (bz > sz + space.getH())) {
                    contact = 0;
            }

            bx /= state->cont->getL();
            by /= state->cont->getW();
            bz /= state->cont->getH();

            float data[5];
            data[0] = static_cast<float>(block.id);
            data[1] = static_cast<float>(bx / state->cont->getL());
            data[2] = static_cast<float>(by / state->cont->getW());
            data[3] = static_cast<float>(bz / state->cont->getH());
            data[4] = static_cast<float>(contact);

            // 2. Enviar el bloque de datos
            std::fwrite(data, sizeof(float), 5, stdout);
        }
        std::fflush(stdout);
    }

    void printVolume() {
        cout << state->cont->getOccupiedVolume() / state->cont->getVolume() << endl;
    }

    void printSpace() {
        Space space = state->cont->spaces->top();
        const bool* anchors = space.get_anchor();
        const Vector3 corner = space.get_corner();

        // Calculamos los valores
        float features[6];
        features[0] = static_cast<float>(anchors[0] ? corner.getX() * -1 + state->cont->getL() : corner.getX()) / state->cont->getL();
        features[1] = static_cast<float>(anchors[1] ? corner.getY() * -1 + state->cont->getW() : corner.getY()) / state->cont->getW();
        features[2] = static_cast<float>(anchors[2] ? corner.getZ() * -1 + state->cont->getH() : corner.getZ()) / state->cont->getH();
        features[3] = static_cast<float>(space.getL()) / state->cont->getL();
        features[4] = static_cast<float>(space.getW()) / state->cont->getW();
        features[5] = static_cast<float>(space.getH()) / state->cont->getH();

        // Escribimos los 6 floats de un solo golpe
        std::fwrite(features, sizeof(float), 6, stdout);
        std::fflush(stdout);
    }
};