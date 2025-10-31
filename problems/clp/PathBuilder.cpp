#include "PathBuilder.h"
#include <algorithm>

namespace clp {

PathBuilder::PathBuilder(const clpState& s0)
    : state(s0)
{
    totalVolume = 0.0;
    numBlocks = 0;
    avgVolume = 0.0;
    quadrantRatio.fill(0.0);
}

void PathBuilder::addAction(const clpAction* action) {
    if (!action) return;

    actions.push_back(action);
    state.transition(*action);
    updateMetrics(action);
}

void PathBuilder::updateQuadrantVolume(const Space s) {
    const Block* cont = state.cont;  // contenedor principal

    // --- Punto de corte dentro del contenedor ---
    double center_x = s.getXmin() + s.getL() / 2.0;
    double center_y = s.getYmin() + s.getW() / 2.0;
    double center_z = s.getZmin() + s.getH() / 2.0;

    // --- Coordenadas del contenedor completo ---
    double x1 = 0;
    double x3 = cont->getL();
    double y1 = 0;
    double y3 = cont->getW();
    double z1 = 0;
    double z3 = cont->getH();

    // --- Inicializar volúmenes ---
    std::array<double, 8> quadrantVolume = {0.0};
    std::array<double, 8> quadrantVolMax = {0.0};

    // --- Calcular volumen máximo de cada cuadrante global ---
    for (int qx = 0; qx < 2; ++qx) {
        for (int qy = 0; qy < 2; ++qy) {
            for (int qz = 0; qz < 2; ++qz) {
                int idx = (qx << 2) | (qy << 1) | qz;

                double qx_min = (qx == 0) ? x1 : center_x;
                double qx_max = (qx == 0) ? center_x : x3;
                double qy_min = (qy == 0) ? y1 : center_y;
                double qy_max = (qy == 0) ? center_y : y3;
                double qz_min = (qz == 0) ? z1 : center_z;
                double qz_max = (qz == 0) ? center_z : z3;

                quadrantVolMax[idx] = (qx_max - qx_min) *
                                      (qy_max - qy_min) *
                                      (qz_max - qz_min);
            }
        }
    }

    // --- Recorremos todos los bloques existentes ---
    for (const clpAction* action : actions) {
        const Block& b = action->block;
        const Space sb = action->space;

        // Posición del bloque dentro del contenedor
        Vector3 coords = sb.get_location(b);
        double bx_min = coords.getX();
        double by_min = coords.getY();
        double bz_min = coords.getZ();

        double bx_max = bx_min + b.getL();
        double by_max = by_min + b.getW();
        double bz_max = bz_min + b.getH();

        // Intersección del bloque con cada cuadrante global
        for (int qx = 0; qx < 2; ++qx) {
            for (int qy = 0; qy < 2; ++qy) {
                for (int qz = 0; qz < 2; ++qz) {
                    int idx = (qx << 2) | (qy << 1) | qz;

                    double qx_min = (qx == 0) ? x1 : center_x;
                    double qx_max = (qx == 0) ? center_x : x3;
                    double qy_min = (qy == 0) ? y1 : center_y;
                    double qy_max = (qy == 0) ? center_y : y3;
                    double qz_min = (qz == 0) ? z1 : center_z;
                    double qz_max = (qz == 0) ? center_z : z3;

                    // Calcular intersección bloque-cuadrante
                    double ix_min = std::max(bx_min, qx_min);
                    double ix_max = std::min(bx_max, qx_max);
                    double iy_min = std::max(by_min, qy_min);
                    double iy_max = std::min(by_max, qy_max);
                    double iz_min = std::max(bz_min, qz_min);
                    double iz_max = std::min(bz_max, qz_max);

                    if (ix_min < ix_max && iy_min < iy_max && iz_min < iz_max) {
                        double inter_vol = (ix_max - ix_min) *
                                           (iy_max - iy_min) *
                                           (iz_max - iz_min);

                        double occupied_ratio = b.getOccupiedVolume() / b.getVolume();
                        quadrantVolume[idx] += inter_vol * occupied_ratio;
                    }
                }
            }
        }
    }

    // --- Calcular ratios ocupados / volumen del cuadrante ---
    for (int i = 0; i < 8; ++i) {
        quadrantRatio[i] = (quadrantVolMax[i] > 0.0)
            ? (quadrantVolume[i] / quadrantVolMax[i])
            : 0.0;
    }
}

void PathBuilder::updateMetrics(const clpAction* action) {
    const Block& b = action->block;
    const Block* cont = state.cont;

    numBlocks++;
    totalVolume += b.getOccupiedVolume();
    totalVolumeRatio = totalVolume / cont->getVolume();

    volumeRatioAcum += b.getOccupiedVolume() / cont->getVolume();
    avgVolumeRatio = volumeRatioAcum / static_cast<double>(numBlocks);

    if (cont->spaces->size() > 0)
        updateQuadrantVolume(cont->spaces->top());
}

// --- Getters ---
const clpState& PathBuilder::getState() const {
    return state;
}

const std::list<const clpAction*>& PathBuilder::getActions() const {
    return actions;
}

double PathBuilder::getTotalVolume() const {
    return totalVolume;
}

double PathBuilder::getTotalVolumeRatio() const
{
    return totalVolumeRatio;
}

size_t PathBuilder::getNumBlocks() const {
    return numBlocks;
}

double PathBuilder::getAvgVolume() const {
    return avgVolume;
}

double PathBuilder::getAvgVolumeRatio() const
{
    return avgVolumeRatio;
}

const std::array<double, 8>& PathBuilder::getQuadrantRatio() const {
    return quadrantRatio;
}

} // namespace clp