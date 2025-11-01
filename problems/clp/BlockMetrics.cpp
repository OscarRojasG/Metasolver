#include "BlockMetrics.h"

namespace clp {

BlockMetrics::BlockMetrics(const Block& block, const Block& cont)
    : block(block), cont(cont)
{
    calculateMetrics();
}

void BlockMetrics::calculateMetrics()
{
    normL = block.getL() / (double) cont.getL();
    normW = block.getW() / (double) cont.getW();
    normH = block.getH() / (double) cont.getH();

    normOccupiedVolumeBlock = block.getOccupiedVolume() / block.getVolume();
    normOccupiedVolumeCont = block.getOccupiedVolume() / cont.getVolume();

    double Lsum = 0, Wsum = 0, Hsum = 0;

    map<const BoxShape*, std::map<BoxShape::Orientation, int>> boxes = block.or_boxes;

    for (const auto& [shape, orientations] : boxes) {
        for (const auto& [o, count] : orientations) {
            if (count == 0) continue;

            boxesAmount += count;

            Lsum += shape->getL_d(o);
            Wsum += shape->getW_d(o);
            Hsum += shape->getH_d(o);
        }
    }

    // --- Calcular promedios ---
    avgBoxL = Lsum / boxesAmount;
    avgBoxW = Wsum / boxesAmount;
    avgBoxH = Hsum / boxesAmount;
    avgBoxV = block.getOccupiedVolume() / boxesAmount;

    // --- Calcular desviación estándar ---
    double Lvar = 0, Wvar = 0, Hvar = 0, Vvar = 0;

    for (const auto& [shape, orientations] : boxes) {
        for (const auto& [o, count] : orientations) {
            if (count == 0) continue;

            const double L = shape->getL_d(o);
            const double W = shape->getW_d(o);
            const double H = shape->getH_d(o);
            const double V = shape->getVolume();

            Lvar += count * pow(L - avgBoxL, 2);
            Wvar += count * pow(W - avgBoxW, 2);
            Hvar += count * pow(H - avgBoxH, 2);
            Vvar += count * pow(V - avgBoxV, 2);
        }
    }

    stdBoxL = (boxesAmount > 1) ? sqrt(Lvar / boxesAmount) : 0.0;
    stdBoxW = (boxesAmount > 1) ? sqrt(Wvar / boxesAmount) : 0.0;
    stdBoxH = (boxesAmount > 1) ? sqrt(Hvar / boxesAmount) : 0.0;
    stdBoxV = (boxesAmount > 1) ? sqrt(Vvar / boxesAmount) : 0.0;

    // --- Normalización ---
    avgBoxL /= block.getL();
    avgBoxW /= block.getW();
    avgBoxH /= block.getH();
    avgBoxV /= block.getOccupiedVolume();

    stdBoxL /= block.getL();
    stdBoxW /= block.getW();
    stdBoxH /= block.getH();
    stdBoxV /= block.getOccupiedVolume();
}

} // namespace clp