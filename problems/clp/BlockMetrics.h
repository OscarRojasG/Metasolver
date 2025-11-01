#ifndef BLOCKMETRICS_H
#define BLOCKMETRICS_H

#include "Block.h"

namespace clp {

class BlockMetrics {
private:
    const Block& block;
    const Block& cont;

    // --- Atributos métricos ---
    double normL = 0;
    double normW = 0;
    double normH = 0;
    double normOccupiedVolumeBlock = 0;
    double normOccupiedVolumeCont = 0;

    double boxesAmount = 0;
    double avgBoxL = 0;
    double avgBoxW = 0;
    double avgBoxH = 0;
    double stdBoxL = 0;
    double stdBoxW = 0;
    double stdBoxH = 0;
    double avgBoxV = 0;
    double stdBoxV = 0;

    void calculateMetrics();

public:
    // --- Constructor ---
    BlockMetrics(const Block& block, const Block& cont);

    // --- Getters ---
    double getNormL() const { return normL; }
    double getNormW() const { return normW; }
    double getNormH() const { return normH; }
    double getNormOccupiedVolumeBlock() const { return normOccupiedVolumeBlock; }
    double getNormOccupiedVolumeCont() const { return normOccupiedVolumeCont; }

    double getBoxesAmountReciprocal() const { return 1/boxesAmount; }
    double getAverageBoxL() const { return avgBoxL; }
    double getAverageBoxW() const { return avgBoxW; }
    double getAverageBoxH() const { return avgBoxH; }
    double getStdBoxL() const { return stdBoxL; }
    double getStdBoxW() const { return stdBoxW; }
    double getStdBoxH() const { return stdBoxH; }
    double getAverageBoxVolume() const { return avgBoxV; }
    double getStdBoxVolume() const { return stdBoxV; }
};

} // namespace clp

#endif // BLOCKMETRICS_H
