#ifndef PATHBUILDER_H
#define PATHBUILDER_H

#include <list>
#include <array>
#include "clpState.h"
#include "Block.h"

namespace clp {

class PathBuilder {
private:
    clpState state;                         // Copia del estado inicial
    std::list<const clpAction*> actions;    // Lista de acciones

    // --- Métricas generales ---
    double totalVolume = 0.0;
    double totalVolumeRatio = 0.0;

    double volumeRatioAcum = 0.0;
    size_t numBlocks = 0;
    double avgVolume = 0.0;
    double avgVolumeRatio = 0.0;

    // --- Distribución espacial ---
    std::array<double, 8> quadrantVolumeRatio = {0.0};
    std::array<double, 8> quadrantMaxVolumeRatio = {0.0};

    // --- Funciones auxiliares ---
    void updateMetrics(const clpAction* action);
    void updateQuadrantVolume(const Space s);

public:
    // --- Constructor ---
    explicit PathBuilder(const clpState& s0);

    // --- Inserción ---
    void addAction(const clpAction* action);

    // --- Acceso ---
    const clpState& getState() const;
    const std::list<const clpAction*>& getActions() const;

    // --- Getters de métricas ---
    double getTotalVolume() const;
    double getTotalVolumeRatio() const;
    size_t getNumBlocks() const;
    double getAvgVolume() const;
    double getAvgVolumeRatio() const;

    // --- Getter del ratio por cuadrante ---
    const std::array<double, 8>& getQuadrantVolumeRatio() const;
    const std::array<double, 8>& getQuadrantMaxVolumeRatio() const;
};

} // namespace clp

#endif // PATHBUILDER_H