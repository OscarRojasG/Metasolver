#ifndef OBJECTS_BOXSHAPE_H_
#define OBJECTS_BOXSHAPE_H_

#include "Volume.h"

namespace clp {

class BoxShape : public Volume {
public:
    enum Orientation{LWH=0, LHW, WLH, WHL, HLW, HWL, OR};

    // Constructor actualizado
    BoxShape(int id, double l, double w, double h, bool rot1=true, bool rot2=true, bool rot3=true, double weight=1.0, double profit=1.0);

    bool is_valid(Orientation o) const {return is_valid_orientation[o];}

    long getL(Orientation o) const;
    long getW(Orientation o) const;
    long getH(Orientation o) const;
    double getL_d(Orientation o) const;
    double getW_d(Orientation o) const;
    double getH_d(Orientation o) const;
    long get(int i, Orientation o) const;

    // Operador de comparación corregido para usar el ID
    bool operator<(const BoxShape& bt) const {
        return (id < bt.id);
    }

    inline int get_id() const { return id; }
    double get_weight() const { return weight; }
    double get_profit() const { return profit; }

    // El ID debe ser accesible para el Comparator
    int id; 
    static int INCREMENTAL_ID;

private:
    bool is_valid_orientation[6];
    double weight;
    double profit;
};

// Definición del comparador determinista para std::map y std::set
struct BoxShapeComparator {
    bool operator()(const BoxShape* a, const BoxShape* b) const {
        if (!a || !b) return a < b;
        if (a->id != b->id) return a->id < b->id;
        return a < b; // Fallback por dirección de memoria solo si los IDs chocan
    }
};

} /* namespace clp */

#endif /* OBJECTS_BOXSHAPE_H_ */