#include "VLossFunction.h"
#include "../clpState.h"
#include <map>
#include "../objects2/BoxShape.h"
#include "../objects2/Vector3.h"

using namespace std;

namespace clp {

// 1. Actualización del Constructor: El mapa de entrada DEBE tener el comparador
VLossFunction::VLossFunction(const std::map<const BoxShape*, int, BoxShapeComparator>& nb_boxes, Vector3& dims, double r) :
                ActionEvaluator (r) {

    mL = new long[dims.getX()+1];
    mW = new long[dims.getY()+1];
    mH = new long[dims.getZ()+1];

    // Se inicializan los arrays de sets con el comparador determinista
    listL = new std::set<const BoxShape*, BoxShapeComparator>[dims.getX()+1];
    listW = new std::set<const BoxShape*, BoxShapeComparator>[dims.getY()+1];
    listH = new std::set<const BoxShape*, BoxShapeComparator>[dims.getZ()+1];

    solveKnapsack(nb_boxes, dims);
}

VLossFunction::~VLossFunction() {
     delete[] mL;
     delete[] mH;
     delete[] mW;
     delete[] listL;
     delete[] listH;
     delete[] listW;
}

// 2. Actualización de Loss: El mapa nb_boxes debe ser el tipo determinista
double VLossFunction::Loss(const std::map<const BoxShape*, int, BoxShapeComparator>& nb_boxes, const Block& block, const Space& free_space){
   long resL = free_space.getL() - block.getL();
   long resW = free_space.getW() - block.getW();
   long resH = free_space.getH() - block.getH();

   long lossL = resL, lossW = resW, lossH = resH;
   
   long maxL = compute_maxX(nb_boxes, block, lossL, resL, mL, listL);
   long maxW = compute_maxX(nb_boxes, block, lossW, resW, mW, listW);
   long maxH = compute_maxX(nb_boxes, block, lossH, resH, mH, listH);

   long vloss = (mL[free_space.getL()]*mW[free_space.getW()]*mH[free_space.getH()]) -
                  ((free_space.getL()-lossL)*(free_space.getW() - lossW)*(free_space.getH()-lossH));
   
   return (double) vloss / (double) (mL[free_space.getL()]*mW[free_space.getW()]*mH[free_space.getH()]);
}

void VLossFunction::solveKnapsack(const std::map<const BoxShape*, int, BoxShapeComparator>& nb_boxes, Vector3& dims){
    compute_mX(nb_boxes, dims.getX(), mL, listL, 0);
    compute_mX(nb_boxes, dims.getY(), mW, listW, 1);
    compute_mX(nb_boxes, dims.getZ(), mH, listH, 2);
}

// 3. Actualización de compute_maxX: El iterador del set ahora es determinista
long VLossFunction::compute_maxX(const std::map<const BoxShape*, int, BoxShapeComparator>& nb_boxes, const Block& block,
        long& lossX, long resX, long* mX, std::set<const BoxShape*, BoxShapeComparator>* listX){
   
   long maxX = mX[resX];
   while(maxX > 0){
     // El set itera por ID, garantizando que el primer match sea siempre el mismo
     for(auto it = listX[maxX].begin(); it != listX[maxX].end(); it++){
        int n = 0;
        if(block.nb_boxes.count(*it)) n = block.nb_boxes.at(*it);

        if(nb_boxes.at(*it) - n > 0) {
               lossX = resX - maxX;
               maxX = 0;
               break;
        }
    }
    if(maxX > 0) maxX = mX[maxX-1];
  }
  return maxX;
}

// 4. Actualización de compute_mX: Garantiza orden de inserción por ID
void VLossFunction::compute_mX(const std::map<const BoxShape*, int, BoxShapeComparator>& nb_boxes, int X, long *mX,
        std::set<const BoxShape*, BoxShapeComparator>* listX, int dim){
     
     bool flag[X+1]; 
     flag[0] = true;

     for(int i=1; i<=X; i++) {
            flag[i] = false;
            listX[i].clear();
     }

     // Este bucle ahora recorre las cajas por ID (porque el mapa nb_boxes usa el Comparator)
     for(auto it_nb = nb_boxes.begin(); it_nb != nb_boxes.end(); it_nb++){
         int n = it_nb->second;
         if(n > 0){
              for(int ii=0; ii<3; ii++){
                  int x = 0;
                  switch(ii){
                        case 0: x = int(it_nb->first->get(dim, BoxShape::LWH)); break;
                        case 1: if(it_nb->first->is_valid(BoxShape::WHL)) x = int(it_nb->first->get(dim, BoxShape::WHL)); break;
                        case 2: if(it_nb->first->is_valid(BoxShape::HLW)) x = int(it_nb->first->get(dim, BoxShape::HLW)); break;
                        default: break;
                  }
                  if(x == 0) continue;

                  for(int i=0; i<=X-x; i++){
                      if(!flag[i]) continue;
                      for(int j=min(n,(X-i)/x); j>=1; j--){
                         flag[i+x*j] = true;
                         listX[i+x*j].insert(it_nb->first);
                      }
                  }
              }
         }
     }

     for(int i=0; i<=X; i++){
        if(flag[i]) mX[i] = i;
        else mX[i] = mX[i-1];
     }
}

} /* namespace clp */