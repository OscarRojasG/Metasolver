#ifndef VCS_FUNCTION_H_
#define VCS_FUNCTION_H_

#include "VLossFunction.h"
#include <vector>
#include <map>

namespace clp {

class VCS_Function : public VLossFunction {
public:
    // Constructor principal actualizado con BoxShapeComparator
    VCS_Function(std::map<const BoxShape*, int, BoxShapeComparator>& nb_boxes, Vector3& dims, 
                 double alpha=4.0, double beta=1.0, double gamma=0.2, double p=0.04, 
                 double delta=1.0, double delta2=0.0, double delta3=0.0, double r=0.0);

    // Constructor secundario (theta) actualizado
    VCS_Function(std::map<const BoxShape*, int, BoxShapeComparator>& nb_boxes, Vector3& dims, 
                 std::vector<double>& theta, double r=0.0) :
        VCS_Function(nb_boxes, dims, theta[4], theta[5], theta[6], theta[3], theta[0], theta[2], theta[1], r) { }

    virtual ~VCS_Function();

    virtual double eval_action(const State& s, const Action& a);

    virtual void set_parameters(std::vector<double>& theta) {
        alpha = theta[4]; beta = theta[5]; gamma = theta[6];
        p = theta[3]; delta = theta[0]; delta2 = theta[2]; delta3 = theta[1];
    }

    virtual void set_parameters2(std::vector<double>& theta) {
        alpha_2 = theta[4]; beta_2 = theta[5]; gamma_2 = theta[6];
        p_2 = theta[3]; delta_2 = theta[0]; delta2_2 = theta[2]; delta3_2 = theta[1];
    }

    double cs[6];
    double weighted_cs;
    static int nn;
    double CS_p(const State& s, const Block& b, const Space& sp, double p);

protected:
    long _surface_in_contact(const AABB& b, const AABB& bi);
    long _surface_in_contact(const AABB& bi, const Block& c);

    double alpha, beta, gamma, p, delta, delta2, delta3;
    double alpha_2, beta_2, gamma_2, p_2, delta_2, delta2_2, delta3_2;
};

} /* namespace clp */

#endif