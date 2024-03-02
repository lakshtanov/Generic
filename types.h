#pragma once
#include <Eigen/Dense>

#ifdef PARALLEL_IMPLEMENTATION
#include <aadc/aadc.h>
#include <aadc/aadc_matrix.h>
using namespace aadc;
typedef idouble vtype;
typedef ibool vbool;

#else
typedef double vtype;
typedef bool vbool;
inline double iIf(const bool cond, const double a, const double b) {
    return cond ? a : b;
}
inline bool iIf(const bool cond, const bool a, const bool b) {
    return (cond && a) || (!cond && b);
}
inline int iIf(const bool& cond, const int& a, const int& b) {
    return cond ? a : b;
}

#endif

typedef double Time;

namespace Eigen {
    typedef Matrix<vtype, Eigen::Dynamic, 1>  vVectorXd;
    typedef Matrix<vtype, Eigen::Dynamic, Dynamic> vMatrixXd;
}

namespace AffineElement {

    ///////////////////////////////////////////////////////////////////////////////
    // Following methods required for sensitivites computations only
    ///////////////////////////////////////////////////////////////////////////////
    inline vtype getI(const int& i, const double& eps, const vtype& asset) { return eps; }
    
    inline Eigen::vVectorXd getI(const int& i, const double& eps, const Eigen::vVectorXd& asset) {
        Eigen::vVectorXd v = Eigen::vVectorXd::Zero(asset.size());
        v[i] = v[i] + eps;
        return v;
    }
}