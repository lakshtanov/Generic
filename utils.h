#pragma once

#ifndef PARALLEL_IMPLEMENTATION

#include <cmath>
#define M_PI 3.14159265358979323846

namespace std {

inline double cdf_normal (double x) {
    return std::erfc(-x / std::sqrt(2)) / 2;
}
 
inline double cdf_normal(const double& x, double& pdf) {
    pdf = exp(-x * x * 0.5) / 2.506628274631;
    return cdf_normal(x);
}

}
#endif


#ifdef PARALLEL_IMPLEMENTATION
inline vtype stepApproximation(const idouble& x, const idouble& delta) {
    vtype r = iIf(x < 0, vtype(0.), x  / delta );  
    return iIf(x>delta, 1., r);
} 
#else 
inline vtype stepApproximation(const double& x, const double& delta) {
    vtype r = x < 0  ? vtype(0.) : x  / delta ;  
    return x>delta ? 1. : r;
}
#endif
