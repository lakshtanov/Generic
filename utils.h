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

// Floor a scalar or vector asset argument at a small positive value so that
// closed-form Pi functions (BlackScholes, RainbowCallMaxBlack, etc. which
// take log(S/K) internally) do not evaluate on a non-positive argument when
// a finite-difference vol shock pushes the simulated asset below zero.
// BasketBachelier uses arithmetic mean without logs, but clamping is still
// safe there (positive floor ≤ any realistic asset).
inline vtype aadcClampPositive(const vtype& x) {
    return std::max(x, vtype(1e-10));
}

#include <Eigen/Dense>
inline Eigen::Matrix<vtype, Eigen::Dynamic, 1>
aadcClampPositive(const Eigen::Matrix<vtype, Eigen::Dynamic, 1>& v) {
    Eigen::Matrix<vtype, Eigen::Dynamic, 1> r(v.size());
    for (int i = 0; i < v.size(); ++i) r[i] = std::max(v[i], vtype(1e-10));
    return r;
}

// Extract a plain double from vtype (which may be idouble in AADC build).
inline double toDouble(const double& x) { return x; }
#ifdef PARALLEL_IMPLEMENTATION
inline double toDouble(const idouble& x) { return x.val; }
#endif

// Adaptive FD step-size: keep |eps * d_component| ≤ max_shock for all
// components. Returns eps_local = min(eps_default, max_shock / ||d||_∞).
// For scalar direction (1-dim process), ||d||_∞ = |d|.
inline vtype adaptiveEps(const vtype& direction,
                         double eps_default, double max_shock) {
    double dir_inf = std::abs(toDouble(direction));
    if (dir_inf < 1e-30) return vtype(eps_default);
    return vtype(std::min(eps_default, max_shock / dir_inf));
}

inline vtype adaptiveEps(const Eigen::Matrix<vtype, Eigen::Dynamic, 1>& direction,
                         double eps_default, double max_shock) {
    double dir_inf = 0;
    for (int i = 0; i < direction.size(); ++i)
        dir_inf = std::max(dir_inf, std::abs(toDouble(direction[i])));
    if (dir_inf < 1e-30) return vtype(eps_default);
    return vtype(std::min(eps_default, max_shock / dir_inf));
}
