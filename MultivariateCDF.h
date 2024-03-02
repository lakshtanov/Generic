#pragma once
#include <memory>

#include <cmath>
#include <vector>

#ifdef PARALLEL_IMPLEMENTATION 
#include "aadc/aadc_matrix.h"
#include <aadc/aadc_eigen.h>
#endif
#include <Eigen/Dense>
using Eigen::Dynamic;
using Eigen::Matrix;

#include "src/Core/Matrix.h"
#include "src/Core/util/Constants.h"
#include "src/Eigenvalues/SelfAdjointEigenSolver.h"
#include "types.h"
#include "CorrelatedBM.h"
#include "QuadratureCoefficients.h"


class BivariateCumulativeNormalDistributionDr78 {
public:
    BivariateCumulativeNormalDistributionDr78(double rho)
        : rho_(rho), rho2_(rho* rho) {

        assert(rho >= -1.0);
        assert(rho <= 1.0);
    }

    double cdf_normal(double x) const {
        return std::erfc(-x / std::sqrt(2)) / 2;
    }




    double operator()(double a, double b) const {

        double pdf;
        double CumNormDistA = cdf_normal(a);
        double CumNormDistB = cdf_normal(b);
        double MaxCumNormDistAB = std::max(CumNormDistA, CumNormDistB);
        double MinCumNormDistAB = std::min(CumNormDistA, CumNormDistB);

        if (1.0 - MaxCumNormDistAB < 1e-15)
            return MinCumNormDistAB;

        if (MinCumNormDistAB < 1e-15)
            return MinCumNormDistAB;

        double a1 = a / std::sqrt(2.0 * (1.0 - rho2_));
        double b1 = b / std::sqrt(2.0 * (1.0 - rho2_));

        double result = -1.0;

        if (a <= 0.0 && b <= 0 && rho_ <= 0) {
            double sum = 0.0;
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    sum += x_[i] * x_[j] *
                        std::exp(a1 * (2.0 * y_[i] - a1) + b1 * (2.0 * y_[j] - b1)
                            + 2.0 * rho_ * (y_[i] - a1) * (y_[j] - b1));
                }
            }
            result = std::sqrt(1.0 - rho2_) / 3.14159265358979323846 * sum;
        }
        else if (a <= 0 && b >= 0 && rho_ >= 0) {
            BivariateCumulativeNormalDistributionDr78 bivCumNormalDist(-rho_);
            result = CumNormDistA - bivCumNormalDist(a, -b);
        }
        else if (a >= 0.0 && b <= 0.0 && rho_ >= 0.0) {
            BivariateCumulativeNormalDistributionDr78 bivCumNormalDist(-rho_);
            result = CumNormDistB - bivCumNormalDist(-a, b);
        }
        else if (a >= 0.0 && b >= 0.0 && rho_ <= 0.0) {
            result = CumNormDistA + CumNormDistB - 1.0 + (*this)(-a, -b);
        }
        else if (a * b * rho_ > 0.0) {
            double rho1 = (rho_ * a - b) * (a > 0.0 ? 1.0 : -1.0) /
                std::sqrt(a * a - 2.0 * rho_ * a * b + b * b);
            BivariateCumulativeNormalDistributionDr78 bivCumNormalDist(rho1);

            double rho2 = (rho_ * b - a) * (b > 0.0 ? 1.0 : -1.0) /
                std::sqrt(a * a - 2.0 * rho_ * a * b + b * b);
            BivariateCumulativeNormalDistributionDr78 CBND2(rho2);

            double delta = (1.0 - (a > 0.0 ? 1.0 : -1.0) * (b > 0.0 ? 1.0 : -1.0)) / 4.0;

            result = bivCumNormalDist(a, 0.0) + CBND2(b, 0.0) - delta;
        }
        else {
            throw("case not handled");
        }

        return result;
    }
public:
    double rho_, rho2_;
    static const double x_[], y_[];


};

const double BivariateCumulativeNormalDistributionDr78::x_[] = {
       0.24840615,
       0.39233107,
       0.21141819,
       0.03324666,
       0.00082485334
};

const double BivariateCumulativeNormalDistributionDr78::y_[] = {
    0.10024215,
    0.48281397,
    1.06094980,
    1.77972940,
    2.66976040000
};


    class eqn3 { /* Relates to eqn3 Genz 2004 */
    public:
        eqn3(double h, double k, double asr)
            : hk_(h* k), asr_(asr), hs_((h* h + k * k) / 2) {}

        double operator()(double x) const {
            double sn = std::sin(asr_ * (-x + 1) * 0.5);
            return std::exp((sn * hk_ - hs_) / (1.0 - sn * sn));
        }
    private:
        double hk_, asr_, hs_;
    };

    class eqn6 { /* Relates to eqn6 Genz 2004 */
    public:
        eqn6(double a, double c, double d, double bs, double hk)
            : a_(a), c_(c), d_(d), bs_(bs), hk_(hk) {}
        double operator()(double x) const {
            double xs = a_ * (-x + 1);
            xs = std::fabs(xs * xs);
            double rs = std::sqrt(1 - xs);
            double asr = -(bs_ / xs + hk_) / 2;
            if (asr > -100.0) {
                return (a_ * std::exp(asr) *
                    (std::exp(-hk_ * (1 - rs) / (2 * (1 + rs))) / rs -
                        (1 + c_ * xs * (1 + d_ * xs))));
            }
            else {
                return 0.0;
            }
        }
    private:
        double a_, c_, d_, bs_, hk_;
    };






double bvnl(double x, double y, double rho) {
    BivariateCumulativeNormalDistributionDr78 a(rho);
    return a(x, y);
}

void sincs(double x, double& sx, double& cs) {
    double ee = (PI / 2 - fabs(x)) * (PI / 2 - abs(x));
    if (ee < 5e-5) {
        sx = (1 - ee * (1 - ee / 12) / 2) * (x > 0 ? 1 : -1);
        cs = ee * (1 - ee * (1 - 2 * ee / 15) / 3);
    }
    else {
        sx = sin(x);
        cs = 1 - sx * sx;
    }
}


double pntgnd(double ba, double  bb, double bc, double ra, double rb, double  r, double  rr) {
    double f(0.);
    //%
    //%Computes Plackett formula integrand;
    //%
    double dt = rr * (rr - (ra - rb) * (ra - rb) - 2 * ra * rb * (1 - r));
    if (dt > 0) {
        double bt = (bc * rr + ba * (r * rb - ra) + bb * (r * ra - rb)) / sqrt(dt);
        double ft = (ba - r * bb) * (ba - r * bb) / rr + bb * bb;
        if (bt > -10 && ft < 100) {
            f = exp(-ft / 2);
            if (bt < 10) f = f * std::cdf_normal(bt);
        }
    }
    return f;
}


#include <cmath> // for std::abs and std::function
#include <functional>

template<class Fun>
void krnrdt(const double a, const double b, Fun f, double& resk, double& err) {
    const double wg0 = 0.2729250867779007;
    double wg[5] = { 0.05566856711617449, 0.1255803694649048, 0.1862902109277352, 0.2331937645919914, 0.2628045445102478 };
    double xgk[11] = { 0.9963696138895427, 0.9782286581460570, 0.9416771085780681, 0.8870625997680953, 0.8160574566562211,
                       0.7301520055740492, 0.6305995201619651, 0.5190961292068118, 0.3979441409523776, 0.2695431559523450,
                       0.1361130007993617 };
    const double wgk0 = 0.1365777947111183;
    double wgk[11] = { 0.00976544104596129, 0.02715655468210443, 0.04582937856442671, 0.06309742475037484, 0.07866457193222764,
                        0.09295309859690074, 0.1058720744813894, 0.1167395024610472, 0.1251587991003195, 0.1312806842298057,
                        0.1351935727998845 };
    const double wid = (b - a) / 2.0;
    const double cen = (b + a) / 2.0;
    double fc = f(cen);
    double resg = fc * wg0;
    resk = fc * wgk0;
    for (int j = 0; j < 5; j++) {
        const double t = wid * xgk[2 * j];
        double fc = f(cen - t) + f(cen + t);
        resk += wgk[2 * j] * fc;
        const double s = wid * xgk[2 * j + 1];
        fc = f(cen - s) + f(cen + s);
        resk += wgk[2 * j + 1] * fc;
        resg += wg[j] * fc;
    }
    const double t = wid * xgk[10];
    fc = f(cen - t) + f(cen + t);
    resk = wid * (resk + wgk[10] * fc);
    err = std::abs(resk - wid * resg);
}


template<class Fun>
double adonet(Fun f, double a, double b, double tol, int nl = 100) {
    double ai[101], bi[101], fi[101], ei[101];
    int ip = 1, im = 1;
    double err = 1, fin;
    ai[1] = a;
    bi[1] = b;
    while (4 * err > tol && im < nl) {
        im++;
        bi[im] = bi[ip];
        ai[im] = (ai[ip] + bi[ip]) / 2;
        bi[ip] = ai[im];
        krnrdt(ai[ip], bi[ip], f, fi[ip], ei[ip]);
        krnrdt(ai[im], bi[im], f, fi[im], ei[im]);
        fin = 0;
        double sq_sum_ei = 0;
        for (int i = 1; i <= im; i++) {
            fin += fi[i];
            sq_sum_ei += ei[i] * ei[i];
        }
        err = sqrt(sq_sum_ei);
        int index;
        double ex = 0;
        for (int i = 1; i <= im; i++) {
            if (ei[i] > ex) {
                ex = ei[i];
                index = i;
            }
        }
        ip = index;
    }
    return fin;
}



double tvnf(double x, double h1, double h2, double h3, double r23, double a12, double a13) {
    double f = 0.0;
    double r12, rr2, r13, rr3;
    sincs(a12 * x, r12, rr2);
    sincs(a13 * x, r13, rr3);
    if (fabs(a12) > 0) {
        f += a12 * pntgnd(h1, h2, h3, r13, r23, r12, rr2);
    }
    if (fabs(a13) > 0) {
        f += a13 * pntgnd(h1, h3, h2, r12, r23, r13, rr3);
    }
    return f;
}

//tvn
double tvnls(const double& h1_, const double& h2_, const double& h3_, const double& r12_, const double& r13_, const double& r23_) {
    //std::cout << 1. / 8 + 1. / (4 * 3.1415926) * (std::asin(r12) + std::asin(r13) + std::asin(r23));
    double h1(h1_), h2(h2_), h3(h3_), r12(r12_), r13(r13_), r23(r23_);
    double hh1 = h1, rr3 = r23;
    double tvn(0);
    double epst = 1e-14, pt = PI / 2;
    double sbst;
    if (fabs(r12) > fabs(r13)) { sbst = h2; h2 = h3; h3 = sbst; sbst = r12; r12 = r13; r13 = sbst; }
    if (fabs(r13) > fabs(r23)) { sbst = h1; h1 = h2; h2 = hh1; sbst = r23; r23 = r13; r13 = rr3; }
    if (fabs(h1) + fabs(h2) + fabs(h3) < epst) 
        tvn = (1 + (asin(r12) + asin(r13) + asin(r23)) / pt) / 8;
    else 
        if (abs(r12) + abs(r13) < epst) tvn = std::cdf_normal(h1) * bvnl(h2, h3, r23);
        else
            if (abs(r13) + abs(r23) < epst) tvn = std::cdf_normal(h3) * bvnl(h1, h2, r12);
            else
                if (abs(r12) + abs(r23) < epst) tvn = std::cdf_normal(h2) * bvnl(h1, h3, r13);
                else
                    if (1 - r23 < epst) tvn = bvnl(h1, std::min(h2, h3), r12);
                    else
                        if (r23 + 1 < epst) { if (h2 > -h3) tvn = bvnl(h1, h2, r12) - bvnl(h1, -h3, r12); }
                        else {
                            tvn =  bvnl(h2, h3, r23)* std::cdf_normal(h1);

                            auto fff = [&](double x) { 
                                return tvnf(x, h1, h2, h3, r23, asin(r12), asin(r13)); 
                            };
                            tvn += adonet(fff, 0, 1, epst) / (2 * 3.1415926);
                        }
    tvn = std::max(0., std::min(tvn, 1.));
    return tvn;
}

#ifdef PARALLEL_IMPLEMENTATION 

class TVNLSWrapper : public aadc::ConstStateExtFunc {
public:
  TVNLSWrapper(idouble& res, const idouble& h1,const  idouble& h2, const idouble& h3, const idouble& r12, const idouble& r13, const idouble& r23)
    : h1i(h1), h2i(h2), h3i(h3), r12i(r12), r13i(r13), r23i(r23)
    , resi(res)
  {
      // This code is executed during the recording stage.
      res.val = tvnls(h1.val, h2.val,h3.val,r12.val,r13.val,r23.val);
  }

  template<typename mmType>
  void forward(mmType* v) const {
      // This code is executed during the forward pass.
  	  // Note that we should use a mutex lock here if myfunc is not multithread safe.
      for (int avxi = 0; avxi < aadc::mmSize<mmType>(); ++avxi)
          toDblPtr(v[resi])[avxi] = tvnls(
              toDblPtr(v[h1i])[avxi], toDblPtr(v[h2i])[avxi], toDblPtr(v[h3i])[avxi], 
              toDblPtr(v[r12i])[avxi], toDblPtr(v[r13i])[avxi], toDblPtr(v[r23i])[avxi]
            );
  }
  // MANUAL
  // MANUAL Reverse using FD
  template<class mmType>
  void reverse(const mmType* v, mmType* d) const {
      // code to calculate the local gradient and update the adjoint variables.
      double h(0.1);
      for (int avxi = 0; avxi < aadc::mmSize<mmType>(); ++avxi) {
          double d1 = (tvnls(toDblPtr(v[h1i])[avxi] + h, toDblPtr(v[h2i])[avxi], toDblPtr(v[h3i])[avxi], toDblPtr(v[r12i])[avxi], toDblPtr(v[r13i])[avxi], toDblPtr(v[r23i])[avxi]) - tvnls(toDblPtr(v[h1i])[avxi]- h, toDblPtr(v[h2i])[avxi], toDblPtr(v[h3i])[avxi], toDblPtr(v[r12i])[avxi], toDblPtr(v[r13i])[avxi], toDblPtr(v[r23i])[avxi])) / (2 * h);
          double d2 = (tvnls(toDblPtr(v[h1i])[avxi], toDblPtr(v[h2i])[avxi]+ h, toDblPtr(v[h3i])[avxi], toDblPtr(v[r12i])[avxi], toDblPtr(v[r13i])[avxi], toDblPtr(v[r23i])[avxi]) - tvnls(toDblPtr(v[h1i])[avxi], toDblPtr(v[h2i])[avxi]- h, toDblPtr(v[h3i])[avxi], toDblPtr(v[r12i])[avxi], toDblPtr(v[r13i])[avxi], toDblPtr(v[r23i])[avxi])) / (2 * h);
          double d3 = (tvnls(toDblPtr(v[h1i])[avxi], toDblPtr(v[h2i])[avxi], toDblPtr(v[h3i])[avxi]+ h, toDblPtr(v[r12i])[avxi], toDblPtr(v[r13i])[avxi], toDblPtr(v[r23i])[avxi]) - tvnls(toDblPtr(v[h1i])[avxi], toDblPtr(v[h2i])[avxi], toDblPtr(v[h3i])[avxi]- h, toDblPtr(v[r12i])[avxi], toDblPtr(v[r13i])[avxi], toDblPtr(v[r23i])[avxi])) / (2 * h);
          double d4 = (tvnls(toDblPtr(v[h1i])[avxi], toDblPtr(v[h2i])[avxi], toDblPtr(v[h3i])[avxi], toDblPtr(v[r12i])[avxi]+ h, toDblPtr(v[r13i])[avxi], toDblPtr(v[r23i])[avxi]) - tvnls(toDblPtr(v[h1i])[avxi], toDblPtr(v[h2i])[avxi], toDblPtr(v[h3i])[avxi], toDblPtr(v[r12i])[avxi]- h, toDblPtr(v[r13i])[avxi], toDblPtr(v[r23i])[avxi])) / (2 * h);
          double d5 = (tvnls(toDblPtr(v[h1i])[avxi], toDblPtr(v[h2i])[avxi], toDblPtr(v[h3i])[avxi], toDblPtr(v[r12i])[avxi], toDblPtr(v[r13i])[avxi]+ h, toDblPtr(v[r23i])[avxi]) - tvnls(toDblPtr(v[h1i])[avxi], toDblPtr(v[h2i])[avxi], toDblPtr(v[h3i])[avxi], toDblPtr(v[r12i])[avxi], toDblPtr(v[r13i])[avxi]- h, toDblPtr(v[r23i])[avxi])) / (2 * h);
          double d6 = (tvnls(toDblPtr(v[h1i])[avxi], toDblPtr(v[h2i])[avxi], toDblPtr(v[h3i])[avxi], toDblPtr(v[r12i])[avxi], toDblPtr(v[r13i])[avxi], toDblPtr(v[r23i])[avxi]+ h) - tvnls(toDblPtr(v[h1i])[avxi], toDblPtr(v[h2i])[avxi], toDblPtr(v[h3i])[avxi], toDblPtr(v[r12i])[avxi], toDblPtr(v[r13i])[avxi], toDblPtr(v[r23i])[avxi]- h)) / (2 * h);

          toDblPtr(d[h1i])[avxi] += toDblPtr(d[resi])[avxi] * d1;
          toDblPtr(d[h2i])[avxi] += toDblPtr(d[resi])[avxi] * d2;
          toDblPtr(d[h3i])[avxi] += toDblPtr(d[resi])[avxi] * d3;
          toDblPtr(d[r12i])[avxi] += toDblPtr(d[resi])[avxi] * d4;
          toDblPtr(d[r13i])[avxi] += toDblPtr(d[resi])[avxi] * d5;
          toDblPtr(d[r23i])[avxi] += toDblPtr(d[resi])[avxi] * d6;
      }
  };
private:
  ExtVarIndex h1i, h2i, h3i, r12i, r13i, r23i;
  ExtVarIndex resi;
};
// MANUAL
// MANUAL Overloaded tvnls
inline idouble tvnls(const idouble& h1, const idouble& h2, const idouble& h3, const idouble& r12, const idouble& r13, const idouble& r23) {
  idouble res;

  aadc::addConstStateExtFunction(std::make_shared<TVNLSWrapper>(res, h1,h2,h3,r12,r13,r23));

  return res;
}
#endif

