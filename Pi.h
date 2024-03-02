#pragma once 
#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>
#ifdef PARALLEL_IMPLEMENTATION
#include "aadc/aadc_eigen.h"
#include "aadc/idouble.h"
#endif
#include "types.h"
#include "OnePathProcessSimulatedData.h"
#include "utils.h"
#include "MultivariateCDF.h"


/////////////////////////////////////////////////////////////////////////////////
//
//  Pi Class // Implements Tractable Model Pricing 
//  
//  template<class AffineElementType>
//  class Pi 
// 
//  Template vals: AffineElementType = scalar, std::vector, Eigen::VectorXd
// 
//   const std::shared_ptr<std::function<vtype(const AffineElementType&)>> barrierFactor;
//   //Computes the if-knocked barrier factor  
// 
//   const std::shared_ptr<BarrierThresholds> barrier_vals;
//   //contains barrier thresholds
//
//  protected:
//  const AffineElementType sigma_m;
//  //Attribute sigma_m guranties that methods () and getVolOfAsset use the same value of sigma_m
//
/////////////////////////////////////////////////////////////////////////////////

template<class AffineElementType>
class Pi {
public:
    
    struct BarrierThresholds {
        const AffineElementType low;
        const AffineElementType upper;
    };

    Pi(
        const AffineElementType& sigma_m_
        , const std::shared_ptr<std::function<const vtype(const AffineElementType&)>>& barrierFactor_ = nullptr
        , const std::shared_ptr<BarrierThresholds>& barrier_vals_ = nullptr
        , const std::shared_ptr<std::function<const void(const AffineElementType&)>>& extrRealizedVal_ = nullptr
    ) : sigma_m(sigma_m_), barrierFactor(barrierFactor_), barrier_vals(barrier_vals_), extrRealizedVal(extrRealizedVal_)
    {}

    void moveNextTime(const Time& t) { curr_time = t; }

    virtual const AffineElementType getVolOfAsset (const AffineElementType& asset) const = 0;

    virtual vtype operator () (
        const Time& t
        , const AffineElementType& current_asset
        , const OnePathProcessSimulatedData<AffineElementType>& sim_data // Generally, there is access to all earlier time sim.data
            = 
        OnePathProcessSimulatedData<AffineElementType>() 
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
    ) const = 0;

    virtual const void reset() const = 0;

public:
    const std::shared_ptr<std::function<const vtype(const AffineElementType&)>> barrierFactor;
    const std::shared_ptr<std::function<const void(const AffineElementType&)>> extrRealizedVal;
    const std::shared_ptr<BarrierThresholds> barrier_vals;

protected:
    const AffineElementType sigma_m;
    Time curr_time=0;
};


/////////////////////////////////////////////////////////////////////////////////
//
//
///////////////////////////////////////////////////////////////////////////////////

class BlackScholes : public Pi<vtype> {
public:
    BlackScholes(
        const vtype& K_, const vtype& r_, const vtype& vol_, const Time& maturity_
    ) : Pi(vol_), K(K_), r(r_), maturity(maturity_) {}

    vtype operator () (
        const Time& t
        , const vtype& asset
        , const OnePathProcessSimulatedData<vtype>& sim_data = OnePathProcessSimulatedData<vtype>()
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
    ) const {
        vtype vol(this->sigma_m);
        Time T(maturity - t);
        if (T<0.0001) return std::max(asset-K,0.);
        vtype vol_norm = (vol * std::sqrt(T));                
        return exp(r * T) * (
            asset * std::cdf_normal(
                (std::log(asset / K) + T * (r + 0.5 * vol * vol )) / vol_norm
            ) - std::exp(-r * T) * K * std::cdf_normal(
                (std::log(asset / K) + T * (r - 0.5 * vol * vol)) / vol_norm
            )
        );
    }

    const void reset() const {}
    const vtype getVolOfAsset (const vtype& asset) const { return asset * this->sigma_m; }
    
private:
    const vtype K, r;
    const Time maturity;  
};



/////////////////////////////////////////////////////////////////////////////////
//
//
///////////////////////////////////////////////////////////////////////////////////

class BachelierAsian : public Pi<vtype> {
public:
    BachelierAsian(
        const vtype& K_, const vtype& r_, const vtype& vol_
        , const Time& maturity_, const std::vector<Time>& fixing_times_
    ) : Pi(vol_), K(K_), r(std::max(r_, 0.000001))
        , maturity(maturity_), fixing_times(fixing_times_)
    {
        int n = fixing_times.size();
        mu_coeff.resize(n+1, 0.);
        sigma_coeff1.resize(n+1, 0.);
        sigma_coeff2.resize(n+1, 0.);
        for (int k=n-1; k >= 0; k--){
            mu_coeff[k] = mu_coeff[k + 1] + std::exp(r * fixing_times[k]); 
            sigma_coeff1[k] = sigma_coeff1[k + 1] + 0.5 *std::exp(
                    r * ((fixing_times[k] + fixing_times[k]))
                );
            sigma_coeff2[k] = 0.5 + sigma_coeff2[k + 1];
            for (int l=k+1; l < n; l++){
                sigma_coeff1[k] += std::exp(
                    r * ((fixing_times[k] + fixing_times[l]))
                );
                sigma_coeff2[k] += std::exp(
                    r * std::abs((fixing_times[k] - fixing_times[l]))
                );
            }
        }
       // std::cout << "PIVOL " << this->sigma_m << std::endl;
    }
    const void reset() const {}

    vtype operator () (
        const Time& t
        , const vtype& asset
        , const OnePathProcessSimulatedData<vtype>& sim_data = OnePathProcessSimulatedData<vtype>()
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
    ) const {
        vtype vol = this->sigma_m;
        int n = fixing_times.size();
        auto pos_in_fixing_vector = std::lower_bound(fixing_times.begin(), fixing_times.end(), t);
        int k = pos_in_fixing_vector - fixing_times.begin();
        if (fixing_times_indexes.size() == 0) k=0;
        vtype average_asset(0.);
        for (int t_i = 0; t_i < k; t_i++) average_asset += sim_data.assets[fixing_times_indexes[t_i]];
        //std::cout << asset  << " , " << k << " " << mu_coeff[k] << " " << std::exp(-r * t) << " " << n << std::endl;
        vtype mu = (asset * mu_coeff[k] * std::exp(-r * t) + average_asset) / n;
        vtype r_softed = std::max(r, 0.000001);
        vtype sqrt_helper = std::max(sigma_coeff1[k] * std::exp(-2 * r_softed * t) - sigma_coeff2[k], 0.) / r_softed;
        sqrt_helper = std::max(sqrt_helper, 0.000001);
        vtype sigma = vol / n * std::sqrt(sqrt_helper);
        vtype pdf_coeff;
        vtype x = (mu - K) / sigma;
        vtype cdf_coeff = std::cdf_normal(x, pdf_coeff);
        vtype price = (mu - K) * cdf_coeff + sigma * pdf_coeff;
        
        return price;
    }

    const vtype getVolOfAsset (const vtype& asset) const { return this->sigma_m; }

private:
    const vtype K, r;
    const Time maturity;  
    const std::vector<Time> fixing_times;
    std::vector<vtype> mu_coeff, sigma_coeff1, sigma_coeff2;
};


/////////////////////////////////////////////////////////////////////////////////
//
//
///////////////////////////////////////////////////////////////////////////////////

class BlackLookbackCall : public Pi<vtype> {
public:
    BlackLookbackCall(
        const vtype& r_, const vtype& vol_
        , const Time& maturity_, const vtype& init_asset
        ) : Pi(
            vol_, nullptr, nullptr
            , std::make_shared<std::function<const void(const vtype&)>>(
                [this](const vtype& asset) { *minS = std::min(asset, *minS); }
            )
        ), r(std::max(r_, 0.05))
        , maturity(maturity_)    
        , d_p((r + this->sigma_m * this->sigma_m / 2))
        , d_m((r - this->sigma_m * this->sigma_m / 2))
        , minS(std::make_shared<vtype>(init_asset))
        {}

    const void reset() const { *minS = 100; }

    vtype operator () (
        const Time& t
        , const vtype& asset
        , const OnePathProcessSimulatedData<vtype>& sim_data = OnePathProcessSimulatedData<vtype>()
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
        ) const {
        vtype vol = this->sigma_m;
        Time T(maturity - t);
        T = std::max(0.001, T);
        //if (t < 0.01) *minS = 100; 
        vtype used_min = *minS; // std::min(asset, *minS);
        vtype x = asset - *minS;
        vtype eps = 0.1;
        //vtype add = -4. / 5 / eps * x * x + 1. / 5 / eps / eps * x * x * x;
        //used_min = iIf((asset + eps - *minS)* (asset - *minS)>0, used_min,  *minS + add);

   
        vtype sq_t = std::sqrt(T);
        vtype c1 = std::log(asset / used_min);
        vtype div = vol * sq_t;
        vtype a1 = (c1 + d_p * T) / div;
        vtype a2 = (c1 + d_m * T) / div;
        vtype a3 = (c1 - d_m * T) / div;

        vtype pdf_coeff;
        vtype cdf_coeff1 = std::cdf_normal(a1, pdf_coeff);
        vtype cdf_coeff2 = std::cdf_normal(a2, pdf_coeff);
        vtype cdf_coeff3 = std::cdf_normal(-a1, pdf_coeff);
        vtype cdf_coeff4 = std::cdf_normal(-a3, pdf_coeff);

        vtype disc = exp(r * T);
        vtype price = 
            disc * asset * cdf_coeff1 
            - used_min * cdf_coeff2
           - disc * asset * vol * vol / 2 / r * (
              cdf_coeff3 - std::exp(-c1 * 2 * r / vol / vol - r * T) * cdf_coeff4
           )
        ;
        //price += asset * ((-c1 - T * vol * vol / 2) * cdf_coeff3 + pdf_coeff * vol * sq_t);
        
        return price;
    }

    const vtype getVolOfAsset(const vtype& asset) const { return asset * this->sigma_m; }

private:
    const std::shared_ptr<vtype> minS;
    const vtype r;
    const Time maturity;
    const vtype d_p, d_m;
};





/////////////////////////////////////////////////////////////////////////////////
//
//
///////////////////////////////////////////////////////////////////////////////////

class DownAndOutEuropCallPrice : public Pi<vtype> {
public:
    DownAndOutEuropCallPrice(
        const vtype& K_, const vtype& r_, const vtype& barrier_, const vtype& vol_, const Time& maturity_
        , const std::vector<Time>& simulation_times_
    ) : 
        Pi(
            vol_
            , std::make_shared<std::function<const vtype(const vtype&)>>(
               [this] (const vtype& asset) { return stepApproximation(asset - this->barrier_vals->low, 0.001);}
            )
            , std::make_shared<Pi::BarrierThresholds>(
                Pi::BarrierThresholds({barrier_ * exp(-0.5826 * vol_ * std::sqrt(maturity_ / simulation_times_.size())) ,-1})
            )
        )
        , K(K_), r(r_)
        , maturity(maturity_)
        , simulation_times(simulation_times_)
        , BS(std::make_shared<BlackScholes>(K, r, this->sigma_m, maturity)) 
        , kappa(2 * r / (this->sigma_m * this->sigma_m))
        , barrier(this->barrier_vals->low)
    {}

    const void reset() const {}

    vtype operator () (
        const Time& t
        , const vtype& asset
        , const OnePathProcessSimulatedData<vtype>& sim_data = OnePathProcessSimulatedData<vtype>()
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
    ) const {
        return  
            (*BS)(t, asset) 
            - pow(asset / barrier, 1 - kappa) * (*BS)(t, barrier * barrier / asset)
        ; 
    }
    const vtype getVolOfAsset (const vtype& asset) const { return asset * this->sigma_m; }

private:
    const vtype K, r, barrier, kappa;
    const Time  maturity; 
    const std::shared_ptr<BlackScholes> BS;
    const std::vector<Time> simulation_times;
};



/////////////////////////////////////////////////////////////////////////////////
//
//
///////////////////////////////////////////////////////////////////////////////////

class BasketBachelier : public Pi<Eigen::vVectorXd> {
public:
    BasketBachelier(
        const vtype& K_, const vtype& r_, const Time& maturity_
        , const Eigen::vVectorXd& sigma_m_vec, const Eigen::MatrixXd& corr_mat
    ) : Pi(sigma_m_vec), K(K_), r(r_), maturity(maturity_)
        , vol(std::sqrt((this->sigma_m.transpose() * corr_mat * this->sigma_m)(0, 0)))
    {}

    const void reset() const {}
    
    vtype operator () (
        const Time& t
        , const Eigen::vVectorXd& asset
        , const OnePathProcessSimulatedData<Eigen::vVectorXd>& sim_data = OnePathProcessSimulatedData<Eigen::vVectorXd>()
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
        ) const {
        
        vtype T(maturity - t);
        if (T < 1e-6) return 0;

        vtype mean = asset.mean();
        vtype sigma_basket = sqrt(T) * vol / asset.size();
        if (r > 1e-5) {
            sigma_basket *= std::sqrt((std::exp(2 * r * T) - 1) / (2 * r * T));
        }
        vtype av = mean * std::exp(r * T) - K;
        vtype z_adj = av / sigma_basket;
        vtype pdf;
        vtype cdf = std::cdf_normal(z_adj, pdf);
        return (av * cdf + sigma_basket * pdf) ;
    }

    const Eigen::vVectorXd getVolOfAsset(const Eigen::vVectorXd& asset) const { return this->sigma_m; }

private:
    const vtype K, r, vol;
    const Time maturity;
};


/////////////////////////////////////////////////////////////////////////////////
//
//
///////////////////////////////////////////////////////////////////////////////////

class RainbowCallMaxBlack : public Pi<Eigen::vVectorXd> {
public:
    RainbowCallMaxBlack(
        const vtype& K_, const vtype& r_, const Time& maturity_
        , const Eigen::vVectorXd& sigma_m_vec, const Eigen::MatrixXd& corr_mat
    ) : Pi(sigma_m_vec), K(K_), r(r_), maturity(maturity_) {
        corrs_2 = Eigen::vMatrixXd(4, 4);
        corrs = Eigen::vMatrixXd::Zero(4, 4);
        sigma_2 = Eigen::vMatrixXd::Zero(4, 4);

        sigma.resize(4);
        for (int i = 0; i < 3; i++) sigma[i] = sigma_m_vec[i];
        sigma[3] = 0;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                corrs(i, j) = corr_mat(i, j);
            }
        }

        for (int k=0; k<4; k++) {
            for (int i = 0; i < 4; i++) {
                sigma_2(k, i) = std::sqrt(sigma[k] * sigma[k] + sigma[i] * sigma[i] - 2 * corrs(k, i) * sigma[k] * sigma[i]);
                for (int j = 0; j < 4; j++) {
                    {
                        vtype numerator =
                            corrs(i, j) * sigma[i] * sigma[j]
                            - corrs(i, k) * sigma[i] * sigma[k]
                            - corrs(k, j) * sigma[k] * sigma[j] + sigma[k] * sigma[k]
                            ;
                        vtype denom_sq = (sigma[i] * sigma[i] + sigma[k] * sigma[k] - 2 * corrs(i, k) * sigma[i] * sigma[k])
                            * (sigma[j] * sigma[j] + sigma[k] * sigma[k] - 2 * corrs(j, k) * sigma[j] * sigma[k])
                            ;
                        corrs_2(i, j) = numerator / sqrt(denom_sq);
                    }
                }
            }
            if (k<3) corrs_2_1.push_back(corrs_2);
        }
        corrs_2_1.push_back(corr_mat);

    }
    const void reset()const {}

    vtype operator () (
        const Time& t
        , const Eigen::vVectorXd& asset
        , const OnePathProcessSimulatedData<Eigen::vVectorXd>& sim_data = OnePathProcessSimulatedData<Eigen::vVectorXd>()
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
        ) const {
        vtype time_to_exp(maturity - t);
        if (time_to_exp == 0) return 0;

        Eigen::vVectorXd d_plus(3), d_minus(3);
        Eigen::vMatrixXd d_2_plus(3, 3), d_2_minus(3, 3);
        for (int i=0; i<3; i++ ) {
            d_plus[i] = (std::log(asset[i] / K) + 0.5 * sigma[i]* sigma[i] * time_to_exp) / (sigma[i] * sqrt(time_to_exp));
            d_minus[i] = d_plus[i] - sigma[i] * sqrt(time_to_exp);

            for (int j=0; j<3; j++) {
                d_2_plus(i, j) = (log(asset[i] / asset[j]) + 0.5 * sigma_2(i, j) * sigma_2(i, j) * time_to_exp) / (
                    sigma_2(i, j)* sqrt(time_to_exp)
                );
                d_2_minus(i, j) = d_2_plus(i, j) - sigma_2(i, j) * sqrt(time_to_exp);
            }
        }
        vtype comp_1 = asset[0] * tvnls(
            -d_2_minus(1, 0), -d_2_minus(2, 0), d_plus[0], corrs_2_1[0](1, 2)
            , corrs_2_1[0](1, 3), corrs_2_1[0](2, 3)
        );

        vtype comp_2 = asset[1] * tvnls(
            -d_2_minus(0, 1), -d_2_minus(2, 1), d_plus[1], corrs_2_1[1](0, 2)
            , corrs_2_1[1](0, 3), corrs_2_1[1](2, 3)
        );

        vtype comp_3 = asset[2] * tvnls(
            -d_2_minus(0, 2), -d_2_minus(1, 2), d_plus[2], corrs_2_1[2](0, 1)
            , corrs_2_1[2](0, 3), corrs_2_1[2](1, 3)
        );
            
        vtype comp_4 = K * (
            1 - tvnls(
                -d_minus[0], -d_minus[1], -d_minus[2], corrs_2_1[3](0, 1)
                , corrs_2_1[3](0, 2), corrs_2_1[3](1, 2)
            ) 
        );

        vtype pi_val = comp_1 + comp_2 + comp_3 - comp_4;

        return pi_val;
    }

    const Eigen::vVectorXd getVolOfAsset(const Eigen::vVectorXd& asset) const { return asset.cwiseProduct(this->sigma_m); }

private:
    const vtype K, r;
    const Time maturity;
    Eigen::vVectorXd sigma;
    Eigen::vMatrixXd corrs_2, sigma_2, corrs;
    std::vector< Eigen::vMatrixXd> corrs_2_1;

};
