#pragma once
#ifdef PARALLEL_IMPLEMENTATION
#include "aadc/aadc_eigen.h"
#endif
#include "types.h"
#include <algorithm>
#include "OnePathProcessSimulatedData.h"

/////////////////////////////////////////////////////////////////////////////////
//
//  Root Payoff class
//
//  template<class AffineElementType>
//  class Payoff
//
//  Template classname  AffineElementType = scalar, std::vector, Eigen::VectorXd
//  Does it required for a convential MC only?
//
//////////////////////////////////////////////////////////////////////////////

template<class AffineElementType>
class Payoff {
public:
    virtual vtype operator () (
        const AffineElementType& asset
        , const OnePathProcessSimulatedData<AffineElementType>& sim_data 
            = OnePathProcessSimulatedData<AffineElementType>()
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
    ) const = 0;
};


/////////////////////////////////////////////////////////////////////////////////
//
//
//
//////////////////////////////////////////////////////////////////////////////

class EuropeanCall : public Payoff<vtype> {
public:
    EuropeanCall (
        const vtype& K_
        , const vtype& r_
        , const Time& T_ ) 
    : K(K_), r(r_), T(T_) 
    {}

    vtype operator () (
        const vtype& asset
        , const OnePathProcessSimulatedData<vtype>& sim_data = OnePathProcessSimulatedData<vtype>()
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
    ) const {
        return std::max(asset - K, vtype(0.));
    }

private:
    const vtype K, r;
    const Time T;
};


/////////////////////////////////////////////////////////////////////////////////
//
//
//
//////////////////////////////////////////////////////////////////////////////

class EuropeanCallAsian : public Payoff<vtype> {
public:
    EuropeanCallAsian (const vtype& K_, const vtype& r_, const Time& T_ ) 
    : K(K_), r(r_), T(T_) 
    {}

    vtype operator () (
        const vtype& asset
        , const OnePathProcessSimulatedData<vtype>& sim_data = OnePathProcessSimulatedData<vtype>()
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
    ) const {
        vtype av(0.);
        int n = fixing_times_indexes.size();
        for (int t_i=0; t_i<n; t_i++) {
            av += sim_data.assets[fixing_times_indexes[t_i]];
        }
        std::cout << "AAAAA " << n << std::endl;
        return std::max(av / n - K, vtype(0.));
    }

private:
    const vtype K, r;
    const Time T;
};



/////////////////////////////////////////////////////////////////////////////////
//
//
//
//////////////////////////////////////////////////////////////////////////////

class LookbackCall : public Payoff<vtype> {
public:
    LookbackCall()
    {}

    vtype operator () (
        const vtype& asset
        , const OnePathProcessSimulatedData<vtype>& sim_data = OnePathProcessSimulatedData<vtype>()
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
        ) const {
        vtype s_min(asset);
        int n = sim_data.assets.size();
        for (int t_i = 0; t_i < n; t_i++) {
            s_min = std::min(s_min, sim_data.assets[t_i]);
        }
        return asset - s_min;
    }

private:
};


/////////////////////////////////////////////////////////////////////////////////
//
//
//
//////////////////////////////////////////////////////////////////////////////

class DownAndOutEuropCallPayoff : public Payoff<vtype> {
public:
    DownAndOutEuropCallPayoff (const vtype& K_, const vtype& B_, const Time& T_) 
    : K(K_), B(B_), T(T_) 
    {}

    vtype operator () (
        const vtype& asset
        , const OnePathProcessSimulatedData<vtype>& sim_data 
            = OnePathProcessSimulatedData<vtype>()
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
    ) const {
        vtype barrier_cond(1.);
        int n = sim_data.assets.size();
        for (int t_i=0; t_i < n; t_i++) {
            #ifdef PARALLEL_IMPLEMENTATION
            //barrier_cond *= iIf(sim_data.assets[t_i] - B > 0., 1., 0.);
            barrier_cond *= stepApproximation(sim_data.assets[t_i] - B , 0.001);
            #else
            barrier_cond *= sim_data.assets[t_i] - B > 0. ? 1. : 0.;
            #endif
        }
        return barrier_cond * std::max(asset  - K, vtype(0.));      
    }

private:
    const vtype K, B;
    const Time T;
};



/////////////////////////////////////////////////////////////////////////////////
//
//
//
//////////////////////////////////////////////////////////////////////////////

class BasketCall : public Payoff<Eigen::vVectorXd> {
public:
    BasketCall (
        const vtype& K_,const Time& maturity
    ) : K(K_), T(maturity) 
    {}

    vtype operator () (
        const Eigen::vVectorXd& asset
        , const OnePathProcessSimulatedData<Eigen::vVectorXd>& sim_data 
            = OnePathProcessSimulatedData<Eigen::vVectorXd>
        ()
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
    ) const {
        return std::max(asset.mean() - K, vtype(0.));
    }

private:
    const Time T;
    const vtype K;
};



/////////////////////////////////////////////////////////////////////////////////
//
//
//
//////////////////////////////////////////////////////////////////////////////

class RainbowCallOnMax : public Payoff<Eigen::vVectorXd> {
public:
    RainbowCallOnMax(
        const vtype& K_, const Time& maturity
    ) : K(K_), T(maturity)
    {}

    vtype operator () (
        const Eigen::vVectorXd& asset
        , const OnePathProcessSimulatedData<Eigen::vVectorXd>& sim_data
        = OnePathProcessSimulatedData<Eigen::vVectorXd>
        ()
        , const std::vector<int>& fixing_times_indexes = std::vector<int>()
        ) const {
        return std::max(asset.maxCoeff() - K, vtype(0.));
    }

private:
    const Time T;
    const vtype K;
};

