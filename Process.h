#pragma once 
#ifdef PARALLEL_IMPLEMENTATION
#include "aadc/aadc_matrix.h"
#include <aadc/aadc_eigen.h>
#endif
#include <memory>
#include <cmath>

#include <Eigen/Dense>
using Eigen::Dynamic;
using Eigen::Matrix;

#include "src/Core/Matrix.h"
#include "src/Core/util/Constants.h"
#include "src/Eigenvalues/SelfAdjointEigenSolver.h"
#include "types.h"
#include "CorrelatedBM.h"
//#include "MultivariateCDF.h"


/////////////////////////////////////////////////////////////////////////////////
//
//  Root Process Class
//
//  template<class AffineElementType>
//  class Process 
//  Template classname  AffineElementType = scalar, std::vector, Eigen::VectorXd
//  Implements a stochastic process interface
//
//  Constructor:
//  Process (const std::shared_ptr<CorrelatedBM> asset_bm_ = std::make_shared<CorrelatedBM>());
//  asset_bm = implements BrownianMotion-correlation; is trivial in the 1D case.
// 
/////////////////////////////////////////////////////////////////////////////////

template<class AffineElementType>
class Process {
public:
    Process (
        const std::shared_ptr<CorrelatedBM> asset_bm_ = std::make_shared<CorrelatedBM>()
    ) : asset_bm(asset_bm_) {} 
    virtual void simulateAssetOneStep(
        const Time& next_t, const std::vector<Eigen::vVectorXd>& random_sample
    ) = 0; 
    virtual const int getRandomVectorsPerTimePoint () const = 0;
    virtual const int getRandomVectorDim () const = 0;
    virtual const AffineElementType getVolOfAsset () const = 0;
    virtual const AffineElementType getCurrentAsset () const = 0; 
    virtual void reset() = 0;
    
    const std::shared_ptr<CorrelatedBM> getAssetBM() const { return asset_bm; }

    void init(
        const Time& init_t_
        , const AffineElementType& init_asset_
        , AffineElementType& init_vol_
    ) {
        init_t = init_t_;
        init_asset = init_asset_;
        init_vol = init_vol_;
    }

    void bump(const double& eps, const int& i = 0) { 
        AffineElementType shocked_vec(AffineElement::getI(i, eps, init_asset));
        init(init_t, init_asset + shocked_vec, init_vol);
    }

    const AffineElementType& getInitAsset () const { return init_asset; }

protected:
    Time current_t;
    AffineElementType curr_asset, curr_vol;

    Time init_t; 
    AffineElementType init_asset, init_vol;
    const std::shared_ptr<CorrelatedBM> asset_bm;
};


/////////////////////////////////////////////////////////////////////////////////
//
//  Heston
//
//
/////////////////////////////////////////////////////////////////////////////////

class Heston : public Process<vtype> {
public: 
    Heston(
        const vtype& rate_, const vtype& rho_, const vtype& theta_
        , const vtype& kappa_, const vtype& vol_vol_
    ) : Process(), rate(rate_), rho(rho_), theta(theta_), kappa(kappa_), vol_vol(vol_vol_) 
    {}

    void reset() {
        this->current_t = this->init_t;
        this->curr_vol = this->init_vol;
        this->curr_asset = 1.;
    }

    const int getRandomVectorsPerTimePoint () const { return 2; }
    const int getRandomVectorDim () const { return 1; }
    const vtype getCurrentAsset () const { return this->curr_asset * this->init_asset; }

    const vtype getVolOfAsset () const {
        return  getCurrentAsset() * std::sqrt(this->curr_vol);
    };
   
    void simulateAssetOneStep(
        const Time& next_t
        , const std::vector<Eigen::vVectorXd>& random_sample
    ) {
        vtype corr_random_sample = 
            random_sample[0][0] * rho + std::sqrt(1 - rho * rho) * random_sample[1][0]
        ;
        double dt = (next_t - this->current_t);
        this->current_t = next_t;
        this->curr_asset = 
            this->curr_asset 
            * (1 + rate * dt + std::sqrt(this->curr_vol * dt) * random_sample[0][0])
        ;
        this->curr_vol += 
            kappa * (theta - this->curr_vol) * dt + vol_vol 
            * std::sqrt(this->curr_vol * dt) * corr_random_sample
        ;
        this->curr_vol = std::max(this->curr_vol, 0.0001);
    }

private:
    const vtype rate, rho, theta, kappa, vol_vol;
};



/////////////////////////////////////////////////////////////////////////////////
//
//  Sabr OneDim
//
//
/////////////////////////////////////////////////////////////////////////////////

class SabrScalar : public Process<vtype> {
public:
    SabrScalar(const vtype& vol_vol_) : Process(), vol_vol(vol_vol_)
    {}

    void reset() {
        this->current_t = this->init_t;
        this->curr_vol = this->init_vol;
        this->curr_asset = this->init_asset;
    }

    const int getRandomVectorsPerTimePoint() const { return 2; }
    const int getRandomVectorDim() const { return 1; }
    const vtype getCurrentAsset() const { return this->curr_asset; }

    const vtype getVolOfAsset() const {
        return std::sqrt(getCurrentAsset()) * this->curr_vol;
    };

    void simulateAssetOneStep(
        const Time& next_t
        , const std::vector<Eigen::vVectorXd>& random_sample
    ) {
        double dt = (next_t - this->current_t);
        this->current_t = next_t;
        
       

        vtype aux_s = std::sqrt(this->curr_asset) * this->curr_vol;
        vtype millst = 0.5 * 0.5 * aux_s * aux_s / this->curr_asset * (random_sample[0][0] * random_sample[0][0] - 1) * dt;
        this->curr_asset += aux_s * std::sqrt(dt) * random_sample[0][0] + millst;
        this->curr_asset = std::max(this->curr_asset, vtype(0.01));

        //this->curr_asset += std::sqrt(this->curr_asset * dt) * this->curr_vol * random_sample[0][0];
        this->curr_vol *= std::exp(vol_vol * std::sqrt(dt) * random_sample[1][0] - vol_vol * vol_vol * 0.5 * dt);


    }

private:
    const vtype vol_vol;
};

/////////////////////////////////////////////////////////////////////////////////
//
//  SABR MultiDim
//
//
/////////////////////////////////////////////////////////////////////////////////

class SABR : public Process<Eigen::vVectorXd> {
public: 
    SABR(
         const Eigen::vVectorXd& vol_vol_
         , const Eigen::Matrix<double, Dynamic, Dynamic>& corr_matrix
    ) : Process(std::make_shared<CorrelatedBM>(corr_matrix))
        , vol_vol(vol_vol_), dim(vol_vol_.size())
    {}

    void reset() {
        this->current_t = this->init_t;
        this->curr_vol = this->init_vol;
        this->curr_asset = this->init_asset; // Eigen::vVectorXd::Ones(dim);
    }

    const int getRandomVectorsPerTimePoint () const { return 2; }
    const int getRandomVectorDim () const { return dim; }
    const Eigen::vVectorXd getCurrentAsset () const { return this->curr_asset; }

    const Eigen::vVectorXd getVolOfAsset () const {
        return getCurrentAsset().cwiseSqrt().cwiseProduct(this->curr_vol);
    };

    void simulateAssetOneStep(
        const Time& next_t
        , const std::vector<Eigen::vVectorXd>& random_sample
    ) {
        double dt = next_t - this->current_t;
        this->current_t = next_t;

        Eigen::vVectorXd dWt = 
            asset_bm->getCorrelated(random_sample[0]) * std::sqrt(dt) 
        ;
        this->curr_asset += dWt.cwiseProduct(this->curr_asset.cwiseSqrt().cwiseProduct(this->curr_vol));
        this->curr_asset = this->curr_asset.cwiseMax(0.0001);

        this->curr_vol += this->curr_vol.cwiseProduct(
            vol_vol.cwiseProduct(random_sample[1] * std::sqrt(dt))
        );
    }

private:
    const int dim;
    const Eigen::vVectorXd vol_vol;
};


/////////////////////////////////////////////////////////////////////////////////
//
//  Heston MultiDim
//
//
/////////////////////////////////////////////////////////////////////////////////

class HestonMultDim : public Process<Eigen::vVectorXd> {
public:
    HestonMultDim(
        const Eigen::vVectorXd& vol_vol_
        , const Eigen::vVectorXd& rate_
        , const Eigen::vVectorXd& kappa_
        , const Eigen::vVectorXd& theta_
        , const Eigen::vVectorXd& rho_
        , const Eigen::Matrix<double, Dynamic, Dynamic>& corr_matrix
    ) : Process(std::make_shared<CorrelatedBM>(corr_matrix))
        , vol_vol(vol_vol_), rate(rate_), kappa(kappa_), theta(theta_), rho(rho_), dim(vol_vol_.size())
    {}

    void reset() {
        this->current_t = this->init_t;
        this->curr_vol = this->init_vol;
        this->curr_asset = Eigen::vVectorXd::Ones(dim);
    }

    const int getRandomVectorsPerTimePoint() const { return 2; }
    const int getRandomVectorDim() const { return dim; }
    const Eigen::vVectorXd getCurrentAsset() const {
        return curr_asset.cwiseProduct(this->init_asset);
    }

    const Eigen::vVectorXd getVolOfAsset() const {
        return getCurrentAsset().cwiseProduct(this->curr_vol.cwiseSqrt());
    };

    void simulateAssetOneStep(
        const Time& next_t
        , const std::vector<Eigen::vVectorXd>& random_sample
    ) {
        double dt = next_t - this->current_t;
        this->current_t = next_t;

        Eigen::vVectorXd dWt = asset_bm->getCorrelated(random_sample[0]);

        Eigen::vVectorXd corr_random_sample =
            dWt.cwiseProduct(rho) + 
            (Eigen::vVectorXd::Ones(dim) - rho.cwiseProduct(rho)).cwiseSqrt().cwiseProduct(random_sample[1])
        ;

        this->curr_asset += this->curr_asset.cwiseProduct(
            rate * dt + 
            dWt.cwiseProduct( (dt * this->curr_vol).cwiseSqrt())
        ) ;
        this->curr_vol += 
            kappa.cwiseProduct(theta - this->curr_vol) * dt 
            +  
            vol_vol.cwiseProduct(corr_random_sample.cwiseProduct((this->curr_vol * dt).cwiseSqrt() ))
        ;

        this->curr_vol = this->curr_vol.cwiseMax(0.0001);
    }

private:
    const int dim;
    const Eigen::vVectorXd vol_vol, rate, kappa, theta, rho;
};