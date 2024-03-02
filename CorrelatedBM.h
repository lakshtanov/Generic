#pragma once 
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

/////////////////////////////////////////////////////////////////////////////////
//
//  Class CorrelatedBM
//  Input: const Eigen::Matrix<double, Dynamic, Dynamic>& corr_matrix
//
//  Method     Eigen::vVectorXd  correlated(const Eigen::vVectorXd& random_sample) const
//             //implements matrix-vector multiplication sqrt(corr_matrix) * randoms_sample 
//
//  Method     std::vector<Eigen::vVectorXd> getVolTimesCorrRankOneRepresentation (const Eigen::vVectorXd& vols) const
//             std::vector<vtype> getVolTimesCorrRankOneRepresentation (vtype vols) const 
//             // Impelements representation (See page. article)
//
//  Method     const int corrMatrixActualRank () const { return corr_ass_eigen_vectors.size(); }
//            // Rank of corr matrix
//
//  Attributes: 
//          std::vector<Eigen::vVectorXd> corr_ass_eigen_vectors;   Eigen vectors
//          Eigen::vVectorXd corr_ass_eigen_vals;       Eigenvalues^{1/4}
//
/////////////////////////////////////////////////////////////////////////////////

class CorrelatedBM {
public:
    friend class MultivariateCDF;
    CorrelatedBM () 
    : corr_ass_eigen_vectors({Eigen::vVectorXd::Ones(1)}) 
    , corr_ass_eigen_vals(Eigen::vVectorXd::Ones(1))
    , dim(1)
    {}
      
    CorrelatedBM (const Eigen::Matrix<double, Dynamic, Dynamic>& corr_matrix) :
        dim(corr_matrix.rows()) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Dynamic, Dynamic>> es(corr_matrix);
        std::cout 
            << "Cut - off is not implemented.Just store less elements in CorrelatedBM.eigenvectors"
            << " and CorrelatedBM.corr_ass_eigen_vals data members" 
        << std::endl;
        Eigen::vMatrixXd eigenvectors = es.eigenvectors();
        corr_ass_eigen_vals  = es.eigenvalues().cwiseSqrt();
        corr_ass_eigen_vals = corr_ass_eigen_vals.cwiseSqrt();
        for (int i=0; i < corr_matrix.col(0).size(); i++) {
            corr_ass_eigen_vectors.push_back(eigenvectors.col(i) * corr_ass_eigen_vals[i]);
        }
    }

    // X -> sqrt(Corr.Matrix) X
    const Eigen::vVectorXd getCorrelated(const Eigen::vVectorXd& random_sample) const {
        Eigen::vVectorXd asset_increment(Eigen::vVectorXd::Zero(random_sample.size()));
        for (int i=0; i < corr_ass_eigen_vectors.size(); i++) {  
            asset_increment += this->corr_ass_eigen_vectors[i] * (this->corr_ass_eigen_vectors[i].transpose() * random_sample); 
        }
        return asset_increment;
    }

    // vols -> e_i vols[i] - See. formula ? in the article
    std::vector<Eigen::vVectorXd> getVolTimesCorrRankOneRepresentation (const Eigen::vVectorXd& vols) const {
        std::vector<Eigen::vVectorXd> return_vect;
        for (int i=0; i < corr_ass_eigen_vectors.size(); i++) {
            return_vect.push_back(
                corr_ass_eigen_vectors[i].cwiseProduct(vols) * corr_ass_eigen_vals[i]
            );
        }
        return return_vect;
    }

    const std::vector<vtype> getVolTimesCorrRankOneRepresentation (vtype vols) const { 
        return std::vector<vtype>(1, vols); 
    }

    const int getDim() const { return dim; }
    const int corrMatrixActualRank () const { return corr_ass_eigen_vectors.size(); }

private:
    std::vector<Eigen::vVectorXd> corr_ass_eigen_vectors;
    Eigen::vVectorXd corr_ass_eigen_vals;
    const int dim;
};