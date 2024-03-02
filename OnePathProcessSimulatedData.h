#pragma once
#include <memory>
#include "CorrelatedBM.h"
#include "types.h"


////////////////////////////////////////////////////////////////////////////////////
//
//   template<class AffineElementType> struct OnePathProcessSimulatedData 
//   At each time point these vectors provide a set of one-rank vol-matrices
// 
////////////////////////////////////////////////////////////////////////////////////

template<class AffineElementType>
struct OnePathProcessSimulatedData {

    OnePathProcessSimulatedData() {};
       
    //  CorrelatedBM
    std::shared_ptr<CorrelatedBM>  asset_bms;

    // time x AffineElementType
    std::vector<AffineElementType> assets, vols;
    vtype min_realized_val;
};
