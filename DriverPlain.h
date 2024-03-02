#pragma once 
#include <algorithm>
#include <locale>
#include <typeinfo>
#include <random>
#include <chrono>

#include "types.h"
#include "Process.h"
#include "Pi.h"
#include "Payoff.h"
#include "QuadratureCoefficients.h"
#include "OnePathProcessSimulatedData.h"
#include "utils.h"
#include "Results.h"


////////////////////////////////////////////////////////////////////////////////////
//
//  Pricing Driver Class.
//          
//          DriverPlain(std::function<std::vector<QuadratureCoefficients>&()>& getLegendre_)
//          // Inputs: getLaguerre20, getLaguerre64
// 
//          Methods:
//          void simulateProcess;  Get Process and sigma_model parameter. Returns a box of simulated data
// 
//          
//          double compute(...);  // Get pi-function and simulated_data. Returns price.
//          double onePathcomputePriceFromSimulatedData(...) // Compute Den
// 
//////////////////////////////////////////////////////////////////////////////////////

class DriverPlain {
public:
    DriverPlain(std::function<std::vector<QuadratureCoefficients>&()>& getLegendre_)
    : getLegendre(getLegendre_) 
    {}

    /////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  Implements process simulation for one MC-path.
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////
    template<class AffineElementType>
    void onePathSimulateProcessData (
        const std::vector<Time>& simulation_times                                //:I
        , const std::vector<int>& legendre_absciss_indexes                       //:I
        , const std::vector<std::vector<Eigen::vVectorXd>>& random_samples       //:I
        , Process<AffineElementType>& process                                    //
        , OnePathProcessSimulatedData<AffineElementType>& simulated_data         //:O
    ) {
        double one_path_impact = double(0.);
        double quadrature = double(0.);
        double maturity(simulation_times.back());
        double t0(simulation_times.front());
        simulated_data.asset_bms = process.getAssetBM();

        for (int t_i = 0; t_i < simulation_times.size(); ++t_i) {
            process.simulateAssetOneStep(simulation_times[t_i], random_samples[t_i]);
            auto legendre_position = std::find(
                legendre_absciss_indexes.begin()
                , legendre_absciss_indexes.end()
                , t_i
            );
            simulated_data.assets.emplace_back(process.getCurrentAsset());
            simulated_data.vols.emplace_back(process.getVolOfAsset());
        } 
    }

    template<class AffineElementType>
    double onePathcomputePriceFromSimulatedData (
        const std::vector<Time>& simulation_times                                //:I
        , const std::vector<int>& legendre_absciss_indexes                       //:I
        , const Pi<AffineElementType>& pi_func                                   //:I
        , const OnePathProcessSimulatedData<AffineElementType>& sim_data         //:I
        , const std::vector<int>& fixing_times_indexes                           //:I
    ) {
        double one_path_impact = double(0.);
        double quadrature = double(0.);
        double maturity(simulation_times.back());
        double t0(simulation_times.front());

        // Compute barrier adjustment factor
        double accumulated = 1.;
        if (pi_func.barrierFactor) {
            for (int i=0; i<sim_data.assets.size(); i++) {
                accumulated *= (*pi_func.barrierFactor)(sim_data.assets[i]);
            }
        }

        AffineElementType curr_asset_p, curr_asset_m;    
        for (int i = 0; i < legendre_absciss_indexes.size(); ++i) {
            double eps=0.01, gamma_ass(0.), gamma_base(0.);

            int t_i = legendre_absciss_indexes[i];
            Time current_t = simulation_times[t_i];

            if (pi_func.extrRealizedVal) {
                (*pi_func.extrRealizedVal)(sim_data.assets[t_i]);
            }


            std::vector<AffineElementType> vol_corr_vect1(
                std::move(
                    sim_data.asset_bms->getVolTimesCorrRankOneRepresentation(sim_data.vols[t_i])
                )
            );
            std::vector<AffineElementType> vol_corr_vect2(
                std::move(
                    sim_data.asset_bms->getVolTimesCorrRankOneRepresentation(pi_func.getVolOfAsset(sim_data.assets[t_i]))
                )
            );

            for (int i=0; i < sim_data.asset_bms->corrMatrixActualRank(); i++) {
                curr_asset_p = sim_data.assets[t_i] + eps * vol_corr_vect1[i];
                curr_asset_m = sim_data.assets[t_i] - eps * vol_corr_vect1[i];
                gamma_ass += pi_func(
                    current_t, curr_asset_p
                    , sim_data, fixing_times_indexes
                ) + pi_func(
                    current_t, curr_asset_m
                    , sim_data, fixing_times_indexes
                );
                curr_asset_p = sim_data.assets[t_i] + eps * vol_corr_vect2[i];
                curr_asset_m = sim_data.assets[t_i] - eps * vol_corr_vect2[i];
                gamma_base += pi_func(
                    current_t, curr_asset_p
                    , sim_data, fixing_times_indexes
                ) + pi_func(
                    current_t, curr_asset_m
                    , sim_data, fixing_times_indexes
                );
            } 
            quadrature += 
                accumulated * (gamma_ass - gamma_base) / (eps * eps) * getLegendre()[i].weight
            ;             
        } 
        return (quadrature / 2 * (maturity - t0));
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    template<class ProcessType, class AffineElementType>
    void simulateProcess(
        const std::vector<Time>& simulation_time_steps
        , std::vector<int>& legendre_absciss_indexes    // Will not be in use             //:O
            // it will be recalculated for each option separately due to different possible maturity values
        , std::vector<OnePathProcessSimulatedData<AffineElementType>>& simulate_data_mc   //:O
        , ProcessType& process
        , const AffineElementType& sigma_m
        , const int num_mc_paths 
    ){
        std::mt19937_64 gen(17);
        std::normal_distribution<> normal_distrib(0.0, 1.0);            

        std::cout << "Num.MC-paths: " << num_mc_paths << "\n";
        auto time_start= std::chrono::high_resolution_clock::now();

        std::size_t num_time_steps = simulation_time_steps.size();
        Time t0 = simulation_time_steps.front();
        Time T = simulation_time_steps.back();

        // Compute indexes of simulation_time_steps that corresponds to  Legendre points
        for (int i=0; i < getLegendre().size(); i++) {
            auto simul_nearest_point = std::lower_bound(
                simulation_time_steps.begin(), simulation_time_steps.end()
                , (T-t0) / 2 * (getLegendre()[i].abscissa + 1) + t0
            );
            legendre_absciss_indexes.push_back(simul_nearest_point - simulation_time_steps.begin());
        }

      
        // Prepare container for random numbers
        std::vector<std::vector<Eigen::vVectorXd>> one_path_randoms(num_time_steps);
        for (int i=0; i < one_path_randoms.size(); i++) {
            one_path_randoms[i].resize(process.getRandomVectorsPerTimePoint()) ;
            for (int j=0; j < process.getRandomVectorsPerTimePoint(); j++) {
                one_path_randoms[i][j].resize(process.getRandomVectorDim());
            }
        }
        // MC starts here 
        for (int mc_i=0; mc_i<num_mc_paths; mc_i++) {
            for (int fix_date_i=0; fix_date_i < num_time_steps; fix_date_i++) {
                for (int rn_i=0; rn_i < one_path_randoms[fix_date_i].size(); rn_i++) {
                    for (int i=0; i < one_path_randoms[fix_date_i][rn_i].size(); i++) {
                        one_path_randoms[fix_date_i][rn_i][i] = normal_distrib(gen); 
                    }
                }
            }
            process.reset();
            OnePathProcessSimulatedData<AffineElementType> simulated_data;
            onePathSimulateProcessData(
                simulation_time_steps, legendre_absciss_indexes  
                , one_path_randoms,  process,  simulated_data
            );
            simulate_data_mc.emplace_back(simulated_data);

        }
        auto time_stop= std::chrono::high_resolution_clock::now();
        auto tot_time = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start);     
        std::cout << "\033[34m" << "Total time to simulate process: " << double(tot_time.count()) / 1e+6 
        << "sec. \033[37m" << std::endl
    ;

    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    template<class AffineElementType, class PiFuncClass, class PayoffFuncClass>
    Results compute(
        const std::vector<Time>& simulation_time_steps                                          //:I
        , const std::vector<OnePathProcessSimulatedData<AffineElementType>>& simulated_data_mc  //:I
        , const PiFuncClass& pi_func                                                            //:I
        , const PayoffFuncClass& payoff                                                         //:I
        , const AffineElementType& sigma_m                                                      //:I
        , const int num_mc_paths                                                                //:I
        , const std::vector<Time>& fixing_times = std::vector<Time>()                           //:I
    ) {
        auto time_start= std::chrono::high_resolution_clock::now();
        
        std::vector<int> fixing_times_indexes;
        for (int t_i=0; t_i<fixing_times.size(); t_i++) {
            auto fixing_time_position = std::lower_bound(
                simulation_time_steps.begin(), simulation_time_steps.end(), fixing_times[t_i]
            );        
            fixing_times_indexes.push_back(fixing_time_position - simulation_time_steps.begin());
        }
        if (fixing_times_indexes.size() != 0) std::cout << fixing_times_indexes.back() << std::endl;

        std::vector<int> legendre_absciss_indexes;
        Time t0 = simulation_time_steps.front();
        Time T = simulation_time_steps.back();
        for (int i=0; i < getLegendre().size(); i++) {
            auto simul_nearest_point = std::lower_bound(
                simulation_time_steps.begin(), simulation_time_steps.end()
                , (T-t0) / 2 * (getLegendre()[i].abscissa + 1) + t0
            );
            legendre_absciss_indexes.push_back(
                simul_nearest_point - simulation_time_steps.begin()
            );
        }
        
        double res(0.);
        double sqd_mean(0.);
        double crude_mc(0.), sqd_crude(0.);
        for (int mc_i=0; mc_i<num_mc_paths; mc_i++) {
            double val = onePathcomputePriceFromSimulatedData(
                simulation_time_steps, legendre_absciss_indexes
                , pi_func
                , simulated_data_mc[mc_i]
                , fixing_times_indexes
            );
            res += val;
            sqd_mean += val * val;
            double crude_val = payoff(
                simulated_data_mc[mc_i].assets.back()
                , simulated_data_mc[mc_i]
                , fixing_times_indexes
            ); 
            crude_mc += crude_val;
            sqd_crude += crude_val * crude_val;
        }
        crude_mc /= num_mc_paths; 
        res /= (num_mc_paths * 2);
        sqd_mean /= (num_mc_paths * 4);
        sqd_crude /= num_mc_paths;
        std::cout << crude_mc << ", Payoff Mean" << std::endl;
        std::cout << res << ", Quadrature Integral "  << std::endl;
        std::cout << std::sqrt(sqd_mean - res * res) / std::sqrt(num_mc_paths) << ", Quadr STD: " << std::endl;
        std::cout << std::sqrt(sqd_crude - crude_mc * crude_mc) / std::sqrt(num_mc_paths) << ", crude STD: " << std::endl;
        auto time_stop= std::chrono::high_resolution_clock::now();
        auto tot_time = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start);     
        std::cout 
            << "\033[34m" << "Time for computation stage: " << double(tot_time.count()) / 1e+6 
            << "sec. \033[37m" << std::endl
        ;
        Results result;
        result.quadr = res;
        result.payoff = crude_mc;
        return result;
        
    }
    
private:
    LegendrePoints getLegendre;
};


