#pragma once 
#include <iomanip>
#ifdef PARALLEL_IMPLEMENTATION
#include <thread>
#include <typeinfo>

#include "aadc/aadc_eigen.h"

#include "types.h"
#include "Process.h"
#include "Pi.h"
#include "Payoff.h"
#include "QuadratureCoefficients.h"
#include "utils.h"


inline void markVariable(const  vtype& x, std::vector<aadc::AADCArgument>& args) {
    args.push_back(x.markAsDiff());
}

inline void markVariable(
    const Eigen::vVectorXd& x
    , std::vector<aadc::AADCArgument>& args
) {
    for (int i=0; i< x.size(); i++) {
        args.push_back(x[i].markAsDiff());
    }
}


////////////////////////////////////////////////////////////////////////////////////
//
//  Pricing Driver.  Recieve instances of Pi,Payoff and Process classes and performs MC-computations
//
//  template<class AffineElementType> void onePathPricing;
//  This method Implements one MC-path, i.e. process dynamics and computation of observables
//
//  Method  void KernelRecordAndRun; then records the AADC-kernel and implements its PARALLEL_IMPLEMENTATION computations
//  
//////////////////////////////////////////////////////////////////////////////////////

class DriverParallel {
public:
    DriverParallel(
        std::function<std::vector<QuadratureCoefficients>&()>& getLegendre_
    ) : getLegendre(getLegendre_) 
    {}


    /////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    template<class AffineElementType>
    void onePathPricing (
        const std::vector<Time>& simulation_times                                //:I
        , const std::vector<int>& legendre_absciss_indexes                       //:I
        , const std::vector<std::vector<Eigen::vVectorXd>>& random_samples       //:I
        , Process<AffineElementType>& process                                    //:I/O
        , Pi<AffineElementType>& pi_func                                   //:I
        , const Payoff<AffineElementType>& payoff                                //:I
        , const std::vector<int>& fixing_times_indexes
    ) {
        vtype one_path_impact = vtype(0.);
        vtype quadrature = vtype(0.);
        vtype maturity(simulation_times.back());
        vtype t0(simulation_times.front());

        OnePathProcessSimulatedData<AffineElementType> sim_data;
        vtype accumulated = 1.;
        pi_func.reset();

        for (int t_i = 1; t_i < simulation_times.size(); ++t_i) {
            //if (dt == 0) continue;
            #ifdef PARALLEL_IMPLEMENTATION
            if (idouble::recording) CAAD_LoopPulse(t_i);
            #endif
            process.simulateAssetOneStep(simulation_times[t_i], random_samples[t_i]);
            pi_func.moveNextTime(simulation_times[t_i]);
            Time current_t = simulation_times[t_i];
            sim_data.assets.push_back(process.getCurrentAsset());
            double dt = simulation_times[t_i] - simulation_times[t_i - 1];

            AffineElementType curr_asset_p, curr_asset_m;
            AffineElementType curr_asset_pppp, curr_asset_ppp, curr_asset_pp;

            std::vector<AffineElementType> vol_corr_vect1(
                std::move(
                    process.getAssetBM()->getVolTimesCorrRankOneRepresentation(
                        process.getVolOfAsset()
                    )
                )
            );
            std::vector<AffineElementType> vol_corr_vect2(
                std::move(
                    process.getAssetBM()->getVolTimesCorrRankOneRepresentation(
                        pi_func.getVolOfAsset(process.getCurrentAsset())
                    )
                )
            );
            vtype base;
            vtype eps=0.01, gamma_ass(0.), gamma_base(0.);
            
            if (pi_func.barrierFactor) {
                accumulated *= (*pi_func.barrierFactor)(process.getCurrentAsset());
            }

            if (pi_func.extrRealizedVal) {
                (*pi_func.extrRealizedVal)(process.getCurrentAsset());
            }


            auto legendre_position = std::find(
                legendre_absciss_indexes.begin()
                , legendre_absciss_indexes.end()
                , t_i
            );

            if (legendre_position != legendre_absciss_indexes.end())  // Comment this line to switch off computation of Hessian at each simulat. point
            {

                for (int i = 0; i < process.getAssetBM()->corrMatrixActualRank(); i++) {
                    curr_asset_p = process.getCurrentAsset() + eps * vol_corr_vect1[i];
                    curr_asset_m = process.getCurrentAsset() - eps * vol_corr_vect1[i];

                    gamma_ass +=
                        pi_func(current_t, curr_asset_p, sim_data, fixing_times_indexes)
                        + pi_func(current_t, curr_asset_m, sim_data, fixing_times_indexes)
                    ;
                    curr_asset_p = process.getCurrentAsset() + eps * vol_corr_vect2[i];
                    curr_asset_m = process.getCurrentAsset() - eps * vol_corr_vect2[i];
                    gamma_base +=
                        pi_func(current_t, curr_asset_p, sim_data, fixing_times_indexes)
                        + pi_func(current_t, curr_asset_m, sim_data, fixing_times_indexes)
                    ;

                }
            }
            vtype inc = accumulated * (gamma_ass - gamma_base) / (eps * eps); 
            one_path_impact += inc * dt;
            if (legendre_position != legendre_absciss_indexes.end()) {
                auto it = legendre_position - legendre_absciss_indexes.begin();
                quadrature += inc * getLegendre()[it].weight;
            }
        } 
        one_path_impact_arg = (0.5 * one_path_impact).markAsOutput();
        one_path_quadr_arg = (0.5 * (quadrature / 2 * (maturity - t0))).markAsOutput();
        payoff_arg = payoff(
            process.getCurrentAsset(), sim_data, fixing_times_indexes
        ).markAsOutput();
    }


    /////////////////////////////////////////////////////////////////////////////////////////////////////
    //
    //  void KernelRecordAndRun
    //  Records the AADC-kernel and implements its PARALLEL_IMPLEMENTATION computations
    //
    /////////////////////////////////////////////////////////////////////////////////////////////////////

    template<class mmType, class ProcessType, class AffineElementType, class PiFuncClass, class PayoffFuncClass>
    Results run(
        const std::vector<Time>& simulation_time_steps_
        , ProcessType& process
        , const AffineElementType& explain_affine_element_type
        , PiFuncClass& pi_func
        , const PayoffFuncClass& payoff
        , const int num_threads 
        , const int num_mc_paths 
        , const std::vector<Time>& fixing_times = std::vector<Time>()
    ){
        std::vector<Time> simulation_time_steps;
        std::merge(
            simulation_time_steps_.begin(), simulation_time_steps_.end()
            , fixing_times.begin(), fixing_times.end()
            , std::back_inserter(simulation_time_steps)
        );
        if (print_info) {
            //std::cout << "Num.threads: " << num_threads << "\n";
            std::cout << std::setw(12) << num_mc_paths << ", Num.MC-paths" << std::endl;
            //std::cout <<  "AVX: " << typeid(mmType).name() << "\n";
        }

        std::size_t num_time_steps = simulation_time_steps.size();
        std::cout << "Initial asset: " << process.getCurrentAsset() << std::endl;
        vtype base = pi_func(0, process.getCurrentAsset());

        double eps = 0.01;
        std::vector<vtype> base_ups;
        for (int i=0; i< process.getRandomVectorDim(); i++) {
            AffineElementType shocked_vec(AffineElement::getI(i, eps, process.getCurrentAsset()));
            base_ups.push_back(pi_func(0, process.getCurrentAsset() + shocked_vec));
        }

        Time t0 = simulation_time_steps.front();
        Time T = simulation_time_steps.back();

        std::vector<int> legendre_absciss_indexes;
        for (int i=0; i < getLegendre().size(); i++) {
            auto simul_nearest_point = std::lower_bound(
                simulation_time_steps.begin(), simulation_time_steps.end()
                , (T-t0) / 2 * (getLegendre()[i].abscissa + 1) + t0
            );
            legendre_absciss_indexes.push_back(simul_nearest_point - simulation_time_steps.begin());
        }

        std::vector<int> fixing_times_indexes;
        for (int t_i=0; t_i<fixing_times.size(); t_i++) {
            auto fixing_time_position = std::lower_bound(
                simulation_time_steps.begin(), simulation_time_steps.end(), fixing_times[t_i]
            );        
            fixing_times_indexes.push_back(fixing_time_position - simulation_time_steps.begin());
        }

        std::size_t AVXsize = aadc::mmSize<mmType>();
        std::size_t paths_per_thread = num_mc_paths / (AVXsize * num_threads);    
        

        //////////// Step 1 ////////////////////////////////
        // Prepare HPC-kernel
        /////////////////////////////////////////////////////
        aadc::AADCFunctions<mmType> aad_funcs({{aadc::AADC_CodeBufferSize, 1e+7}});
        std::vector<std::vector<Eigen::vVectorXd>> one_path_randoms(num_time_steps);
        for (int i=0; i < one_path_randoms.size(); i++) {
            one_path_randoms[i].resize(process.getRandomVectorsPerTimePoint()) ;
            for (int j=0; j < process.getRandomVectorsPerTimePoint(); j++) {
                one_path_randoms[i][j].resize(process.getRandomVectorDim());
                for (int k=0; k < one_path_randoms[i][j].size(); k++) one_path_randoms[i][j][k]=1;
            }
        }
        std::vector<aadc::VectorArg> random_args;
        aad_funcs.startRecording();
            asset_args.resize(0);
            markVariable(process.getInitAsset(), asset_args);
            process.reset();

            int rnd_counter = 0;
            for (int fix_date_i=0; fix_date_i<num_time_steps; fix_date_i++) {
                for (int rn_i=0; rn_i<one_path_randoms[fix_date_i].size(); rn_i++) {
                    for (int i=0; i<one_path_randoms[fix_date_i][rn_i].size(); i++) {
                        one_path_randoms[fix_date_i][rn_i][i].markAsArrayInputNoDiff(rnd_counter++);
                    }
                }
            }
            int num_onepath_randoms = rnd_counter;

            onePathPricing(
                simulation_time_steps, legendre_absciss_indexes  
                , one_path_randoms, process, pi_func, payoff, fixing_times_indexes
            );
            
        aad_funcs.stopRecording();
        if (print_info) {
            aad_funcs.printPassiveExtractLocations(std::cout, "VolRed");
            //aad_funcs.outStats(std::cout, "A");
        }

        //////////// Step 2 ////////////////////////////////
        // Prepare a lambda-function that will call the recorded kernel from each thread
        /////////////////////////////////////////////////////

        auto time_start= std::chrono::high_resolution_clock::now();
        std::vector<mmVector<mmType>> mm_quadratures(num_threads);
        std::vector<mmVector<mmType>> mm_impacts(num_threads);
        std::vector<mmVector<mmType>> mm_payoffs(num_threads);
        std::vector<std::vector<mmVector<mmType>>> mm_diff_p(num_threads);
        std::vector<std::vector<mmVector<mmType>>> mm_diff_q(num_threads);
        
        for (int th_i=0; th_i < num_threads; th_i++) {
            mm_impacts[th_i].resize(paths_per_thread);
            mm_payoffs[th_i].resize(paths_per_thread);
            mm_quadratures[th_i].resize(paths_per_thread);
            mm_diff_p[th_i].resize(asset_args.size());
            mm_diff_q[th_i].resize(asset_args.size());
            for (int dim_i=0; dim_i < asset_args.size(); dim_i++) {
                mm_diff_p[th_i][dim_i].resize(paths_per_thread);
                mm_diff_q[th_i][dim_i].resize(paths_per_thread);
            }
        }

        auto threadWorker = [&] (
            int th_i 
            , mmVector<mmType>& mm_impacts_per_thread
            , mmVector<mmType>& mm_payoffs_per_thread
            , mmVector<mmType>& mm_quadratures_per_thread
            , std::vector<mmVector<mmType>>& mm_diff_p_per_thread
            , std::vector<mmVector<mmType>>& mm_diff_q_per_thread
        ) {
            std::mt19937_64 gen(th_i+135);
            std::normal_distribution<> normal_distrib(0.0, 1.0);            
            std::shared_ptr<aadc::AADCWorkSpace<mmType> > ws(aad_funcs.createWorkSpace());
            ws->resetDiff();

            mmVector<mmType> mm_randoms(num_onepath_randoms);

            if (th_i==0) std::cout << "Paths per thread: " << paths_per_thread << std::endl;
            for (int mc_i = 0; mc_i < paths_per_thread; ++mc_i) {
                if ((mc_i % 10 == 0) && (th_i==0)) {
                    std::cout << mc_i << " " << std::flush;
                    for (int i = 0; i < std::log10(mc_i + 1) + 1; i++) std::cout << "\b" << std::flush;
                }
                for (int avx_i=0; avx_i<AVXsize; avx_i++) {
                    for (int i=0; i < num_onepath_randoms; i++) {
                        #ifndef DISABLE_RANDOM_GENERATOR
                        toDblPtr(mm_randoms[i])[avx_i] = normal_distrib(gen); 
                        #endif
                    }
                }
                ws->setArray(&mm_randoms[0]);            
                aad_funcs.forward(*ws);
                
                mm_impacts_per_thread[mc_i] = ws->val(one_path_impact_arg);
                mm_payoffs_per_thread[mc_i] = ws->val(payoff_arg); 
                mm_quadratures_per_thread[mc_i] = ws->val(one_path_quadr_arg); 
                ws->resetDiff();
                ws->setDiff(one_path_impact_arg, 0.);
                ws->setDiff(payoff_arg ,1.);
                ws->setDiff(one_path_quadr_arg ,0.);
                aad_funcs.reverse(*ws);
                for (int dim_i=0; dim_i < asset_args.size(); dim_i++) {
                    mm_diff_p_per_thread[dim_i][mc_i] = ws->diff(asset_args[dim_i]);
                }
                ws->resetDiff();
                ws->setDiff(one_path_impact_arg, 0.);
                ws->setDiff(payoff_arg ,0.);
                ws->setDiff(one_path_quadr_arg, 1.);
                aad_funcs.reverse(*ws);

                for (int dim_i=0; dim_i < asset_args.size(); dim_i++) {
                    mm_diff_q_per_thread[dim_i][mc_i] = ws->diff(asset_args[dim_i]);
                }
            }

        };

        //////////// Step 3 ////////////////////////////////
        // Run the recorded kernel in many threads
        /////////////////////////////////////////////////////

        std::vector<std::unique_ptr<std::thread>> threads;
        for(int th_i=0; th_i < num_threads; th_i++) {
            threads.push_back(
                std::unique_ptr<std::thread>(
                    new std::thread(
                        threadWorker
                        , th_i
                        , std::ref(mm_impacts[th_i])
                        , std::ref(mm_payoffs[th_i])
                        , std::ref(mm_quadratures[th_i])
                        , std::ref(mm_diff_p[th_i])
                        , std::ref(mm_diff_q[th_i])
                    )
                )
            );
        }
        for(auto&& t: threads) t->join();
        
        double mean_integral(0.), mean_payoff(0.), mean_quadratures(0.);
        std::vector<double> mean_diff_p(asset_args.size(), 0.), mean_diff_q(asset_args.size(), 0.);
        double mean_sqd_integral(0.), mean_sqd_payoff(0.),  mean_sqd_quadratures(0.);
        std::vector<double> mean_sqd_p(asset_args.size(), 0.), mean_sqd_q(asset_args.size(), 0.);
        for (int th_i=0; th_i < num_threads; th_i++) {
            for (int mc_i=0; mc_i < paths_per_thread; mc_i++) {
                for (int avx_i=0; avx_i<AVXsize; avx_i++) { 
                    double v_i = toDblPtr(mm_impacts[th_i][mc_i])[avx_i];
                    double v_p = toDblPtr(mm_payoffs[th_i][mc_i])[avx_i];
                    double v_q = toDblPtr(mm_quadratures[th_i][mc_i])[avx_i];
                    for (int dim_i=0; dim_i<asset_args.size(); dim_i++){
                        double diff_p = toDblPtr(mm_diff_p[th_i][dim_i][mc_i])[avx_i];
                        double diff_q = toDblPtr(mm_diff_q[th_i][dim_i][mc_i])[avx_i];
                        mean_diff_p[dim_i] += diff_p;
                        mean_diff_q[dim_i] += diff_q;
                        mean_sqd_p[dim_i] += diff_p * diff_p;
                        mean_sqd_q[dim_i] += diff_q * diff_q;
                    }
                    mean_integral += v_i;
                    mean_sqd_integral += v_i * v_i;
                    mean_payoff += v_p;
                    mean_sqd_payoff += v_p * v_p;
                    mean_quadratures += v_q;
                    mean_sqd_quadratures += v_q * v_q;
        
                }
            }
        }
        mean_integral /= num_mc_paths;
        mean_sqd_integral /= num_mc_paths;
        mean_payoff /= num_mc_paths;
        mean_sqd_payoff /= num_mc_paths;
        mean_sqd_quadratures /= num_mc_paths;
        mean_quadratures /= num_mc_paths;
        for (int dim_i=0; dim_i<asset_args.size(); dim_i++){
            mean_diff_p[dim_i] /= num_mc_paths;
            mean_diff_q[dim_i] /= num_mc_paths;
            mean_sqd_p[dim_i] /= num_mc_paths;
            mean_sqd_q[dim_i] /= num_mc_paths;
        }
        auto time_stop= std::chrono::high_resolution_clock::now();
        auto tot_time = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start);

        if (print_info) {
            std::cout << std::setprecision(7);
            std::cout 
                << std::setw(12) << mean_integral + base  << ", Denoising" << std::endl
                << std::setw(12) << mean_integral  << ", Integral:" << std::endl
                << std::setw(12) << base  << ", Base:" << std::endl
            ;
            std::cout
                << std::setw(12)
                << std::sqrt(mean_sqd_integral - mean_integral * mean_integral) / std::sqrt(num_mc_paths)
                << ", std(Integral)" << std::endl
            ;
            std::cout << std::setw(12) << mean_payoff << ", Mean(payoff): " << std::endl;
            std::cout  
                << std::setw(12) << std::sqrt(mean_sqd_payoff - mean_payoff * mean_payoff) / std::sqrt(num_mc_paths)
                << ", stdev (payoff): \n"
            ;
            std::cout
                << std::setw(12) << mean_quadratures + base << ", Mean(Quadrature): " << std::endl
                << std::setw(12) << mean_quadratures << ", IntegralQuadr " << std::endl
                << std::setw(12) << base << ", base" << std::endl;
            ;
            std::cout
                << std::setw(12)
                << std::sqrt(mean_sqd_quadratures - mean_quadratures * mean_quadratures) / std::sqrt(num_mc_paths)
                << ", stdev (Quadrature): " << std::endl
            ;

            //std::cout << "\033[34m" << "\nAnalytic Deltas: \033[37m" << std::endl;
        
            for (int dim_i=0; dim_i<asset_args.size(); dim_i++){
                std::cout
                    << std::setw(12) << mean_diff_p[dim_i] << ", Analytic Payoff Delta:" << std::endl
                    << std::setw(12) << mean_diff_q[dim_i] + (base_ups[dim_i] - base) / eps << ", Analytic Quadr Delta" << std::endl
                    << std::setw(12) << (base_ups[dim_i] - base) / eps << ", DerivativeOfBase(analytic)" << std::endl
                    << std::setw(12) << mean_diff_q[dim_i] << ", QuadrDerivative(analytic)" << std::endl
                ;
                std::cout 
                    << std::setw(12) << std::sqrt(mean_sqd_q[dim_i] - mean_diff_q[dim_i] * mean_diff_q[dim_i])
                    / std::sqrt(num_mc_paths) << ", StdQuadrDelta:" << std::endl
                ;
                std::cout 
                    << std::setw(12)
                    << std::sqrt(mean_sqd_p[dim_i] - mean_diff_p[dim_i] * mean_diff_p[dim_i]) 
                    / std::sqrt(num_mc_paths) << ", stdPayoffDelta:" << std::endl
                ;
            } 
            std::cout << "\033[34m" << std::setw(12)  << double(tot_time.count()) / 1e+6
                << ", Total time: "  << "sec. \033[37m \n\n" << std::endl
            ;
        }

        Results result;
        result.fancy = mean_integral + AAD_PASSIVE(base);
        result.quadr = mean_quadratures + AAD_PASSIVE(base);
        result.payoff = mean_payoff;
        return result;
   }

public:
    bool print_info = true;
private:
    aadc::AADCResult one_path_impact_arg, one_path_quadr_arg,  payoff_arg;
    std::vector<aadc::AADCArgument> asset_args;
    LegendrePoints getLegendre;
};

#endif