#define PARALLEL_IMPLEMENTATION
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <thread>
#include <string>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#ifdef PARALLEL_IMPLEMENTATION 
#define AADC_ALLOW_TO_PASSIVE_BOOL
#include <aadc/aadc.h>
#include <aadc/aadc_matrix.h>
#include "aadc/aadc_eigen.h"
using namespace aadc;
#endif

#include "Results.h"
#include "types.h"
#include "QuadratureCoefficients.h"
#include "Process.h"
#include "Pi.h"
#include "Payoff.h"
#include "DriverParallel.h"
#include "DriverPlain.h"
#include "jsonReader.h"
#include "MultivariateCDF.h"


template<class mmType, class AffineElementType>
double runOneCase(json& data_in) {
    std::shared_ptr<Process<AffineElementType>> process = 
        data_in["Process"].get<std::shared_ptr<Process<AffineElementType>>>()
    ;
    std::cout << "Process Init asset(s): " << process->getCurrentAsset() << std::endl;

    // Create list of simulation time points
    Time Horizont = data_in["Process"]["Horizont"].get<double>();
    int num_time_steps = data_in["Process"]["num_time_steps"].get<int>();
    std::vector<Time> simulation_time_steps_init(num_time_steps + 1 , 0.); 
    for (int i=0; i < simulation_time_steps_init.size(); i++) {
        simulation_time_steps_init[i] = i * (Horizont / num_time_steps);
    }

    // Using simulated process data we will be able to price various models option prices. But its rate should be 
    // inherited from the process
    vtype rate = data_in["Process"]["rate"].get<vtype>();
    vtype process_beta = data_in["Process"]["beta"].get<double>();

    int dim = 1;
    if (data_in["Process"].contains("dim")) dim = data_in["Process"]["dim"].get<int>();
    int num_mc_paths = data_in["Process"]["num_mc_paths"].get<int>(); // should be a multiple of num_threads * 4(8)

    auto getLeg = data_in.get<std::function<std::vector<QuadratureCoefficients>&()>>();
    #ifdef PARALLEL_IMPLEMENTATION
    DriverParallel driver(getLeg);
    #else
    DriverPlain driver(getLeg);;
    #endif

    #ifndef PARALLEL_IMPLEMENTATION
    // In parallel implementation we will resimulate process data for each option price. It is not related with 
    // the parallelelazability/ not parallelazability. It is just another use case, where one can add LEgendre points 
    // and fixing times into the list of simulation points, and not just used the closest ones.
    // The following  code (runned in the non-parallel implementation only) simulate process data and will reuse it 
    // for each option price 
    std::vector<OnePathProcessSimulatedData<AffineElementType>> simulated_data_mc;
    std::vector<int> legendre_absciss_indexes;
    driver.simulateProcess(
        simulation_time_steps_init, legendre_absciss_indexes
        , simulated_data_mc
        , *process, AffineElementType(), num_mc_paths
    );
    process->reset();
    #endif

    // Loop other various option settings
    for (json payoff_pi_data :  data_in["Cases"]) {
        std::cout << "\n\n/////////////////////////////////////////////////// \n";
        if (payoff_pi_data["maturity"].get<double>() >  Horizont) {
            std::cout << "Maturity is larget than proces horizont" << std::endl;
            continue;
        }
        auto simulation_time_steps_ = simulation_time_steps_init;
        simulation_time_steps_.resize(
            payoff_pi_data["maturity"].get<double>() 
            / Horizont * simulation_time_steps_init.size()
        );
        payoff_pi_data["simulation_times"] = simulation_time_steps_;
        payoff_pi_data["rate"] = rate; 
        payoff_pi_data["dim"] = dim;
        payoff_pi_data["Type"] = data_in["Process"]["Type"];
        payoff_pi_data["init_asset"] = data_in["Process"]["Params"]["init_asset"].get<vtype>();
        auto payoff = payoff_pi_data.get<std::shared_ptr<Payoff<AffineElementType>>>();


        vtype declared_vol;
        if (data_in["Process"]["Type"] == "Heston") {
            declared_vol = std::sqrt(data_in["Process"]["Params"]["theta"].get<vtype>());
        }
        if (data_in["Process"]["Type"] == "SabrScalar") {
            declared_vol = data_in["Process"]["Params"]["init_vol"].get<std::vector<vtype>>()[0];
        }
        payoff_pi_data["declared_vol"] = declared_vol;
        payoff_pi_data["init_vol"] =data_in["Process"]["Params"]["init_vol"].get<std::vector<double>>();
        payoff_pi_data["process_beta"] = process_beta;
        int num_threads = 8;
        std::vector<double> fixing_times = 
            payoff_pi_data["fixing_times"].get<std::vector<double>>()
        ; 
        auto pi_func = payoff_pi_data.get<std::shared_ptr<Pi<AffineElementType>>>();
        #ifdef PARALLEL_IMPLEMENTATION
        driver.print_info = true;
        Results res_base = driver.run<mmType>(
            simulation_time_steps_, *process, AffineElementType()
            , *pi_func, *payoff, num_threads, num_mc_paths, fixing_times
        );
        process->reset();
        driver.print_info = false;

        //std::cout << "Bump&Revalue stage: " << std::endl;        
        /*
        double eps = 0.1;
        process->bump(eps);
        Results res_up = driver.run<mmType>(
            simulation_time_steps_, *process, AffineElementType()
            , *pi_func, *payoff, num_threads, num_mc_paths, fixing_times
        );
        process->reset();

        process->bump(-2 * eps);
        Results res_down = driver.run<mmType>(
            simulation_time_steps_, *process, AffineElementType()
            , *pi_func, *payoff, num_threads, num_mc_paths, fixing_times
        );
        process->reset();
        process->bump(eps);

        //std::cout << "\033[34m Deltas: B&R (centrel 2d-order scheme with eps=): \033[37m" << eps << std::endl;
        std::cout 
            << std::setw(12) << (res_up.payoff - res_down.payoff) / eps / 2 << ", PayoffBumpDelta" << std::endl
            << std::setw(12) << (res_up.fancy - res_down.fancy) / eps / 2 << ", IntegralBumpDelta" << std::endl 
            << std::setw(12) << (res_up.quadr - res_down.quadr) / eps / 2 << ", QuadrBumpDelta" << std::endl
        ;
        //end
        */
        #else
        std::vector<Time> gen_simulation_times = simulation_time_steps_;
        double base = (*pi_func)(0, process->getCurrentAsset());

        auto simulated_data_mc_copy = simulated_data_mc;
        for (int i=0; i<simulated_data_mc_copy.size(); i++) {
            simulated_data_mc_copy[i].assets.resize(gen_simulation_times.size());
            simulated_data_mc_copy[i].vols.resize(gen_simulation_times.size());
        }
        Results res = driver.compute(
            gen_simulation_times, simulated_data_mc_copy
            , *pi_func, *payoff, AffineElementType(), num_mc_paths, fixing_times
        );
        std::cout << base << ", base " << std::endl;
        std::cout << base + res.quadr << ", Quadrature Mean" << std::endl;
        std::cout << "\n-------------------------------------" << std:: endl;
        #endif
    }
    return 0;
}


int main(int argc, char** argv) {
    std::cout << "!! " << tvnls(0.2, 0, 2, 0.3, 0.1, 0.7) << std::endl;
    typedef __m256d mmType;  // __mm256d and __mm512d are supported
    std::string file_path = "../../../../Generic/Examples/HestonMultiDim.json";
    //std::string file_path = "../../../../Generic/Examples/SabrScalar.json";
    //std::string file_path = "../../../../Generic/Examples/Heston.json";
    //std::string file_path = "../../../../Generic/Examples/SABR.json";
    if (argc > 1) file_path = argv[1];

    std::ifstream file_stream(file_path);
    if (file_stream.fail()) { 
        std::cout << "Failed to open " << file_path << "\n";
        return 1;
    }
    json data_in;
    file_stream >> data_in;

    try {
        std::string proc_type = data_in["AffineElementType"].get<std::string>();
        if (proc_type == "double") {
                runOneCase<mmType, vtype>(data_in);
        } else if (proc_type == "Eigen::VectorXd") {
                runOneCase<mmType, Eigen::vVectorXd>(data_in);
        } else {
            throw std::runtime_error(
                std::string("Process specifier ") + proc_type + std::string(" is not supported")
            );
        }
    }
    catch(std::exception const& e) {
        std::cout << "Exception: " << e.what() << "\n";
    }
    // Price the first case    

    return 0;
}

