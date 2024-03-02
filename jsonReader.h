#pragma once
#include <chrono>
#include <memory>
#include <random>
#include <string>
#include <iostream>
#include <iomanip>

#include <nlohmann/json.hpp>

#include "Payoff.h"
#include "Pi.h"
#include "types.h"
#include "Process.h"
#include "QuadratureCoefficients.h"

using json = nlohmann::json;

inline void jsonSanityCheck (const json& data_in, const char* field1, const char* field2) {            
    if (!data_in[field1].contains(field2)) {
        throw std::runtime_error(
            std::string(field1) + std::string(".") 
            + std::string(field2) + std::string(" is not provided")
        );
    }
};

inline void jsonSanityCheck (const json& data_in, const char* field1) {            
    if (!data_in.contains(field1)) {
        throw std::runtime_error(
            std::string(field1) +  std::string(" is not provided")
        );
    }
};

#ifdef  PARALLEL_IMPLEMENTATION
void from_json(const json& data, idouble& v) { v = idouble(data.get<double>()); }
static void to_json(json& j, idouble v) { j = v.val; }
#endif

inline void from_json(const json& data, std::shared_ptr<Process<vtype>>& process) {
    jsonSanityCheck(data, "Horizont");        
    jsonSanityCheck(data, "rate");
    jsonSanityCheck(data, "num_mc_paths");
    jsonSanityCheck(data, "beta");
    jsonSanityCheck(data, "Params");
    jsonSanityCheck(data, "Params", "vol_vol");
    jsonSanityCheck(data, "Params", "init_asset");
    jsonSanityCheck(data, "Params", "init_vol");
    std::string process_type = data["Type"].template get<std::string>();
    std::cout << std::setw(12) << process_type << ", Process" << "\n";

    if (process_type == "Heston") {
        jsonSanityCheck(data, "Params", "rho");
        jsonSanityCheck(data, "Params", "theta");
        jsonSanityCheck(data, "Params", "kappa");
        jsonSanityCheck(data, "rate");

        vtype rate = data["rate"].template get<vtype>();
        vtype rho = data["Params"]["rho"].template get<vtype>();
        vtype theta = data["Params"]["theta"].template get<vtype>();
        vtype kappa = data["Params"]["kappa"].template get<vtype>();
        vtype vol_vol = data["Params"]["vol_vol"].template get<vtype>();
        process = std::make_shared<Heston>(rate, rho, theta, kappa, vol_vol);
        vtype init_asset = data["Params"]["init_asset"].template get<vtype>();
        vtype init_vol = data["Params"]["init_vol"].template get<std::vector<vtype>>()[0];

        process->init(0., init_asset, init_vol);
        process->reset();
    }
    else if (process_type == "SabrScalar") {
        vtype vol_vol = data["Params"]["vol_vol"].template get<vtype>();
        process = std::make_shared<SabrScalar>(vol_vol);
        vtype init_asset = data["Params"]["init_asset"].template get<vtype>();
        vtype init_vol = data["Params"]["init_vol"].template get<std::vector<vtype>>()[0];
        process->init(0., init_asset, init_vol);
        process->reset();
    }   else throw std::runtime_error("Process specifier is not supported");

}

inline void from_json(const json& data, std::shared_ptr<Process<Eigen::vVectorXd>>& process) {
    jsonSanityCheck(data, "Horizont");
    jsonSanityCheck(data, "rate");
    jsonSanityCheck(data, "num_mc_paths");
    jsonSanityCheck(data, "Params");
    jsonSanityCheck(data, "beta");

    std::string process_type = data["Type"].template get<std::string>();
    std::cout << std::setw(12) << process_type<< ", Process" << "\n";
    if (process_type == "SABR") {
        std::cout << "Process: Sabr " << std::endl;

        int dim = data["dim"].template get<int>();
        
        vtype vol_vol_scal = data["Params"]["vol_vol"].template get<vtype>();
        Eigen::vVectorXd vol_vol(Eigen::vVectorXd::Ones(dim) * vol_vol_scal);

        Eigen::MatrixXd corr_mat(Eigen::MatrixXd::Ones(dim, dim));
        corr_mat = 0.4 * corr_mat + 0.6 * Eigen::MatrixXd::Identity(dim, dim);
        std::cout << "The structure of the correlation matrix is hardwired as \n 0.6 Id + 0.4 1^T 1\n";
        process = std::make_shared<SABR>(vol_vol, corr_mat);

        vtype init_asset_scal = data["Params"]["init_asset"].template get<vtype>();
        std::vector<double> init_vol_ = data["Params"]["init_vol"].get<std::vector<double>>();

        Eigen::vVectorXd init_asset(Eigen::vVectorXd::Ones(dim) * init_asset_scal);
        Eigen::vVectorXd init_vol(Eigen::vVectorXd::Ones(dim) );
        
        std::cout << " Init.vols: ";
        for (int i = 0; i < dim; i++) {
            init_vol[i] = init_vol_[i];
            std::cout << init_vol[i] << " ";
        }
        std::cout << std::endl;
        process->init(0., init_asset, init_vol);
        process->reset();
    }
    else 
        if (process_type == "HestonMultDim") {
            std::cout << "Process: Heston Multidimensional " << std::endl;
            jsonSanityCheck(data, "Params", "theta");
            jsonSanityCheck(data, "Params", "rho");
            jsonSanityCheck(data, "Params", "kappa");
            jsonSanityCheck(data, "Params", "vol_vol");
            jsonSanityCheck(data, "Params", "init_asset");
            jsonSanityCheck(data, "Params", "init_vol");

            int dim = data["dim"].template get<int>();
            vtype vol_vol_scal = data["Params"]["vol_vol"].template get<vtype>();
            vtype rate_scal = data["rate"].template get<vtype>();
            vtype theta_scal = data["Params"]["theta"].template get<vtype>();
            vtype rho_scal = data["Params"]["rho"].template get<vtype>();
            vtype kappa_scal = data["Params"]["kappa"].template get<vtype>();
            vtype init_asset_scal = data["Params"]["init_asset"].template get<vtype>();
            std::vector<double> init_vol_ = data["Params"]["init_vol"].get<std::vector<double>>();

            std::cout << "Process: Heston Multidimensional " << std::endl;

            Eigen::vVectorXd vol_vol(Eigen::vVectorXd::Ones(dim) * vol_vol_scal);
            Eigen::vVectorXd rate(Eigen::vVectorXd::Ones(dim) * rate_scal);
            Eigen::vVectorXd theta(Eigen::vVectorXd::Ones(dim) * theta_scal);
            Eigen::vVectorXd rho(Eigen::vVectorXd::Ones(dim) * rho_scal);
            Eigen::vVectorXd kappa(Eigen::vVectorXd::Ones(dim) * kappa_scal);
            Eigen::vVectorXd init_asset(Eigen::vVectorXd::Ones(dim) * init_asset_scal);
            Eigen::vVectorXd init_vol(Eigen::vVectorXd::Ones(dim));

            std::cout << " Init.vols: ";
            for (int i = 0; i < dim; i++) {
                init_vol[i] = init_vol_[i];
                std::cout << init_vol[i] << " ";
            }


            Eigen::MatrixXd corr_mat(Eigen::MatrixXd::Ones(dim, dim));
            corr_mat = 0.4 * corr_mat + 0.6 * Eigen::MatrixXd::Identity(dim, dim);
            std::cout << "The structure of the correlation matrix is hardwired as \n 0.6 Id + 0.4 1^T 1\n";
            process = std::make_shared<HestonMultDim>(vol_vol, rate, kappa, theta, rho, corr_mat);
            std::cout << std::endl;
            process->init(0., init_asset, init_vol);
            process->reset();
        }   else
                throw std::runtime_error("Process specifier is not supported");
}


inline void from_json(const json& data, std::shared_ptr<Payoff<vtype>>& payoff) {
    jsonSanityCheck(data, "strike");
    jsonSanityCheck(data, "maturity");        
    jsonSanityCheck(data, "Payoff");        

    vtype strike = data["strike"].template get<vtype>();
    double T = data["maturity"].template get<double>();
    vtype rate = data["rate"].template get<vtype>();
    std::string payoff_type = data["Payoff"].template get<std::string>();
    std::cout << std::setw(12) << payoff_type << ", Payoff" << "\n";
    if (payoff_type == "EuropeanCall") {       
        payoff = std::make_shared<EuropeanCall>(strike, rate, T);
    } else 
    if (payoff_type == "EuropeanCallAsian") {       
        payoff = std::make_shared<EuropeanCallAsian>(strike, rate, T);
    } else
    if (payoff_type == "LookbackCall") {
        payoff = std::make_shared<LookbackCall>();
    } else
    if (payoff_type == "DownAndOutEuropCallPayoff") {    
        vtype barrier = data["barrier"].template get<vtype>();   
        payoff = std::make_shared<DownAndOutEuropCallPayoff>(strike, barrier, T);
    } else 
        throw std::runtime_error("Payoff specifier is not supported");
}

inline void from_json(const json& data, std::shared_ptr<Pi<vtype>>& pi_func) {

    jsonSanityCheck(data, "maturity");
    jsonSanityCheck(data, "Pi");
    vtype blackscholes_beta = 1.;
    vtype bachelier_beta = 0.;
    vtype init_asset = data["init_asset"].template get<vtype>();
    vtype sigma_m = data["declared_vol"].template get<vtype>();
    std::cout << "Declared vol: " << sigma_m << std::endl;
    vtype process_beta = data["process_beta"].template get<vtype>();

    double T = data["maturity"].template get<double>();
    vtype rate = data["rate"].template get<vtype>();
    std::string pi_type = data["Pi"].template get<std::string>();
    std::cout << std::setw(12)  << pi_type << ", Pi " << std::endl;
    if (pi_type == "BlackScholes") {       
        vtype beta = 1;
        sigma_m *= pow(init_asset, process_beta) / pow(init_asset, beta);
        jsonSanityCheck(data, "strike");
        vtype strike = data["strike"].template get<vtype>();
        std::cout << std::setw(12) << strike << ", strike" << std::endl;
        pi_func = std::make_shared<BlackScholes>(strike, rate, sigma_m, T);        
    } else
    if (pi_type == "BachelierAsian") {       
        vtype beta = 0;
        sigma_m *= pow(init_asset, process_beta) / pow(init_asset, beta);
        jsonSanityCheck(data, "strike");
        vtype strike = data["strike"].template get<vtype>();
        std::cout << std::setw(12) << strike << ", strike" << std::endl;
        std::vector<double> fixing_times = data["fixing_times"].get<std::vector<double>>(); 
        for (auto item : fixing_times) std::cout << item << " ";
        std::cout << ": AveragingTimes" << std::endl;
        pi_func = std::make_shared<BachelierAsian>(strike, rate, sigma_m, T, fixing_times);        
    } else 
    if (pi_type == "BlackLookbackCall") {
        vtype beta = 1;
        sigma_m *=  pow(init_asset, process_beta) / pow(init_asset, beta);
        std::cout << "sss" << sigma_m << std::endl;
        vtype init_asset = data["init_asset"].template get<vtype>();
        pi_func = std::make_shared<BlackLookbackCall>(rate, sigma_m, T, init_asset);
    } else
    if (pi_type == "DownAndOutEuropCallPrice") {       
        vtype beta = 1;
        sigma_m *= pow(init_asset, process_beta) / pow(init_asset, beta);
        jsonSanityCheck(data, "strike");
        vtype strike = data["strike"].template get<vtype>();
        std::vector<double> fixing_times = data["fixing_times"].get<std::vector<double>>(); 
        vtype barrier = data["barrier"].template get<vtype>();
        std::cout << std::setw(12) << barrier << ", barrier" << std::endl;
        std::vector<double> st= data["simulation_times"].template get<std::vector<double>>();
        pi_func = std::make_shared<DownAndOutEuropCallPrice>(strike, rate, barrier, sigma_m, T, st);        
    } else {
        throw std::runtime_error("Pi function specifier is not supported");
    }
}

inline void from_json(const json& data, std::shared_ptr<Pi<Eigen::vVectorXd>>& pi_func) {
    vtype strike = data["strike"].template get<vtype>();
    std::vector<double> sigma_m_scal = data["init_vol"].get<std::vector<double>>();
    vtype init_asset = data["init_asset"].template get<vtype>();
    vtype process_beta = data["process_beta"].template get<vtype>();
    double T = data["maturity"].template get<double>();
    vtype rate = data["rate"].template get<vtype>();
    int dim = data["dim"].template get<int>();
    if (dim != sigma_m_scal.size()) throw std::runtime_error("dim is not equal size of sigma_m");
    Eigen::MatrixXd corr_mat(Eigen::MatrixXd::Ones(dim, dim));
    corr_mat = 0.4 * corr_mat + 0.6 * Eigen::MatrixXd::Identity(dim, dim);

    Eigen::vVectorXd sigma_m_vec(dim);
    for (int i = 0; i < dim; i++) {
        sigma_m_vec[i] = sigma_m_scal[i];
    }
    std::string process_type = data["Type"].template get<std::string>();
    if (process_type == "HestonMultDim") {
        sigma_m_vec = sigma_m_vec.cwiseSqrt();
    }


    std::string pi_type = data["Pi"].template get<std::string>();
    std::cout << "Pi: " << pi_type << std::endl;
    if (pi_type == "BasketBachelier") {          
        vtype beta = 0;
        
        sigma_m_vec = sigma_m_vec * pow(init_asset, process_beta) / pow(init_asset, beta);
        //Eigen::vVectorXd sigma_m_vec(Eigen::vVectorXd::Ones(dim) * sigma_m_scal);
        std::cout << "sigma_m_vec:" << sigma_m_vec << std::endl;
        pi_func = std::make_shared<BasketBachelier>(strike, rate, T, sigma_m_vec, corr_mat);        
    } else 
    if (pi_type == "RainbowCallMaxBlack") {
        vtype beta = 1;
        std::cout << "CAUTION: ONLY DIM=3 is implemented;" << std::endl;
        sigma_m_vec = sigma_m_vec * pow(init_asset, process_beta) / pow(init_asset, beta);
        pi_func = std::make_shared<RainbowCallMaxBlack>(strike, rate, T, sigma_m_vec, corr_mat);
    }
    else {
        throw std::runtime_error("Pi function specifier is not supported");
    }
}


inline void from_json(const json& data, std::shared_ptr<Payoff<Eigen::vVectorXd>>& payoff) {
    vtype strike = data["strike"].template get<vtype>();
    double T = data["maturity"].template get<double>();
    std::string payoff_type = data["Payoff"].template get<std::string>();
    std::cout << std::setw(12)  << payoff_type << ", Payoff " << "\n";
    if (payoff_type == "BasketCall") {       
        payoff = std::make_shared<BasketCall>(strike, T);
    } else 
    if (payoff_type == "RainbowCallOnMax") {
        payoff = std::make_shared<RainbowCallOnMax>(strike, T);
    }  else
        throw std::runtime_error("Payoff specifier is not supported");
}


inline void from_json(const json& data, std::function<std::vector<QuadratureCoefficients>&()>& getlegendre) {
    if (!data.contains("num_Legendre_points")) {
        throw std::runtime_error("Json field num_Legendre_points is absent");           
    }
    int num_Legendre_points = data["num_Legendre_points"].template get<int>();
    std::cout << std::setw(12) << num_Legendre_points << ", NumLegendrePoints" << std::endl;
    switch (num_Legendre_points) {
        case 20:
            getlegendre = getLegendre20;
            break;    
        case 64:
            getlegendre = getLegendre64;
            break;    
        case 24:
            getlegendre = getLegendre24;
            break;
        default:
            throw std::runtime_error("20,24 and 64 only supported as a Legendre points number");
    } 
}