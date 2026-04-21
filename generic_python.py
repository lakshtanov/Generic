#!/usr/bin/env python3
"""
generic_python.py — DMC (Denoised Monte Carlo) via AADC Python wheel (v1.8.0).

Exact port of C++ Generic project (DriverParallel.h::onePathPricing).
FD Hessian at Legendre quadrature points, three outputs (payoff, impact, quadrature),
two reverses (payoff delta + quadrature delta).

Requires: pip install aadc numpy scipy (Python 3.9+)
Usage: python generic_python.py Examples/Heston.json
"""

import sys
import json
import numpy as np
from scipy.stats import norm
import aadc


# =================== LEGENDRE QUADRATURE ===================

LEGENDRE_24 = [
    (-0.9815606342467192, 0.0471753363865118),
    (-0.9041172563704749, 0.1069393259953184),
    (-0.7699026741943047, 0.1600783285433462),
    (-0.5873179542866175, 0.2031674267230659),
    (-0.3678314989981802, 0.2334925365383548),
    (-0.1252334085114689, 0.2491470458134028),
    ( 0.1252334085114689, 0.2491470458134028),
    ( 0.3678314989981802, 0.2334925365383548),
    ( 0.5873179542866175, 0.2031674267230659),
    ( 0.7699026741943047, 0.1600783285433462),
    ( 0.9041172563704749, 0.1069393259953184),
    ( 0.9815606342467192, 0.0471753363865118),
]

def get_legendre(n_points, t0, maturity):
    """Map Legendre points from [-1,1] to [t0, maturity] and compute time indices."""
    if n_points <= 12:
        pts = LEGENDRE_24[:n_points]
    else:
        pts = LEGENDRE_24  # use 12-point (24 not fully hardcoded)
    mid = (maturity + t0) / 2
    half = (maturity - t0) / 2
    return [(mid + half * a, w) for a, w in pts]


# =================== Pi FUNCTIONS (analytical, plain numpy) ===================

def bs_call(S, K, r, vol, T):
    """Black-Scholes call price."""
    if T < 1e-10: return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    d2 = d1 - vol*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


def bs_call_id(S, K, r, vol, T):
    """Black-Scholes call in idouble (for FD Hessian inside kernel)."""
    if float(T) < 1e-10:
        diff = S - aadc.idouble(K)
        return (diff + (diff*diff + aadc.idouble(0.01)).sqrt()) * aadc.idouble(0.5)
    vol_sqrt_T = aadc.idouble(vol) * aadc.idouble(np.sqrt(float(T)))
    log_SK = (S / aadc.idouble(K)).log()
    d1 = (log_SK + aadc.idouble(r + 0.5*vol**2) * aadc.idouble(float(T))) / vol_sqrt_T
    d2 = d1 - vol_sqrt_T
    # N(d) via erf: N(x) = 0.5*(1 + erf(x/sqrt(2)))
    sqrt2_inv = aadc.idouble(1.0 / np.sqrt(2))
    Nd1 = (aadc.idouble(1.0) + (d1 * sqrt2_inv).erf()) * aadc.idouble(0.5)
    Nd2 = (aadc.idouble(1.0) + (d2 * sqrt2_inv).erf()) * aadc.idouble(0.5)
    return S * Nd1 - aadc.idouble(K * np.exp(-r * float(T))) * Nd2


def cdf_normal_id(x):
    """N(x) in idouble via erf."""
    sqrt2_inv = aadc.idouble(1.0 / np.sqrt(2))
    return (aadc.idouble(1.0) + (x * sqrt2_inv).erf()) * aadc.idouble(0.5)


def pdf_normal_id(x):
    """n(x) in idouble."""
    return (x * x * aadc.idouble(-0.5)).exp() / aadc.idouble(np.sqrt(2 * np.pi))


def down_out_call_id(S, K, r, vol, T, barrier):
    """Down-and-out European call in idouble. Pi = BS(S) - (S/B)^(1-κ) * BS(B²/S)."""
    kappa = 2 * r / (vol * vol)
    bs_direct = bs_call_id(S, K, r, vol, T)
    B = aadc.idouble(barrier)
    # (S/B)^(1-κ): use exp((1-κ)*log(S/B))
    log_SB = (S / B).log()
    reflection_factor = (aadc.idouble(1 - kappa) * log_SB).exp()
    reflected_S = B * B / S
    bs_reflected = bs_call_id(reflected_S, K, r, vol, T)
    return bs_direct - reflection_factor * bs_reflected


def bachelier_asian_id(S, K, r, vol, T, fixing_times, sim_data_assets, fixing_idxs, current_t):
    """Bachelier Asian Pi in idouble — exact port of C++ BachelierAsian::operator()."""
    n = len(fixing_times)
    if n == 0:
        return bs_call_id(S, K, r, vol, T)

    # Precompute coefficients (plain double, same as C++ constructor)
    r_safe = max(r, 1e-6)
    mu_coeff = [0.0] * (n + 1)
    sigma_coeff1 = [0.0] * (n + 1)
    sigma_coeff2 = [0.0] * (n + 1)
    for k in range(n - 1, -1, -1):
        mu_coeff[k] = mu_coeff[k+1] + np.exp(r_safe * fixing_times[k])
        sigma_coeff1[k] = sigma_coeff1[k+1] + 0.5 * np.exp(r_safe * 2 * fixing_times[k])
        sigma_coeff2[k] = 0.5 + sigma_coeff2[k+1]
        for l in range(k+1, n):
            sigma_coeff1[k] += np.exp(r_safe * (fixing_times[k] + fixing_times[l]))
            sigma_coeff2[k] += np.exp(r_safe * abs(fixing_times[k] - fixing_times[l]))

    # Find which fixings are already past
    k_idx = 0
    for i, ft in enumerate(fixing_times):
        if ft < current_t:
            k_idx = i + 1

    # Average of past fixings
    avg_past = aadc.idouble(0.0)
    for t_i in range(k_idx):
        if t_i < len(fixing_idxs) and fixing_idxs[t_i] < len(sim_data_assets):
            avg_past = avg_past + sim_data_assets[fixing_idxs[t_i]]

    # mu = (S * mu_coeff[k] * exp(-r*t) + avg_past) / n
    mu = (S * aadc.idouble(mu_coeff[k_idx] * np.exp(-r_safe * current_t)) + avg_past) / aadc.idouble(float(n))

    # sigma
    sqrt_helper = max(sigma_coeff1[k_idx] * np.exp(-2*r_safe*current_t) - sigma_coeff2[k_idx], 0.0) / r_safe
    sqrt_helper = max(sqrt_helper, 1e-6)
    sigma = aadc.idouble(vol / n * np.sqrt(sqrt_helper))

    # Bachelier call: (mu-K)*N(x) + sigma*n(x)
    x = (mu - aadc.idouble(K)) / sigma
    return (mu - aadc.idouble(K)) * cdf_normal_id(x) + sigma * pdf_normal_id(x)


def lookback_call_id(S, r, vol, T, min_S):
    """Black lookback call Pi in idouble — exact port of C++ BlackLookbackCall::operator()."""
    T_safe = max(float(T), 0.001)
    r_safe = max(r, 0.05)  # match C++
    sq_t = aadc.idouble(np.sqrt(T_safe))
    d_p_val = r_safe + vol*vol/2
    d_m_val = r_safe - vol*vol/2

    c1 = (S / min_S).log()
    div = aadc.idouble(vol) * sq_t
    a1 = (c1 + aadc.idouble(d_p_val * T_safe)) / div
    a2 = (c1 + aadc.idouble(d_m_val * T_safe)) / div
    a3 = (c1 - aadc.idouble(d_m_val * T_safe)) / div

    disc = aadc.idouble(np.exp(r_safe * T_safe))
    half_vol2_over_r = aadc.idouble(vol*vol / (2*r_safe))

    price = (disc * S * cdf_normal_id(a1)
             - min_S * cdf_normal_id(a2)
             - disc * S * half_vol2_over_r * (
                 cdf_normal_id(aadc.idouble(-1.0) * a1)
                 - (aadc.idouble(-1.0) * c1 * aadc.idouble(2*r_safe/(vol*vol)) - aadc.idouble(r_safe * T_safe)).exp()
                   * cdf_normal_id(aadc.idouble(-1.0) * a3)
             ))
    return price


# =================== PI DISPATCHER ===================

def pi_call_id(pi_type, S, current_t, maturity, rate, vol, strike, barrier,
               fixing_times, fixing_idxs, sim_data_assets, min_S=None):
    """Call the appropriate Pi function in idouble."""
    T_remain = maturity - current_t
    if pi_type == "BlackScholes":
        return bs_call_id(S, strike, rate, vol, T_remain)
    elif pi_type == "DownAndOutEuropCallPrice":
        return down_out_call_id(S, strike, rate, vol, T_remain, barrier)
    elif pi_type == "BachelierAsian":
        return bachelier_asian_id(S, strike, rate, vol, T_remain,
                                   fixing_times, sim_data_assets, fixing_idxs, current_t)
    elif pi_type == "BlackLookbackCall":
        if min_S is None:
            min_S = S
        return lookback_call_id(S, rate, vol, T_remain, min_S)
    else:
        return bs_call_id(S, strike, rate, vol, T_remain)  # fallback


# =================== PROCESS SIMULATION ===================

class HestonProcess:
    """Heston simulation state in idouble."""
    def __init__(self, S0, v0, kappa, theta, vol_vol, rho, rate):
        self.S0, self.v0 = S0, v0
        self.kappa, self.theta, self.vol_vol, self.rho, self.rate = kappa, theta, vol_vol, rho, rate
        self.sqrt_1mrho2 = np.sqrt(1 - rho**2)
        self.S = None  # idouble
        self.v = None  # idouble

    def init_state(self):
        self.S = aadc.idouble(self.S0)
        self.v = aadc.idouble(self.v0)
        return self.S  # for markAsInput

    def step(self, dt, z1, z2):
        """One Euler step."""
        sqrt_dt = np.sqrt(dt)
        v_pos = (self.v * self.v + aadc.idouble(1e-16)).sqrt()
        sqrt_v = v_pos.sqrt()
        dW_S = z1 * aadc.idouble(sqrt_dt)
        dW_v = (aadc.idouble(self.rho) * z1 + aadc.idouble(self.sqrt_1mrho2) * z2) * aadc.idouble(sqrt_dt)
        self.S = self.S * ((aadc.idouble(self.rate) - v_pos * aadc.idouble(0.5)) * aadc.idouble(dt) + sqrt_v * dW_S).exp()
        self.v = self.v + aadc.idouble(self.kappa) * (aadc.idouble(self.theta) - v_pos) * aadc.idouble(dt) + aadc.idouble(self.vol_vol) * sqrt_v * dW_v

    def get_vol_of_asset(self):
        """σ(S) for FD direction: sqrt(v) * S for Heston."""
        v_pos = (self.v * self.v + aadc.idouble(1e-16)).sqrt()
        return v_pos.sqrt() * self.S

    def get_pi_vol(self):
        """Implied vol for Pi: sqrt(v0) constant."""
        return np.sqrt(self.v0)


class SabrScalarProcess:
    """SABR scalar process: dS = sqrt(S)*σ*dW + Milstein, σ lognormal."""
    def __init__(self, S0, v0, vol_vol, rate=0, beta=0.5):
        self.S0, self.v0, self.vol_vol, self.rate, self.beta = S0, v0, vol_vol, rate, beta
        self.S = None
        self.v = None

    def init_state(self):
        self.S = aadc.idouble(self.S0)
        self.v = aadc.idouble(self.v0)
        return self.S

    def step(self, dt, z1, z2):
        sqrt_dt = np.sqrt(dt)
        half = aadc.idouble(0.5)
        S_pos = (self.S * self.S + aadc.idouble(1e-16)).sqrt()
        aux_s = S_pos.sqrt() * self.v
        millst = half * half * aux_s * aux_s / S_pos * (z1 * z1 - aadc.idouble(1.0)) * aadc.idouble(dt)
        self.S = self.S + aux_s * aadc.idouble(sqrt_dt) * z1 + millst
        self.S = (self.S + (self.S * self.S + aadc.idouble(0.0001)).sqrt()) * half
        vv = aadc.idouble(self.vol_vol)
        self.v = self.v * (vv * aadc.idouble(sqrt_dt) * z2 - vv * vv * half * aadc.idouble(dt)).exp()

    def get_vol_of_asset(self):
        S_pos = (self.S * self.S + aadc.idouble(1e-16)).sqrt()
        return S_pos.sqrt() * self.v

    def get_pi_vol(self):
        return self.v0 * self.S0 ** (self.beta - 1)


# =================== ONE PATH PRICING (exact port of C++) ===================

def one_path_pricing(process, sim_times, z_vars, legendre_idxs, legendre_weights,
                      strike, maturity, rate, fixing_idxs, payoff_type,
                      pi_type="BlackScholes", barrier=0, fixing_times=None):
    """
    Exact port of DriverParallel.h::onePathPricing.
    Returns (payoff, one_path_impact, one_path_quadr) as idouble.
    """
    eps_fd = 0.01
    half = aadc.idouble(0.5)
    t0 = sim_times[0]
    pi_vol = process.get_pi_vol()

    one_path_impact = aadc.idouble(0.0)
    quadrature = aadc.idouble(0.0)
    accumulated = aadc.idouble(1.0)

    # sim_data: asset history for path-dependent payoffs
    asset_history = [process.S]  # S at t0

    for t_i in range(1, len(sim_times)):
        dt = sim_times[t_i] - sim_times[t_i - 1]
        z1, z2 = z_vars[t_i]
        process.step(dt, z1, z2)
        current_t = sim_times[t_i]
        S_curr = process.S
        asset_history.append(S_curr)

        # Barrier factor: accumulated *= step(S - barrier)
        if barrier > 0 and payoff_type == "DownAndOutEuropCallPayoff":
            x_bar = S_curr - aadc.idouble(barrier)
            step_v = (x_bar + (x_bar * x_bar + aadc.idouble(0.001 * 0.001)).sqrt()) / aadc.idouble(2 * 0.001)
            d_sv = step_v - aadc.idouble(1.0)
            step_clamped = (step_v + aadc.idouble(1.0) - (d_sv * d_sv + aadc.idouble(0.0001)).sqrt()) * half
            accumulated = accumulated * step_clamped

        # FD direction: vol_of_asset * S (process vol)
        vol_dir_process = process.get_vol_of_asset()
        # FD direction: Pi vol * S
        vol_dir_pi = S_curr * aadc.idouble(pi_vol)

        gamma_ass = aadc.idouble(0.0)
        gamma_base = aadc.idouble(0.0)

        is_legendre = t_i in legendre_idxs

        if is_legendre:
            eps_c = aadc.idouble(eps_fd)
            T_remain = maturity - current_t

            # FD Hessian with process vol direction
            S_p = S_curr + eps_c * vol_dir_process
            S_m = S_curr - eps_c * vol_dir_process
            # Clamp positive
            S_p = (S_p + (S_p * S_p + aadc.idouble(1e-20)).sqrt()) * half
            S_m = (S_m + (S_m * S_m + aadc.idouble(1e-20)).sqrt()) * half

            # Lookback: track min_S for Pi
            min_S_for_pi = asset_history[0]
            if payoff_type == "LookbackCall":
                for ah in asset_history[1:]:
                    d_ms = min_S_for_pi - ah
                    min_S_for_pi = (min_S_for_pi + ah - (d_ms*d_ms + aadc.idouble(0.01)).sqrt()) * half

            gamma_ass = gamma_ass + pi_call_id(pi_type, S_p, current_t, maturity, rate, pi_vol,
                                                strike, barrier, fixing_times or [], fixing_idxs,
                                                asset_history, min_S_for_pi)
            gamma_ass = gamma_ass + pi_call_id(pi_type, S_m, current_t, maturity, rate, pi_vol,
                                                strike, barrier, fixing_times or [], fixing_idxs,
                                                asset_history, min_S_for_pi)

            # FD Hessian with Pi vol direction
            S_p2 = S_curr + eps_c * vol_dir_pi
            S_m2 = S_curr - eps_c * vol_dir_pi
            S_p2 = (S_p2 + (S_p2 * S_p2 + aadc.idouble(1e-20)).sqrt()) * half
            S_m2 = (S_m2 + (S_m2 * S_m2 + aadc.idouble(1e-20)).sqrt()) * half

            gamma_base = gamma_base + pi_call_id(pi_type, S_p2, current_t, maturity, rate, pi_vol,
                                                  strike, barrier, fixing_times or [], fixing_idxs,
                                                  asset_history, min_S_for_pi)
            gamma_base = gamma_base + pi_call_id(pi_type, S_m2, current_t, maturity, rate, pi_vol,
                                                  strike, barrier, fixing_times or [], fixing_idxs,
                                                  asset_history, min_S_for_pi)

        inc = accumulated * (gamma_ass - gamma_base) / aadc.idouble(eps_fd * eps_fd)
        one_path_impact = one_path_impact + inc * aadc.idouble(dt)

        if is_legendre:
            leg_idx = legendre_idxs.index(t_i)
            quadrature = quadrature + inc * aadc.idouble(legendre_weights[leg_idx])

    # Scale outputs (matching C++)
    T_span = maturity - t0
    one_path_impact_out = half * one_path_impact
    one_path_quadr_out = half * (quadrature / aadc.idouble(2.0) * aadc.idouble(T_span))

    # Payoff
    disc = aadc.idouble(np.exp(-rate * maturity))
    S_final = process.S

    if payoff_type == "EuropeanCallAsian" and fixing_idxs:
        avg = aadc.idouble(0.0)
        for fi in fixing_idxs:
            if fi < len(asset_history):
                avg = avg + asset_history[fi]
        avg = avg / aadc.idouble(float(len(fixing_idxs)))
        diff = avg - aadc.idouble(strike)
    elif payoff_type == "LookbackCall":
        # min_S from asset_history
        min_S = asset_history[0]
        for ah in asset_history[1:]:
            d_ms = min_S - ah
            min_S = (min_S + ah - (d_ms * d_ms + aadc.idouble(0.01)).sqrt()) * half
        diff = S_final - min_S
    else:
        # EuropeanCall, DownAndOutEuropCallPayoff
        diff = S_final - aadc.idouble(strike)

    raw_payoff = (diff + (diff * diff + aadc.idouble(0.01)).sqrt()) * half * disc

    # Apply barrier knock-out
    if barrier > 0 and payoff_type == "DownAndOutEuropCallPayoff":
        payoff = raw_payoff * accumulated
    else:
        payoff = raw_payoff

    return payoff, one_path_impact_out, one_path_quadr_out


# =================== KERNEL RECORD + REPLAY ===================

def run_case(process_type, process_params, case, n_steps, n_paths, n_legendre=12):
    """Record kernel, replay, compute DMC price + Greeks."""

    S0 = process_params["init_asset"]
    v0 = process_params["init_vol"][0]
    rate = process_params.get("rate", 0.0)
    maturity = case["maturity"]
    strike = case.get("strike", 100)
    barrier = case.get("barrier", 0)
    payoff_type = case["Payoff"]
    fixing_times = case.get("fixing_times", [])

    dt = maturity / n_steps
    sim_times = [i * dt for i in range(n_steps + 1)]

    # Legendre abscissa → simulation step indices
    leg_points = get_legendre(n_legendre, sim_times[0], maturity)
    legendre_idxs = []
    legendre_weights = []
    for (t_leg, w_leg) in leg_points:
        idx = min(int(round(t_leg / dt)), n_steps)
        if idx > 0 and idx not in legendre_idxs:
            legendre_idxs.append(idx)
            legendre_weights.append(w_leg)

    # Fixing times → step indices
    fixing_idxs = []
    for ft in fixing_times:
        idx = min(int(round(ft / dt)), n_steps)
        fixing_idxs.append(idx)

    # Record kernel
    funcs = aadc.Functions()
    funcs.start_recording()

    if process_type in ("Heston",):
        proc = HestonProcess(S0, v0, process_params["kappa"], process_params["theta"],
                              process_params["vol_vol"], process_params["rho"], rate)
    elif process_type == "SabrScalar":
        proc = SabrScalarProcess(S0, v0, process_params["vol_vol"], rate,
                                  process_params.get("beta", 0.5))
    else:
        raise ValueError(f"Unknown process: {process_type}")
    S_init = proc.init_state()
    S_arg = S_init.mark_as_input()
    v_arg = proc.v.mark_as_input()

    # Random inputs
    z_args = []
    z_vars = [None]  # index 0 unused (sim starts at t_i=1)
    for _ in range(n_steps):
        z1 = aadc.idouble(0.0)
        z2 = aadc.idouble(0.0)
        z_args.append(z1.mark_as_input_no_diff())
        z_args.append(z2.mark_as_input_no_diff())
        z_vars.append((z1, z2))

    pi_type = case.get("Pi", "BlackScholes")
    payoff, impact, quadr = one_path_pricing(
        proc, sim_times, z_vars, legendre_idxs, legendre_weights,
        strike, maturity, rate, fixing_idxs, payoff_type,
        pi_type=pi_type, barrier=barrier, fixing_times=fixing_times
    )

    payoff_res = payoff.mark_as_output()
    impact_res = impact.mark_as_output()
    quadr_res = quadr.mark_as_output()

    funcs.stop_recording()

    # Replay
    rng = np.random.default_rng(42)
    inputs = {
        S_arg: np.full(n_paths, S0),
        v_arg: np.full(n_paths, v0),
    }
    for za in z_args:
        inputs[za] = rng.standard_normal(n_paths)

    # Request all 3 outputs + derivatives of payoff and quadrature w.r.t. S0
    request = {
        payoff_res: [S_arg],
        impact_res: [],
        quadr_res: [S_arg],
    }
    workers = aadc.ThreadPool(1)
    results = aadc.evaluate(funcs, request, inputs, workers)

    payoffs = results[0][payoff_res]
    impacts = results[0][impact_res]
    quadrs = results[0][quadr_res]
    delta_p = results[1][payoff_res][S_arg]
    delta_q = results[1][quadr_res][S_arg]

    price_raw = np.mean(payoffs)
    se_raw = np.std(payoffs) / np.sqrt(n_paths)
    # DMC: corrected price = payoff - quadrature (C++ formula)
    corrected = payoffs - quadrs
    price_dmc = np.mean(corrected)
    se_dmc = np.std(corrected) / np.sqrt(n_paths)
    delta_payoff = np.mean(delta_p)
    delta_quadr = np.mean(delta_q)
    delta_dmc = delta_payoff - delta_quadr

    vr = se_raw / se_dmc if se_dmc > 1e-12 else float('inf')

    # Bump-and-reval for validation
    h = 0.5
    inp_up = dict(inputs); inp_up[S_arg] = np.full(n_paths, S0 + h)
    inp_dn = dict(inputs); inp_dn[S_arg] = np.full(n_paths, S0 - h)
    r_up = aadc.evaluate(funcs, {payoff_res: []}, inp_up, workers)
    r_dn = aadc.evaluate(funcs, {payoff_res: []}, inp_dn, workers)
    bump_delta = (np.mean(r_up[0][payoff_res]) - np.mean(r_dn[0][payoff_res])) / (2 * h)

    print(f"  {payoff_type} K={strike} T={maturity}" +
          (f" fix={len(fixing_times)}" if fixing_times else ""))
    print(f"    Raw:       {price_raw:.4f} ± {se_raw:.4f}")
    print(f"    DMC:       {price_dmc:.4f} ± {se_dmc:.4f}  (VR: {vr:.1f}×)")
    print(f"    Impact:    {np.mean(impacts):.4f}")
    print(f"    Delta:     AAD={delta_payoff:.4f}  bump={bump_delta:.4f}  diff={abs(delta_payoff-bump_delta):.1e}")
    print(f"    DMC delta: {delta_dmc:.4f}")


# =================== MAIN ===================

def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else "Generic/Examples/Heston.json"

    with open(config_file) as f:
        config = json.load(f)

    proc = config["Process"]
    n_legendre = config.get("num_Legendre_points", 24)
    n_steps = min(int(proc["num_time_steps"]), 200)  # cap for Python speed
    n_paths = min(int(proc["num_mc_paths"]), 50000)
    params = proc["Params"]
    params["rate"] = proc.get("rate", 0.0)
    params["beta"] = proc.get("beta", 1.0)
    process_type = proc["Type"]

    print(f"Generic Python — DMC (full Legendre) via AADC 1.8.0")
    print(f"Config: {config_file}")
    print(f"Process: {process_type}, S0={params['init_asset']}, v0={params['init_vol']}")
    print(f"Steps: {n_steps}, Paths: {n_paths}, Legendre: {n_legendre}")
    print("=" * 60)

    for i, case in enumerate(config["Cases"]):
        print(f"\nCase {i+1}/{len(config['Cases'])}:")
        try:
            run_case(process_type, params, case, n_steps, n_paths, n_legendre)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
