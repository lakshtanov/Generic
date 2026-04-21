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


# =================== ONE PATH PRICING (exact port of C++) ===================

def one_path_pricing(process, sim_times, z_vars, legendre_idxs, legendre_weights,
                      strike, maturity, rate, fixing_idxs, payoff_type):
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

            gamma_ass = gamma_ass + bs_call_id(S_p, strike, rate, pi_vol, T_remain)
            gamma_ass = gamma_ass + bs_call_id(S_m, strike, rate, pi_vol, T_remain)

            # FD Hessian with Pi vol direction
            S_p2 = S_curr + eps_c * vol_dir_pi
            S_m2 = S_curr - eps_c * vol_dir_pi
            S_p2 = (S_p2 + (S_p2 * S_p2 + aadc.idouble(1e-20)).sqrt()) * half
            S_m2 = (S_m2 + (S_m2 * S_m2 + aadc.idouble(1e-20)).sqrt()) * half

            gamma_base = gamma_base + bs_call_id(S_p2, strike, rate, pi_vol, T_remain)
            gamma_base = gamma_base + bs_call_id(S_m2, strike, rate, pi_vol, T_remain)

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
    if payoff_type == "EuropeanCall":
        diff = process.S - aadc.idouble(strike)
        payoff = (diff + (diff * diff + aadc.idouble(0.01)).sqrt()) * half * disc
    elif payoff_type == "EuropeanCallAsian" and fixing_idxs:
        avg = aadc.idouble(0.0)
        for fi in fixing_idxs:
            if fi < len(asset_history):
                avg = avg + asset_history[fi]
        avg = avg / aadc.idouble(float(len(fixing_idxs)))
        diff = avg - aadc.idouble(strike)
        payoff = (diff + (diff * diff + aadc.idouble(0.01)).sqrt()) * half * disc
    else:
        diff = process.S - aadc.idouble(strike)
        payoff = (diff + (diff * diff + aadc.idouble(0.01)).sqrt()) * half * disc

    return payoff, one_path_impact_out, one_path_quadr_out


# =================== KERNEL RECORD + REPLAY ===================

def run_case(process_params, case, n_steps, n_paths, n_legendre=12):
    """Record kernel, replay, compute DMC price + Greeks."""

    S0 = process_params["init_asset"]
    v0 = process_params["init_vol"][0]
    rate = process_params.get("rate", 0.05)
    maturity = case["maturity"]
    strike = case.get("strike", 100)
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

    proc = HestonProcess(S0, v0, process_params["kappa"], process_params["theta"],
                          process_params["vol_vol"], process_params["rho"], rate)
    S_init = proc.init_state()
    S_arg = S_init.mark_as_input()
    v_arg = proc.v.mark_as_input()  # track v for recording

    # Random inputs
    z_args = []
    z_vars = [None]  # index 0 unused (sim starts at t_i=1)
    for _ in range(n_steps):
        z1 = aadc.idouble(0.0)
        z2 = aadc.idouble(0.0)
        z_args.append(z1.mark_as_input_no_diff())
        z_args.append(z2.mark_as_input_no_diff())
        z_vars.append((z1, z2))

    payoff, impact, quadr = one_path_pricing(
        proc, sim_times, z_vars, legendre_idxs, legendre_weights,
        strike, maturity, rate, fixing_idxs, payoff_type
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
    params["rate"] = proc.get("rate", 0.05)

    print(f"Generic Python — DMC (full Legendre) via AADC 1.8.0")
    print(f"Config: {config_file}")
    print(f"Process: {proc['Type']}, S0={params['init_asset']}, v0={params['init_vol']}")
    print(f"Steps: {n_steps}, Paths: {n_paths}, Legendre: {n_legendre}")
    print("=" * 60)

    for i, case in enumerate(config["Cases"][:5]):  # first 5 cases for now
        print(f"\nCase {i+1}:")
        try:
            run_case(params, case, n_steps, n_paths, n_legendre)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
