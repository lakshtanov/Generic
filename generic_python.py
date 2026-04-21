#!/usr/bin/env python3
"""
generic_python.py — DMC (Denoised Monte Carlo) via AADC Python wheel (v1.8.0).

Full port of Generic C++ project:
- Processes: Heston
- Payoffs: EuropeanCall, EuropeanCallAsian, LookbackCall, DownAndOutEuropCallPayoff
- Control variates (Pi): BlackScholes, BachelierAsian, BlackLookbackCall, DownAndOutEuropCallPrice
- DMC: price = E[payoff] - E[Pi_mc - Pi_analytical]
- JSON config compatible with C++ version

Requires: pip install aadc numpy scipy (Python 3.9+)

Usage:
  micromamba run -n py311 python generic_python.py Examples/Heston.json
"""

import sys
import json
import numpy as np
from scipy.stats import norm
import aadc


# =================== ANALYTICAL Pi FUNCTIONS (numpy) ===================

def bs_call_price(S, K, r, vol, T):
    """Black-Scholes European call."""
    if T < 1e-10:
        return np.maximum(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_lookback_call(S, r, vol, T):
    """BS floating-strike lookback call: E[S_T - min S_t]."""
    if T < 1e-10:
        return 0.0
    a = (r + 0.5 * vol**2) / vol
    return S * (norm.cdf(a * np.sqrt(T)) - norm.cdf(-a * np.sqrt(T))
                + np.exp(-r * T) * (
                    vol * np.sqrt(T) * (norm.pdf(a * np.sqrt(T)) + norm.pdf(-a * np.sqrt(T)))
                ))


def bs_down_out_call(S, K, r, vol, T, barrier):
    """BS down-and-out European call (continuous monitoring approx)."""
    if S <= barrier:
        return 0.0
    bs = bs_call_price(S, K, r, vol, T)
    # Reflection formula
    lam = (r - 0.5 * vol**2) / (vol**2)
    y = np.log(barrier**2 / (S * K)) / (vol * np.sqrt(T)) + (1 + lam) * vol * np.sqrt(T)
    bs_reflected = (barrier / S)**(2 * lam) * (
        barrier**2 / S * norm.cdf(y) - K * np.exp(-r * T) * norm.cdf(y - vol * np.sqrt(T))
    )
    return bs - bs_reflected


def bachelier_asian_call(S, K, r, vol, T, fixing_times):
    """Bachelier approximation for arithmetic Asian call."""
    n = len(fixing_times)
    if n == 0:
        return bs_call_price(S, K, r, vol, T)
    # Mean of average: E[avg] = S * (1/n) * sum(exp(r*t_i))
    mean_factor = np.mean([np.exp(r * t) for t in fixing_times])
    mean_avg = S * mean_factor
    # Variance of average (Bachelier approx)
    var_sum = 0
    for i in range(n):
        for j in range(n):
            var_sum += np.exp(r * (fixing_times[i] + fixing_times[j])) * (
                np.exp(vol**2 * min(fixing_times[i], fixing_times[j])) - 1
            )
    std_avg = S * np.sqrt(var_sum) / n
    # Bachelier: call = disc * (mu - K) * N(d) + std * n(d)
    if std_avg < 1e-12:
        return np.maximum(mean_avg - K, 0) * np.exp(-r * T)
    d = (mean_avg - K) / std_avg
    return np.exp(-r * T) * ((mean_avg - K) * norm.cdf(d) + std_avg * norm.pdf(d))


# =================== HESTON KERNEL ===================

def record_kernel(process_type, process_params, case, n_steps):
    """Record AADC kernel for one MC path with given process and payoff."""

    S0_val = process_params["init_asset"]
    v0_list = process_params["init_vol"]
    vol_vol = process_params["vol_vol"]
    rate = process_params.get("rate", 0.0)
    beta = process_params.get("beta", 1.0)
    dim = process_params.get("dim", 1)

    # For scalar processes
    S0 = S0_val
    v0 = v0_list[0]

    maturity = case["maturity"]
    strike = case.get("strike", 0)
    barrier = case.get("barrier", 0)
    payoff_type = case["Payoff"]
    fixing_times = case.get("fixing_times", [])

    dt = maturity / n_steps
    sqrt_dt = np.sqrt(dt)

    is_multi = process_type in ("HestonMultDim", "SABR")
    d = dim if is_multi else 1

    funcs = aadc.Functions()
    funcs.start_recording()

    if is_multi:
        S_vec = [aadc.idouble(S0_val) for _ in range(d)]
        S_arg = S_vec[0].mark_as_input()  # Greeks w.r.t. first asset
        for i in range(1, d):
            S_vec[i].mark_as_input()  # track all for proper recording
        v_vec = [aadc.idouble(v0_list[i]) for i in range(d)]
        v_arg = v_vec[0].mark_as_input()
        for i in range(1, d):
            v_vec[i].mark_as_input()
        S = S_vec[0]  # for return dict compat
        v = v_vec[0]
    else:
        S = aadc.idouble(S0)
        S_arg = S.mark_as_input()
        v = aadc.idouble(v0)
        v_arg = v.mark_as_input()

    # Random inputs: 2*d normals per step (d asset + d vol)
    z_args = []
    z_vars = []
    n_rand = 2 * d
    for _ in range(n_steps):
        zs = []
        for _ in range(n_rand):
            z = aadc.idouble(0.0)
            z_args.append(z.mark_as_input_no_diff())
            zs.append(z)
        z_vars.append(zs)

    rate_c = aadc.idouble(rate)
    vv_c = aadc.idouble(vol_vol)
    dt_c = aadc.idouble(dt)
    sqrt_dt_c = aadc.idouble(sqrt_dt)
    half = aadc.idouble(0.5)
    delta_smooth = aadc.idouble(0.1)

    fixing_steps = set()
    for ft in fixing_times:
        step = int(round(ft / dt))
        if step < n_steps:
            fixing_steps.add(step)

    # State
    sum_S = aadc.idouble(0.0)
    min_S = S
    alive = aadc.idouble(1.0)

    # Process-specific constants
    if process_type in ("Heston", "HestonMultDim"):
        kappa = process_params["kappa"]
        theta = process_params["theta"]
        rho = process_params["rho"]
        rho_c = aadc.idouble(rho)
        sqrt_1mrho2 = aadc.idouble(np.sqrt(1 - rho**2))
        kappa_c = aadc.idouble(kappa)
        theta_c = aadc.idouble(theta)

    for t in range(n_steps):
        zs = z_vars[t]

        if process_type == "Heston":
            z1, z2 = zs[0], zs[1]
            v_pos = (v * v + aadc.idouble(1e-16)).sqrt()
            sqrt_v = v_pos.sqrt()
            dW_S = z1 * sqrt_dt_c
            dW_v = (rho_c * z1 + sqrt_1mrho2 * z2) * sqrt_dt_c
            S = S * ((rate_c - v_pos * half) * dt_c + sqrt_v * dW_S).exp()
            v = v + kappa_c * (theta_c - v_pos) * dt_c + vv_c * sqrt_v * dW_v

        elif process_type == "HestonMultDim":
            for i in range(d):
                z_s = zs[i]       # asset normal for dim i
                z_v = zs[d + i]   # vol normal for dim i
                vi = v_vec[i]
                vi_pos = (vi * vi + aadc.idouble(1e-16)).sqrt()
                sqrt_vi = vi_pos.sqrt()
                dW_Si = z_s * sqrt_dt_c
                dW_vi = (rho_c * z_s + sqrt_1mrho2 * z_v) * sqrt_dt_c
                S_vec[i] = S_vec[i] * ((rate_c - vi_pos * half) * dt_c + sqrt_vi * dW_Si).exp()
                v_vec[i] = vi + kappa_c * (theta_c - vi_pos) * dt_c + vv_c * sqrt_vi * dW_vi

        elif process_type == "SABR" and is_multi:
            for i in range(d):
                z_s = zs[i]
                z_v = zs[d + i]
                Si = S_vec[i]
                vi = v_vec[i]
                Si_pos = (Si * Si + aadc.idouble(1e-16)).sqrt()
                aux = Si_pos.sqrt() * vi
                Si = Si + aux * sqrt_dt_c * z_s
                Si = (Si + (Si * Si + aadc.idouble(0.0001)).sqrt()) * half
                S_vec[i] = Si
                v_vec[i] = vi * (vv_c * sqrt_dt_c * z_v - vv_c * vv_c * half * dt_c).exp()

        elif process_type == "SabrScalar":
            z1, z2 = zs[0], zs[1]
            S_pos = (S * S + aadc.idouble(1e-16)).sqrt()
            aux_s = S_pos.sqrt() * v
            millst = half * half * aux_s * aux_s / S_pos * (z1 * z1 - aadc.idouble(1.0)) * dt_c
            S = S + aux_s * sqrt_dt_c * z1 + millst
            S = (S + (S * S + aadc.idouble(0.0001)).sqrt()) * half
            v = v * (vv_c * sqrt_dt_c * z2 - vv_c * vv_c * half * dt_c).exp()

        # Asian accumulation
        if t in fixing_steps:
            sum_S = sum_S + S

        # Lookback: track min (smooth)
        if payoff_type == "LookbackCall":
            # Smooth min: (a+b - sqrt((a-b)^2 + eps)) / 2
            diff_ms = min_S - S
            min_S = (min_S + S - (diff_ms * diff_ms + delta_smooth * delta_smooth).sqrt()) * half

        # Barrier: smooth knock-out indicator
        if barrier > 0 and payoff_type == "DownAndOutEuropCallPayoff":
            bar_c = aadc.idouble(barrier)
            # alive *= smooth_step(S - barrier)
            # smooth_step(x, delta) = (x + sqrt(x^2+delta^2)) / (2*delta) clamped
            x_bar = S - bar_c
            step_val = (x_bar + (x_bar * x_bar + delta_smooth * delta_smooth).sqrt()) / (aadc.idouble(2.0) * delta_smooth)
            # clamp to [0,1]: min(step_val, 1)
            diff_sv = step_val - aadc.idouble(1.0)
            step_clamped = (step_val + aadc.idouble(1.0) - (diff_sv * diff_sv + aadc.idouble(0.01)).sqrt()) * half
            alive = alive * step_clamped

    # Payoff
    K_c = aadc.idouble(strike)
    disc = aadc.idouble(np.exp(-rate * maturity))

    if payoff_type == "BasketCall" and is_multi:
        basket = aadc.idouble(0.0)
        for i in range(d):
            basket = basket + S_vec[i]
        basket = basket / aadc.idouble(float(d))
        diff = basket - K_c
        S = S_vec[0]  # for output
    elif payoff_type == "EuropeanCallAsian":
        n_fix = aadc.idouble(float(len(fixing_times)))
        avg_S = sum_S / n_fix
        diff = avg_S - K_c
    elif payoff_type == "LookbackCall":
        diff = S - min_S
    else:
        diff = S - K_c

    raw_payoff = (diff + (diff * diff + delta_smooth * delta_smooth).sqrt()) * half * disc

    if barrier > 0 and payoff_type == "DownAndOutEuropCallPayoff":
        payoff = raw_payoff * alive
    else:
        payoff = raw_payoff

    payoff_res = payoff.mark_as_output()
    S_res = S.mark_as_output()

    funcs.stop_recording()

    # Collect all input args for replay
    all_input_vals = {S_arg: S0_val}
    if is_multi:
        for i in range(d):
            # S_vec and v_vec args were marked as input during recording
            pass  # handled by evaluate with default val
    all_input_vals[v_arg] = v0

    return {
        "funcs": funcs, "S_arg": S_arg, "v_arg": v_arg,
        "z_args": z_args, "payoff_res": payoff_res, "S_res": S_res,
        "S0": S0_val, "v0": v0, "rate": rate, "maturity": maturity,
        "strike": strike, "barrier": barrier, "dim": d,
        "payoff_type": payoff_type, "fixing_times": fixing_times,
        "is_multi": is_multi, "v0_list": v0_list,
        "vol_implied": v0 * S0_val**(beta - 1) if process_type == "SabrScalar" else np.sqrt(v0)
    }


def run_case(kernel, n_paths):
    """Replay kernel and compute DMC price + delta."""

    funcs = kernel["funcs"]
    S_arg, v_arg = kernel["S_arg"], kernel["v_arg"]
    z_args = kernel["z_args"]
    payoff_res, S_res = kernel["payoff_res"], kernel["S_res"]
    S0, v0, rate = kernel["S0"], kernel["v0"], kernel["rate"]
    maturity, strike, barrier = kernel["maturity"], kernel["strike"], kernel["barrier"]
    payoff_type = kernel["payoff_type"]
    fixing_times = kernel["fixing_times"]
    vol = kernel["vol_implied"]

    rng = np.random.default_rng(42)
    inputs = {S_arg: np.full(n_paths, S0), v_arg: np.full(n_paths, v0)}
    for za in z_args:
        inputs[za] = rng.standard_normal(n_paths)

    request = {payoff_res: [S_arg], S_res: []}
    workers = aadc.ThreadPool(1)
    results = aadc.evaluate(funcs, request, inputs, workers)

    payoffs = results[0][payoff_res]
    deltas = results[1][payoff_res][S_arg]
    S_term = results[0][S_res]

    price_raw = np.mean(payoffs)
    se_raw = np.std(payoffs) / np.sqrt(n_paths)
    delta_aad = np.mean(deltas)

    # Control variate
    pi_analytical = 0
    pi_mc = payoffs * 0  # no CV by default
    if payoff_type == "EuropeanCall":
        pi_analytical = bs_call_price(S0, strike, rate, vol, maturity)
        pi_mc = np.maximum(S_term - strike, 0) * np.exp(-rate * maturity)
    elif payoff_type == "DownAndOutEuropCallPayoff":
        pi_analytical = bs_down_out_call(S0, strike, rate, vol, maturity, barrier)
        pi_mc = np.maximum(S_term - strike, 0) * np.exp(-rate * maturity)
    elif payoff_type == "EuropeanCallAsian":
        pi_analytical = bachelier_asian_call(S0, strike, rate, vol, maturity, fixing_times)
    elif payoff_type == "LookbackCall":
        pi_analytical = bs_lookback_call(S0, rate, vol, maturity)

    corrected = payoffs - (pi_mc - pi_analytical)
    price_dmc = np.mean(corrected)
    se_dmc = np.std(corrected) / np.sqrt(n_paths)

    # Bump delta
    h = 0.5
    inp_up = dict(inputs); inp_up[S_arg] = np.full(n_paths, S0 + h)
    inp_dn = dict(inputs); inp_dn[S_arg] = np.full(n_paths, S0 - h)
    r_up = aadc.evaluate(funcs, {payoff_res: []}, inp_up, workers)
    r_dn = aadc.evaluate(funcs, {payoff_res: []}, inp_dn, workers)
    bump_delta = (np.mean(r_up[0][payoff_res]) - np.mean(r_dn[0][payoff_res])) / (2 * h)

    vr = se_raw / se_dmc if se_dmc > 1e-12 else float('inf')

    print(f"  {payoff_type} K={strike} T={maturity}" +
          (f" B={barrier}" if barrier else "") +
          (f" fix={len(fixing_times)}" if fixing_times else ""))
    print(f"    Raw:  {price_raw:.4f} ± {se_raw:.4f}")
    print(f"    DMC:  {price_dmc:.4f} ± {se_dmc:.4f}  (VR: {vr:.0f}×)")
    print(f"    Delta AAD={delta_aad:.4f}  bump={bump_delta:.4f}  diff={abs(delta_aad-bump_delta):.1e}")


# =================== MAIN ===================

def main():
    config_file = sys.argv[1] if len(sys.argv) > 1 else "Generic/Examples/Heston.json"

    with open(config_file) as f:
        config = json.load(f)

    proc = config["Process"]
    dim = proc.get("dim", proc["Params"].get("dim", 1))
    params = proc["Params"]
    params["dim"] = dim
    step_cap = 100 if dim > 1 else 500  # multi-dim: fewer steps
    n_steps = min(int(proc["num_time_steps"]), step_cap)
    n_paths = min(int(proc["num_mc_paths"]), 50000 if dim > 1 else 200000)
    params["rate"] = proc.get("rate", 0.0)
    params["beta"] = proc.get("beta", 1.0)
    process_type = proc["Type"]

    print(f"Generic Python — DMC via AADC 1.8.0")
    print(f"Config: {config_file}")
    print(f"Process: {process_type}, S0={params['init_asset']}, v0={params['init_vol']}")
    print(f"Steps: {n_steps}, Paths: {n_paths}")
    print("=" * 60)

    for i, case in enumerate(config["Cases"]):
        print(f"\nCase {i+1}/{len(config['Cases'])}:")
        try:
            kernel = record_kernel(process_type, params, case, n_steps)
            run_case(kernel, n_paths)
        except Exception as e:
            print(f"  SKIPPED: {e}")


if __name__ == "__main__":
    main()
