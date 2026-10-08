"""Fixed-power optimization used by the selected-distance power sweeps."""

import numpy as np
from scipy.optimize import brentq, differential_evolution

import system_model as model


def robust(power_mw, distance_km, seed=1):
    def score(candidate):
        fc, fq, va = map(float, candidate)
        lower, upper = model._delta_interval(fc, fq)
        if upper < lower:
            return -1e6

        result = model.evaluate(fc, fq, upper, va, power_mw, distance_km)
        violation = max(0, model.SNR_MIN_DB - result["SNR_c_dB"])
        violation += max(0, -result["C_receiver_GHz"])
        violation += max(0, -result["C_reverse_Mbps"]) / 100
        violation += 100 * max(0, -result["secret_fraction"])
        violation += max(0, result["R_PE_Mbps"] - result["R_c_net_Mbps"]) / 100
        return result["SCR_Mbps"] - 1e3 * violation

    fit = differential_evolution(
        lambda candidate: -score(candidate),
        [model.FC_BOUNDS, model.FQ_BOUNDS, model.VA_BOUNDS],
        seed=seed,
        popsize=8,
        maxiter=45,
        tol=1e-7,
        polish=True,
    )
    fc, fq, va = map(float, fit.x)
    lower, upper = model._delta_interval(fc, fq)
    optimum = model.evaluate(fc, fq, upper, va, power_mw, distance_km)
    primary_rate = optimum["SCR_Mbps"]
    if primary_rate <= 0 or not model._feasible(optimum):
        return {"L_km": distance_km, "P_c_mW": power_mw, "SCR_Mbps": 0.0}

    # Among points within 0.1% of the rate maximum, select the smallest gap.
    target_rate = 0.999 * primary_rate

    def rate_difference(delta):
        return model.evaluate(fc, fq, delta, va, power_mw, distance_km)["SCR_Mbps"] - target_rate

    selected = lower if rate_difference(lower) >= 0 else brentq(
        rate_difference, lower, upper, xtol=1e-10
    )
    result = model.evaluate(fc, fq, selected, va, power_mw, distance_km)
    result["SCR_primary_Mbps"] = primary_rate
    separated_rate = model.optimize_separated(distance_km)["SKR_sep_Mbps"]
    result["SKR_sep_Mbps"] = separated_rate
    result["penalty_dB"] = (
        10 * np.log10(separated_rate / result["SCR_Mbps"])
        if result["SCR_Mbps"] > 0 else np.inf
    )
    return result
