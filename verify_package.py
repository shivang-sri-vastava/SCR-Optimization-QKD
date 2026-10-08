#!/usr/bin/env python3
"""Check stored numerical results against model identities and direct calculations."""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import differential_evolution

import system_model as model
from key_rate_model import evaluate_trusted_key_rate

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, rtol=3e-8, atol=1e-8):
    check(
        np.isclose(actual, expected, rtol=rtol, atol=atol),
        f"{message}: {actual} != {expected}",
    )


def check_model_identities():
    platform = pd.read_csv(RESULTS / "platform_configuration_25km.csv").iloc[0]
    separated = model.BACKEND.optimize(
        model.T_ch(25.0), model.XI_A, model.ETA_B, model.XI_REC, model.BETA
    )
    matched_rate = 1000 * 0.125 * (1 - model.NU_PE) * separated.secret_fraction
    close(platform.SKR_sep_Mbps, matched_rate, "Matched 125-MBd reference")

    # SI conversion is calculated independently of the model constants.
    raman_si = (1550e-9) ** 3 * (3e-9 * 1e6) * 0.001 * 25000
    raman_si /= 6.62607015e-34 * 299792458**2
    close(model.raman_noise(1.0, 25.0), raman_si, "Raman SI conversion",
          rtol=1e-12, atol=1e-15)
    close(model.classical_snr_db(model.reference_power_mW(), 4.0, 25.0),
          14.0, "Reference SNR")
    close(model.required_power_mW(2.0, 25.0),
          model.reference_power_mW() / 2, "Baud-rate power scaling")
    close(model.required_power_mW(4.0, 75.0),
          10 * model.reference_power_mW(), "Distance power scaling")
    close(model.T_ch(25.0), 0.28, "Reference transmittance")


def check_stored_points():
    tables = (
        "distance_joint_optimization.csv",
        "spectral_sweep_25km.csv",
        "platform_configuration_25km.csv",
        "power_sweep_selected_distances.csv",
        "raman_sensitivity_100km.csv",
        "transmitter_noise_sensitivity.csv",
        "transport_efficiency_sensitivity.csv",
    )
    checked = 0
    try:
        for name in tables:
            for _, row in pd.read_csv(RESULTS / name).iterrows():
                if row.SCR_Mbps <= 0:
                    continue
                checked += 1
                efficiency = float(row.get("classical_net_efficiency", 0.95))
                model.CLASSICAL_NET_EFFICIENCY = efficiency
                model.OOB_ADMITTED_FRACTION = float(row.get("oob_admitted_fraction", 0.0))
                raman_coefficient = float(row.get("beta_R", model.BETA_R_REF))
                result = model.evaluate(
                    row.fc_GBd, row.fq_GBd, row.delta_f_GHz, row.V_A_SNU,
                    row.P_c_mW, row.L_km, beta_R=raman_coefficient,
                )
                for column in (
                    "SCR_Mbps", "R_sup_Mbps", "R_PE_Mbps", "R_EC_Mbps",
                    "R_c_net_Mbps", "xi_leak_SNU", "xi_Raman_SNU",
                ):
                    close(row[column], result[column], f"{name}: {column}")
                check(model._feasible(result, tol=1e-4), f"{name}: infeasible point")
                check(result["C_classical_Mbps"] >= -1e-4,
                      f"{name}: forward capacity exceeded")
                check(model.PC_BOUNDS_MW[0] - 1e-7 <= row.P_c_mW
                      <= model.PC_BOUNDS_MW[1] + 1e-7, f"{name}: power bound")
                close(row.R_sup_Mbps, row.R_PE_Mbps + row.R_EC_Mbps,
                      "Directional support traffic")
                close(row.R_c_net_Mbps, 2000 * efficiency * row.fc_GBd,
                      "Net forward throughput")
                if "penalty_dB" in row and np.isfinite(row.penalty_dB):
                    close(row.penalty_dB,
                          10 * np.log10(row.SKR_sep_Mbps / row.SCR_Mbps),
                          "Sharing penalty")
                check(min(result[f"nu_{index}"] for index in range(1, 6))
                      >= 1 - 1e-7, "Covariance physicality")
    finally:
        model.OOB_ADMITTED_FRACTION = 0.0
        model.CLASSICAL_NET_EFFICIENCY = 0.95
    return checked


def check_direct_rates(distance):
    direct = []
    for length in (25.0, 50.0, 75.0, 100.0):
        row = distance[distance.L_km == length].iloc[0]
        key_rate = evaluate_trusted_key_rate(
            row.V_A_SNU, model.T_ch(length), row.xi_excess_input_SNU,
            model.ETA_B, model.XI_REC, model.BETA, cutoff=60, gh_order=32,
        )
        rate = 1000 * row.fq_GBd * (1 - model.NU_PE) * max(
            key_rate.secret_fraction, 0.0
        )
        close(rate, row.K_q_Mbps, "Direct security evaluator",
              rtol=2e-5, atol=3e-6)
        direct.append({"L_km": length, "direct_Mbps": rate,
                       "stored_Mbps": row.K_q_Mbps,
                       "difference_Mbps": rate - row.K_q_Mbps})
    return direct


def check_spectral_overlap(row):
    fc, fq, delta = row.fc_GBd, row.fq_GBd, row.delta_f_GHz

    def classical_psd(frequency):
        absolute = abs(frequency)
        inner, outer = 0.9 * fc / 2, 1.1 * fc / 2
        if absolute <= inner:
            return 1 / fc
        if absolute >= outer:
            return 0.0
        return (1 + np.cos(np.pi * (absolute - inner) / (0.1 * fc))) / (2 * fc)

    cutoff = 1.3 * fq / 2
    edges = [-1.1 * fc / 2, -0.9 * fc / 2, 0.9 * fc / 2, 1.1 * fc / 2]
    overlap = sum(
        quad(lambda frequency: classical_psd(frequency) /
             (1 + ((frequency + delta) / cutoff) ** 8),
             left, right, epsabs=1e-15, epsrel=1e-9)[0]
        for left, right in zip(edges[:-1], edges[1:])
    )
    close(overlap, row.spectral_overlap, "Independent overlap integral",
          rtol=2e-5, atol=1e-13)
    return overlap


def check_return_capacity(row):
    original = model.RETURN_CAPACITY_MBPS
    try:
        model.RETURN_CAPACITY_MBPS = 1.0
        blocked = model.evaluate(
            row.fc_GBd, row.fq_GBd, row.delta_f_GHz,
            row.V_A_SNU, row.P_c_mW, 25.0,
        )
        check(blocked["SCR_Mbps"] == 0 and blocked["C_reverse_Mbps"] < 0,
              "Return-capacity rejection")
    finally:
        model.RETURN_CAPACITY_MBPS = original


def check_receiver_points():
    original = (model.B_RX_GHZ, model.FQ_BOUNDS, model.XI_REC)
    try:
        for _, row in pd.read_csv(
            RESULTS / "receiver_bandwidth_codesign_25km.csv"
        ).iterrows():
            model.B_RX_GHZ = row.B_RX_GHz
            model.FQ_BOUNDS = (0.05, row.B_RX_GHz / 1.3)
            model.XI_REC = model.receiver_electronic_noise(row.B_RX_GHz)
            result = model.evaluate(
                row.fc_opt_GBd, row.fq_opt_GBd, row.delta_f_opt_GHz,
                row.V_A_opt_SNU, row.P_c_opt_mW, 25.0,
            )
            close(result["SCR_Mbps"], row.SCR_shared_Mbps, "Receiver rate")
            close(result["R_EC_Mbps"], row.R_EC_Mbps, "Receiver return load")
            check(model._feasible(result), "Receiver feasibility")
    finally:
        model.B_RX_GHZ, model.FQ_BOUNDS, model.XI_REC = original


def check_zero_rate_scenarios():
    checks = []
    floor = pd.read_csv(RESULTS / "transmitter_noise_sensitivity.csv")
    for _, row in floor[floor.SCR_Mbps <= 0].iterrows():
        # Relax forward throughput and intended leakage for an optimistic bound.
        power = max(model.PC_BOUNDS_MW[0],
                    model.required_power_mW(model.FC_BOUNDS[0], row.L_km))
        symbol_rate = model.FQ_BOUNDS[1]
        added_noise = (
            2 * model.ETA_POL * power * 1e-3 * row.oob_admitted_fraction
            / (model.H_PLANCK * model.NU_Q_HZ * symbol_rate * 1e9)
            + model.raman_noise(power, row.L_km)
        )
        relaxed_rate = model.shared_noise_rate(added_noise, row.L_km)
        check(relaxed_rate < 1e-8, "Uncorroborated zero-rate scenario")
        checks.append({"L_km": row.L_km,
                       "epsilon_OOB": row.oob_admitted_fraction,
                       "relaxed_rate_Mbps": relaxed_rate})
    return checks


def check_optimizer(distance):
    checks = []
    for length in (25.0, 100.0):
        target = distance[distance.L_km == length].iloc[0]

        def objective(candidate):
            fc, fq, va = map(float, candidate)
            lower, upper = model._delta_interval(fc, fq)
            if upper < lower:
                return 1e5
            power = max(model.PC_BOUNDS_MW[0], model.required_power_mW(fc, length))
            result = model.evaluate(fc, fq, upper, va, power, length)
            violation = max(0, -result["C_reverse_Mbps"])
            violation += max(0, -result["C_classical_Mbps"])
            violation += 1e3 * max(0, power - model.PC_BOUNDS_MW[1])
            return -result["SCR_Mbps"] + 1e3 * violation

        fit = differential_evolution(
            objective, [model.FC_BOUNDS, model.FQ_BOUNDS, model.VA_BOUNDS],
            seed=20260915 + int(length), popsize=10, maxiter=120,
            tol=1e-8, polish=True,
        )
        best_rate = -fit.fun
        check(best_rate <= target.SCR_primary_Mbps * 1.0002 + 2e-5,
              "Independent optimizer exceeds stored maximum")
        check(best_rate >= target.SCR_primary_Mbps * 0.999 - 2e-5,
              "Independent optimizer misses stored maximum")
        checks.append({"L_km": length, "DE_primary_Mbps": best_rate,
                       "SLSQP_primary_Mbps": target.SCR_primary_Mbps})
    return checks


def main():
    started = time.time()
    check_model_identities()
    checked = check_stored_points()
    distance = pd.read_csv(RESULTS / "distance_joint_optimization.csv")
    check(list(distance.L_km) == list(np.arange(25.0, 101.0, 5.0)),
          "Distance sweep coverage")
    direct = check_direct_rates(distance)
    overlap = check_spectral_overlap(distance.iloc[0])
    check_return_capacity(distance.iloc[0])
    check_receiver_points()
    zero_rates = check_zero_rate_scenarios()
    optimizers = check_optimizer(distance)

    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(RESULTS.glob("*.csv"))
    }
    report = {
        "status": "passed",
        "positive_operating_points_checked": checked,
        "direct_checks": direct,
        "zero_rate_checks": zero_rates,
        "optimizer_cross_checks": optimizers,
        "overlap_integral": overlap,
        "elapsed_seconds": time.time() - started,
        "result_sha256": hashes,
        "scope": "Finite numerical consistency checks; not a security proof or experimental validation.",
    }
    output = ROOT / (sys.argv[1] if len(sys.argv) > 1
                     else "validation/verification.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
