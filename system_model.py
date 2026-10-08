#!/usr/bin/env python3
"""Physical SCR model and nested optimizers for Sections IV-V."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar, brentq

from key_rate_model import FastTrustedKeyRate, joint_quantized_heterodyne_information

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Platform and protocol values. Sources and derivations are stated in Sections IV-V.
L_REF_KM = 25.0
T_REF = 0.28
# The manuscript adopts the reported 4.3 mSNU value as a channel-output-
# referred platform contribution before trusted detection.  Its input-
# referred value is 4.3 mSNU/T_REF = 15.36 mSNU.
XI_B_REF = 4.3e-3
XI_A = XI_B_REF / T_REF
# Baseline detector electronic-noise variance at B_RX=0.35 GHz.
XI_REC = 0.09
ALPHA_DB_PER_KM = 0.2
ETA_REC = 0.60
ETA_DET = 0.88
ETA_DSP = 0.884
ETA_B = ETA_REC * ETA_DET * ETA_DSP
BETA = 0.95
NU_PE = 0.10
M_Q_BITS = 6.0
K_QUANT_BITS_PER_QUADRATURE = 6
K_QUANT_RANGE_SIGMA = 4.0
R_AUTH_MBPS = 0.0
# Declared transport scenarios, not measured coding/backhaul specifications.
CLASSICAL_NET_EFFICIENCY = 0.95
RETURN_CAPACITY_MBPS = 10000.0  # net reliable authenticated Bob-to-Alice service
EC_PROVISION_FACTOR = 1.0 / 0.95  # capacity allowance, independent of security BETA
OOB_ADMITTED_FRACTION = 0.0  # nominal ideal-spectrum limit; swept separately

RHO_C = 0.10
RHO_Q = 0.30
B_RX_GHZ = 0.35
# Reference-informed bandwidth law used only by the receiver co-design study.
# The two anchors are (0.35 GHz, 0.09 SNU) and (0.75 GHz, 0.15 SNU).
XI_REC_BW_INTERCEPT = 0.07329545454545455
XI_REC_BW_QUADRATIC = 0.13636363636363635
B_TOTAL_GHZ = 6.8
B_GUARD_GHZ = 0.12
FC_BOUNDS = (0.05, 4.0)
FQ_BOUNDS = (0.05, min(1.0, B_RX_GHZ / (1.0 + RHO_Q)))
DELTA_BOUNDS = (0.35, 6.0)
VA_BOUNDS = (0.5, 12.0)
PC_BOUNDS_MW = (0.004, 4.0)

H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
LAMBDA_Q_M = 1550e-9
LAMBDA_Q_NM = LAMBDA_Q_M / 1e-9
NU_Q_HZ = C_LIGHT / LAMBDA_Q_M
PER_DB = 30.0
ETA_POL = 10.0 ** (-PER_DB / 10.0)
FILTER_ORDER = 4
BETA_R_REF = 3.0e-9
KM_TO_M = 1.0e3
MW_TO_W = 1.0e-3
BETA_R_KM_INV_NM_INV_TO_M_INV2 = 1.0e6

# Assumed physical zero for the digitized relative-power curve.  The source
# curve does not supply an absolute-power reference, so predictions involving
# absolute classical power, leakage, or Raman noise are conditional on it.
P_C_0_MW = 1.0
F_C_REF_GBD = 4.0
SNR_MIN_DB = 14.0
_digitized = pd.read_csv(DATA / "platform_snr_digitized.csv")
REL_POWER_SLOPE, REL_POWER_INTERCEPT = np.polyfit(
    _digitized["SNR_dB"], _digitized["relative_launch_power_dB"], 1
)
REL_POWER_FIT_RMSE_DB = float(np.sqrt(np.mean((
    _digitized["relative_launch_power_dB"]
    - (REL_POWER_SLOPE * _digitized["SNR_dB"] + REL_POWER_INTERCEPT)
) ** 2)))

BACKEND = FastTrustedKeyRate(
    va_min=VA_BOUNDS[0], va_max=VA_BOUNDS[1], va_step=0.05,
    cutoff=60, gh_order=32, snr_max=10.0, snr_step=0.005,
)


def receiver_electronic_noise(b_rx_ghz: float) -> float:
    """Trusted electronic noise for the adopted bandwidth model.

    The bandwidth is expressed in GHz and the returned variance is in SNU.
    """
    bandwidth = float(b_rx_ghz)
    if bandwidth <= 0.0:
        raise ValueError("Receiver bandwidth must be positive")
    return float(XI_REC_BW_INTERCEPT + XI_REC_BW_QUADRATIC * bandwidth**2)


def T_ch(L_km: float) -> float:
    return float(np.clip(T_REF * 10.0 ** (-ALPHA_DB_PER_KM * (L_km - L_REF_KM) / 10.0), 1e-12, 1.0))


def raised_cosine_psd(f_hz: np.ndarray, symbol_rate_hz: float, rolloff: float) -> np.ndarray:
    af = np.abs(f_hz)
    f1 = (1.0 - rolloff) * symbol_rate_hz / 2.0
    f2 = (1.0 + rolloff) * symbol_rate_hz / 2.0
    out = np.zeros_like(af)
    out[af <= f1] = 1.0 / symbol_rate_hz
    mask = (af > f1) & (af < f2)
    if rolloff > 0:
        out[mask] = 0.5 / symbol_rate_hz * (
            1.0 + np.cos(np.pi * (af[mask] - f1) / (rolloff * symbol_rate_hz))
        )
    return out


def spectral_overlap(fc_GBd: float, fq_GBd: float, delta_GHz: float,
                     filter_order: int = FILTER_ORDER, points: int = 601) -> float:
    fc_hz, fq_hz, delta_hz = fc_GBd * 1e9, fq_GBd * 1e9, delta_GHz * 1e9
    half = (1.0 + RHO_C) * fc_hz / 2.0
    f = np.linspace(delta_hz - half, delta_hz + half, points)
    spectrum = raised_cosine_psd(f - delta_hz, fc_hz, RHO_C)
    cutoff = (1.0 + RHO_Q) * fq_hz / 2.0
    H2 = 1.0 / (1.0 + (np.abs(f) / max(cutoff, 1e-30)) ** (2 * filter_order))
    return float(np.clip(np.trapezoid(spectrum * H2, f), 0.0, 1.0))


def leakage_noise(pc_mW: float, L_km: float, fc_GBd: float, fq_GBd: float,
                  delta_GHz: float, per_db: float = PER_DB,
                  filter_order: int = FILTER_ORDER) -> tuple[float, float]:
    """Return input-referred leakage excess noise and spectral overlap."""
    omega = spectral_overlap(fc_GBd, fq_GBd, delta_GHz, filter_order) + OOB_ADMITTED_FRACTION
    eta_pol = 10.0 ** (-per_db / 10.0)
    xi = 2.0 * eta_pol * pc_mW * 1e-3 * omega / (
        H_PLANCK * NU_Q_HZ * fq_GBd * 1e9
    )
    return float(xi), omega


def raman_noise(pc_mW: float, L_km: float, beta_R: float = BETA_R_REF) -> float:
    """Input-referred forward Raman excess noise from the SI expression.

    ``beta_R`` is supplied in km^-1 nm^-1, ``pc_mW`` in mW, and ``L_km``
    in km.  The Raman coefficient is converted to m^-2 before evaluating
    lambda^3 beta_R P_c L/(h c^2).
    """
    if pc_mW < 0.0 or L_km < 0.0 or beta_R < 0.0:
        raise ValueError("Raman inputs must be non-negative")
    beta_si = beta_R * BETA_R_KM_INV_NM_INV_TO_M_INV2
    power_w = pc_mW * MW_TO_W
    length_m = L_km * KM_TO_M
    return float(
        LAMBDA_Q_M**3 * beta_si * power_w * length_m
        / (H_PLANCK * C_LIGHT**2)
    )


def reference_power_mW(target_snr_db: float = SNR_MIN_DB) -> float:
    """Interpolate the measured reference-rate calibration at the target SNR."""
    return float(P_C_0_MW * 10.0**((REL_POWER_SLOPE*target_snr_db + REL_POWER_INTERCEPT)/10.0))


def classical_snr_db(pc_mW: float, fc_GBd: float, L_km: float) -> float:
    # Physical power/rate/loss scaling anchored at the in-range 14-dB fit point.
    return float(SNR_MIN_DB + 10.0*np.log10(max(pc_mW, 1e-30)/reference_power_mW())
                 + 10.0*np.log10(T_ch(L_km)/T_REF)
                 + 10.0*np.log10(F_C_REF_GBD/fc_GBd))


def required_power_mW(fc_GBd: float, L_km: float, target_snr_db: float = SNR_MIN_DB) -> float:
    return float(reference_power_mW() * (fc_GBd/F_C_REF_GBD) * (T_REF/T_ch(L_km))
                 * 10.0**((target_snr_db-SNR_MIN_DB)/10.0))


def support_bits_per_qsymbol(H_xb: float, I_x_xb: float) -> tuple[float, float]:
    """Return total support bits and reconciliation bits per quantum symbol."""
    lambda_ec = max(H_xb - I_x_xb, 0.0) * EC_PROVISION_FACTOR
    total = NU_PE * M_Q_BITS + (1.0 - NU_PE) * lambda_ec
    return float(total), float(lambda_ec)


def occupied_bandwidths(fc_GBd: float, fq_GBd: float) -> tuple[float, float]:
    return (1.0 + RHO_C) * fc_GBd, (1.0 + RHO_Q) * fq_GBd


def evaluate(fc_GBd: float, fq_GBd: float, delta_GHz: float, V_A: float,
             pc_mW: float, L_km: float, beta_R: float = BETA_R_REF,
             eta_B: float = ETA_B, T_override: float | None = None,
             per_db: float = PER_DB, filter_order: int = FILTER_ORDER) -> dict:
    T = T_ch(L_km) if T_override is None else T_override
    xi_leak, omega = leakage_noise(pc_mW, L_km, fc_GBd, fq_GBd,
                                    delta_GHz, per_db, filter_order)
    xi_ram = raman_noise(pc_mW, L_km, beta_R)
    xi_excess_input = XI_A + xi_leak + xi_ram
    xi_tot = XI_REC + xi_excess_input
    kr = BACKEND.evaluate(V_A, T, xi_excess_input, eta_B, XI_REC, BETA)
    H_xb, I_x_xb = joint_quantized_heterodyne_information(
        V_A, T, xi_excess_input, eta_B, XI_REC,
        bits_per_quadrature=K_QUANT_BITS_PER_QUADRATURE,
        range_sigma=K_QUANT_RANGE_SIGMA,
    )
    support_bits, lambda_ec = support_bits_per_qsymbol(H_xb, I_x_xb)
    secret = max(float(kr.secret_fraction), 0.0) if np.isfinite(kr.secret_fraction) else 0.0
    Kq = 1000.0 * fq_GBd * (1.0 - NU_PE) * secret
    Rsup = 1000.0 * fq_GBd * support_bits + R_AUTH_MBPS
    Rpe = 1000.0 * fq_GBd * NU_PE * M_Q_BITS + R_AUTH_MBPS
    Rec = 1000.0 * fq_GBd * (1.0 - NU_PE) * lambda_ec
    Rc = 2000.0 * CLASSICAL_NET_EFFICIENCY * fc_GBd
    SCR = max(min(Kq, Rc - Rpe), 0.0) if Rec <= RETURN_CAPACITY_MBPS + 1e-8 else 0.0
    Bc, Bq = occupied_bandwidths(fc_GBd, fq_GBd)
    return {
        "L_km": L_km, "P_c_mW": pc_mW, "fc_GBd": fc_GBd, "fq_GBd": fq_GBd,
        "delta_f_GHz": delta_GHz, "V_A_SNU": V_A, "T_ch": T,
        "eta_B": eta_B, "xi_rec_SNU": XI_REC, "xi_A_SNU": XI_A,
        "xi_leak_SNU": xi_leak, "xi_Raman_SNU": xi_ram,
        "xi_excess_input_SNU": xi_excess_input, "xi_tot_SNU": xi_tot,
        "spectral_overlap": omega, "I_XY": float(kr.mutual_information),
        "H_XB_quantized": H_xb, "I_X_XB_quantized": I_x_xb,
        "lambda_EC_bits_per_qsymbol": lambda_ec,
        "chi_YE": float(kr.holevo_information), "secret_fraction": float(kr.secret_fraction),
        "R_PE_Mbps": Rpe, "R_EC_Mbps": Rec,
        "R_return_net_Mbps": RETURN_CAPACITY_MBPS,
        "classical_net_efficiency": CLASSICAL_NET_EFFICIENCY,
        "oob_admitted_fraction": OOB_ADMITTED_FRACTION,
        "xi_oob_SNU": 2.0 * 10**(-per_db/10) * pc_mW * 1e-3 * OOB_ADMITTED_FRACTION / (H_PLANCK*NU_Q_HZ*fq_GBd*1e9),
        "C_reverse_Mbps": RETURN_CAPACITY_MBPS - Rec,
        "K_q_Mbps": Kq, "R_sup_Mbps": Rsup, "R_c_net_Mbps": Rc, "SCR_Mbps": SCR,
        "SNR_c_dB": classical_snr_db(pc_mW, fc_GBd, L_km),
        "C_key_Mbps": Kq - SCR, "C_classical_Mbps": Rc - Rpe - SCR,
        "C_inner_GHz": delta_GHz - (0.5 * (Bc + Bq) + B_GUARD_GHZ),
        "C_outer_GHz": B_TOTAL_GHZ - (delta_GHz + 0.5 * (Bc + Bq)),
        "C_receiver_GHz": B_RX_GHZ - Bq,
        "nu_1": kr.nu_1, "nu_2": kr.nu_2, "nu_3": kr.nu_3, "nu_4": kr.nu_4, "nu_5": kr.nu_5,
    }


def _feasible(d: dict, tol: float = 2e-5) -> bool:
    return (d["SNR_c_dB"] >= SNR_MIN_DB - tol
            and d["C_inner_GHz"] >= -tol and d["C_outer_GHz"] >= -tol
            and d["C_receiver_GHz"] >= -tol
            and d["secret_fraction"] >= -tol and d["R_c_net_Mbps"] >= d["R_PE_Mbps"] - tol and d["C_reverse_Mbps"] >= -tol)


def optimize_separated(L_km: float, eta_B: float = ETA_B, T_override: float | None = None) -> dict:
    T = T_ch(L_km) if T_override is None else T_override
    fq = FQ_BOUNDS[1]
    def objective(va):
        r = BACKEND.evaluate(float(va), T, XI_A, eta_B, XI_REC, BETA)
        return -r.secret_fraction if np.isfinite(r.secret_fraction) else 1e6
    opt = minimize_scalar(objective, bounds=VA_BOUNDS, method="bounded", options={"xatol": 2e-7})
    kr = BACKEND.evaluate(float(opt.x), T, XI_A, eta_B, XI_REC, BETA)
    return {
        "L_km": L_km, "V_A_SNU": float(opt.x), "fq_GBd": fq,
        "secret_fraction": float(kr.secret_fraction),
        "SKR_sep_Mbps": 1000.0 * fq * (1.0 - NU_PE) * max(float(kr.secret_fraction), 0.0),
        "nu_1": kr.nu_1, "nu_2": kr.nu_2, "nu_3": kr.nu_3, "nu_4": kr.nu_4, "nu_5": kr.nu_5,
    }


def optimize_fixed_delta(delta_GHz: float, L_km: float = 25.0,
                         per_db: float = PER_DB, filter_order: int = FILTER_ORDER,
                         beta_R: float = BETA_R_REF) -> dict:
    """Optimize fc, fq and V_A at fixed separation; P_c is the SNR-boundary power."""
    bounds = [FC_BOUNDS, FQ_BOUNDS, VA_BOUNDS]
    starts = [np.array([fc, fq, va]) for fc in [0.2, 0.7, 1.5, 3.5]
              for fq in [0.08, 0.18, FQ_BOUNDS[1]] for va in [3.5, 5.0]]

    def calc(x):
        fc, fq, va = map(float, x)
        pc = max(PC_BOUNDS_MW[0], required_power_mW(fc, L_km))
        return evaluate(fc, fq, delta_GHz, va, pc, L_km, beta_R=beta_R,
                        per_db=per_db, filter_order=filter_order)

    def cons(x):
        d = calc(x)
        return np.array([d["C_inner_GHz"], d["C_outer_GHz"], d["C_receiver_GHz"],
                         d["secret_fraction"], d["R_c_net_Mbps"] - d["R_PE_Mbps"], d["C_reverse_Mbps"],
                         PC_BOUNDS_MW[1] - d["P_c_mW"]])

    best = None
    constraint = {"type": "ineq", "fun": cons}
    for start in starts:
        r = minimize(lambda x: -calc(x)["SCR_Mbps"], start, method="SLSQP",
                     bounds=bounds, constraints=constraint,
                     options={"maxiter": 300, "ftol": 1e-10})
        d = calc(r.x)
        if _feasible(d) and d["P_c_mW"] <= PC_BOUNDS_MW[1] + 1e-5:
            if best is None or d["SCR_Mbps"] > best["SCR_Mbps"]:
                best = d
    if best is None:
        # Deterministic fallback.
        for fc in np.linspace(FC_BOUNDS[0], FC_BOUNDS[1], 25):
            for fq in np.linspace(FQ_BOUNDS[0], FQ_BOUNDS[1], 14):
                for va in np.linspace(1.0, 8.0, 20):
                    d = calc([fc, fq, va])
                    if _feasible(d) and d["P_c_mW"] <= PC_BOUNDS_MW[1] and (best is None or d["SCR_Mbps"] > best["SCR_Mbps"]):
                        best = d
    if best is None:
        return {"delta_f_GHz": delta_GHz, "SCR_Mbps": 0.0}
    return best


def _delta_interval(fc: float, fq: float) -> tuple[float, float]:
    Bc, Bq = occupied_bandwidths(fc, fq)
    lower = max(DELTA_BOUNDS[0], 0.5 * (Bc + Bq) + B_GUARD_GHZ)
    upper = min(DELTA_BOUNDS[1], B_TOTAL_GHZ - 0.5 * (Bc + Bq))
    return lower, upper


def optimize_fixed_power(pc_mW: float, L_km: float, beta_R: float = BETA_R_REF) -> dict:
    """Optimize fc, fq, V_A and select the minimum 99.9%-plateau separation."""
    bounds = [FC_BOUNDS, FQ_BOUNDS, VA_BOUNDS]
    starts = [np.array([fc, fq, va]) for fc in [0.15, 0.7, 1.5]
              for fq in [0.08, 0.18, FQ_BOUNDS[1]] for va in [3.5, 5.0]]

    def calc_primary(x):
        fc, fq, va = map(float, x)
        lo, hi = _delta_interval(fc, fq)
        if hi < lo:
            return None
        return evaluate(fc, fq, hi, va, pc_mW, L_km, beta_R=beta_R)

    def constraints(x):
        d = calc_primary(x)
        if d is None:
            return np.full(6, -1.0)
        lo, hi = _delta_interval(float(x[0]), float(x[1]))
        return np.array([hi - lo, d["C_receiver_GHz"], d["secret_fraction"],
                         d["R_c_net_Mbps"] - d["R_PE_Mbps"], d["C_reverse_Mbps"], d["SNR_c_dB"] - SNR_MIN_DB])

    best = None
    c = {"type": "ineq", "fun": constraints}
    for start in starts:
        r = minimize(lambda x: 1e5 if calc_primary(x) is None else -calc_primary(x)["SCR_Mbps"],
                     start, method="SLSQP", bounds=bounds, constraints=c,
                     options={"maxiter": 280, "ftol": 1e-10})
        d = calc_primary(r.x)
        if d is not None and _feasible(d) and (best is None or d["SCR_Mbps"] > best["SCR_Mbps"]):
            best = d
    if best is None:
        return {"L_km": L_km, "P_c_mW": pc_mW, "SCR_Mbps": 0.0}

    primary_scr = best["SCR_Mbps"]
    floor = 0.999 * primary_scr
    fc, fq, va = best["fc_GBd"], best["fq_GBd"], best["V_A_SNU"]
    lo, hi = _delta_interval(fc, fq)
    def plateau(delta):
        return evaluate(fc, fq, delta, va, pc_mW, L_km, beta_R=beta_R)["SCR_Mbps"] - floor
    if plateau(lo) >= 0:
        selected = lo
    else:
        selected = brentq(plateau, lo, hi, xtol=1e-10)
    result = evaluate(fc, fq, selected, va, pc_mW, L_km, beta_R=beta_R)
    result["SCR_primary_Mbps"] = primary_scr
    return result


def optimize_distance(L_km: float, beta_R: float = BETA_R_REF) -> dict:
    """Outer optimization of classical power at fixed distance."""
    # Broad logarithmic scan locates the operating region.
    powers = np.geomspace(PC_BOUNDS_MW[0], PC_BOUNDS_MW[1], 44)
    rows = [optimize_fixed_power(float(p), L_km, beta_R=beta_R) for p in powers]
    vals = np.asarray([r.get("SCR_Mbps", 0.0) for r in rows])
    i = int(np.argmax(vals))
    lo = powers[max(i - 2, 0)]
    hi = powers[min(i + 2, len(powers) - 1)]
    cache = {float(p): r for p, r in zip(powers, rows)}
    def objective(logp):
        p = float(10.0 ** logp)
        if p not in cache:
            cache[p] = optimize_fixed_power(p, L_km, beta_R=beta_R)
        return -cache[p].get("SCR_Mbps", 0.0)
    opt = minimize_scalar(objective, bounds=(np.log10(lo), np.log10(hi)), method="bounded",
                          options={"xatol": 2e-5, "maxiter": 60})
    pstar = float(10.0 ** opt.x)
    result = optimize_fixed_power(pstar, L_km, beta_R=beta_R)
    sep = optimize_separated(L_km)
    result["SKR_sep_Mbps"] = sep["SKR_sep_Mbps"]
    result["penalty_dB"] = 10.0 * np.log10(sep["SKR_sep_Mbps"] / result["SCR_Mbps"]) \
        if sep["SKR_sep_Mbps"] > 0 and result["SCR_Mbps"] > 0 else np.inf
    return result


def shared_noise_rate(xi_shared: float, L_km: float = 25.0) -> float:
    fq = FQ_BOUNDS[1]
    r = BACKEND.optimize(T_ch(L_km), XI_A + xi_shared, ETA_B, XI_REC, BETA)
    return 1000.0 * fq * (1.0 - NU_PE) * max(float(r.secret_fraction), 0.0)


def optimize_full_distance(L_km: float, beta_R: float = BETA_R_REF) -> dict:
    """Jointly optimize fc, fq, delta, V_A; Pc is the minimum SNR-feasible power.

    At fixed fc and L, any power above the SNR boundary leaves classical
    throughput unchanged and increases both leakage and Raman noise. Hence the
    power optimum is attained at Pc=required_power_mW(fc,L).
    """
    bounds = [FC_BOUNDS, FQ_BOUNDS, DELTA_BOUNDS, VA_BOUNDS]
    starts = [np.array([fc, fq, delta, va])
              for fc in [0.25, 0.7, 1.5]
              for fq in [0.10, 0.20, FQ_BOUNDS[1]]
              for delta in [1.2, 2.5, 4.5]
              for va in [3.5, 5.0]]

    def calc(x):
        fc, fq, delta, va = map(float, x)
        pc = max(PC_BOUNDS_MW[0], required_power_mW(fc, L_km))
        return evaluate(fc, fq, delta, va, pc, L_km, beta_R=beta_R)

    def cons(x):
        d = calc(x)
        return np.array([d["C_inner_GHz"], d["C_outer_GHz"], d["C_receiver_GHz"],
                         d["secret_fraction"], d["R_c_net_Mbps"] - d["R_PE_Mbps"], d["C_reverse_Mbps"],
                         PC_BOUNDS_MW[1] - d["P_c_mW"]])

    best = None
    c = {"type": "ineq", "fun": cons}
    for start in starts:
        r = minimize(lambda x: -calc(x)["SCR_Mbps"], start, method="SLSQP",
                     bounds=bounds, constraints=c,
                     options={"maxiter": 350, "ftol": 1e-10})
        d = calc(r.x)
        if _feasible(d) and d["P_c_mW"] <= PC_BOUNDS_MW[1] + 1e-5 and (best is None or d["SCR_Mbps"] > best["SCR_Mbps"]):
            best = d
    if best is None:
        return {"L_km": L_km, "SCR_Mbps": 0.0}

    primary = best["SCR_Mbps"]
    floor = 0.999 * primary
    fc, fq, va = best["fc_GBd"], best["fq_GBd"], best["V_A_SNU"]
    pc = best["P_c_mW"]
    lo, hi = _delta_interval(fc, fq)
    def plateau(delta):
        return evaluate(fc, fq, delta, va, pc, L_km, beta_R=beta_R)["SCR_Mbps"] - floor
    if plateau(lo) >= 0:
        selected = lo
    else:
        selected = brentq(plateau, lo, hi, xtol=1e-10)
    result = evaluate(fc, fq, selected, va, pc, L_km, beta_R=beta_R)
    result["SCR_primary_Mbps"] = primary
    sep = optimize_separated(L_km)
    result["SKR_sep_Mbps"] = sep["SKR_sep_Mbps"]
    result["penalty_dB"] = 10.0 * np.log10(sep["SKR_sep_Mbps"] / result["SCR_Mbps"]) \
        if sep["SKR_sep_Mbps"] > 0 and result["SCR_Mbps"] > 0 else np.inf
    return result
