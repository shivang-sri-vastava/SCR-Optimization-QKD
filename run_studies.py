#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar

import system_model as m

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
R.mkdir(exist_ok=True)

# Parameter ledger: every row is a cited input, derivation, or declared design value.
parameters = [
    ("T_ch,25", m.T_REF, "1", "Ricard thesis, representative 25-km operating point"),
    ("xi_B,ref", m.XI_B_REF, "SNU", "Adopted channel-output-referred convention for the reported 4.3-mSNU platform value"),
    ("xi_A", m.XI_A, "SNU", "Derived as xi_B,ref/T_ch,25 at Alice's reference plane"),
    ("xi_rec", m.XI_REC, "SNU", "Ricard thesis, Ref. [28]"),
    ("eta_rec", m.ETA_REC, "1", "Ricard thesis, platform receiver calibration"),
    ("eta_det", m.ETA_DET, "1", "Ricard thesis, platform photodiode efficiency"),
    ("eta_DSP", m.ETA_DSP, "1", "Ricard thesis, optimized RRC-mode efficiency"),
    ("eta_B", m.ETA_B, "1", "Derived product eta_rec*eta_det*eta_DSP"),
    ("beta", m.BETA, "1", "Ricard thesis support/security calculation"),
    ("nu_PE", m.NU_PE, "1", "Ricard thesis support-traffic calculation"),
    ("quantizer_bits_per_quadrature", m.K_QUANT_BITS_PER_QUADRATURE, "bit", "Declared six-bit scalar quantizer on each heterodyne quadrature"),
    ("quantizer_range", m.K_QUANT_RANGE_SIGMA, "sigma_B", "Declared symmetric clipping limit on each heterodyne quadrature"),
    ("B_RX", m.B_RX_GHZ, "GHz", "Adopted usable receiver budget informed by the platform bandwidth"),
    ("xi_rec,a", m.XI_REC_BW_INTERCEPT, "SNU", "Derived receiver-noise model intercept"),
    ("xi_rec,c", m.XI_REC_BW_QUADRATIC, "SNU/GHz^2", "Derived receiver-noise model quadratic coefficient"),
    ("rho_q", m.RHO_Q, "1", "Ricard thesis, 125-MBd 64-QAM platform signal"),
    ("rho_c", m.RHO_C, "1", "Ricard thesis, 4-GBd QPSK platform signal"),
    ("alpha_f", m.ALPHA_DB_PER_KM, "dB/km", "Derived from the reported 5-dB loss over 25 km"),
    ("PER", m.PER_DB, "dB", "Declared central spectral-design value; sensitivity 20-40 dB"),
    ("filter_order", m.FILTER_ORDER, "1", "Declared central Butterworth order; sensitivity 2,4,6"),
    ("beta_R", m.BETA_R_REF, "km^-1 nm^-1", "Representative effective design value informed by Kumar; not calibrated at the optimized GHz-scale detuning"),
    ("lambda_q", m.LAMBDA_Q_NM, "nm", "Quantum-carrier wavelength used in the analytical Raman conversion"),
    ("B_tot", m.B_TOTAL_GHZ, "GHz", "Declared equivalent-baseband design limit"),
    ("B_guard", m.B_GUARD_GHZ, "GHz", "Declared edge-to-edge guard-band design limit"),
    ("eta_c", m.CLASSICAL_NET_EFFICIENCY, "1", "Assumed net forward transport efficiency; sensitivity 0.90,0.95,1.00"),
    ("R_back", m.RETURN_CAPACITY_MBPS, "Mbit/s", "Provisioned net authenticated return capacity on a separate fiber"),
    ("f_EC", m.EC_PROVISION_FACTOR, "1", "Conservative transport allowance multiplier, independent of security efficiency beta"),
    ("epsilon_OOB", m.OOB_ADMITTED_FRACTION, "1", "Nominal ideal-spectrum limit; additional filter-admitted power fraction swept separately"),
    ("P_c,14dB", m.reference_power_mW(), "mW", "In-range reference fit at 14 dB, 4 GBd and 25 km"),
    ("R_auth", m.R_AUTH_MBPS, "Mbit/s", "Declared scope value; fixed allowance excluded"),
    ("P_c,0", m.P_C_0_MW, "mW", "Assumed physical reference for the digitized relative-power scale"),
]
pd.DataFrame(parameters, columns=["parameter", "value", "unit", "basis"]).to_csv(R / "parameter_ledger.csv", index=False)

fit = pd.DataFrame([{
    "slope": m.REL_POWER_SLOPE,
    "intercept_dB": m.REL_POWER_INTERCEPT,
    "RMSE_dB": m.REL_POWER_FIT_RMSE_DB,
    "N_markers": 15,
}])
fit.to_csv(R / "classical_snr_fit.csv", index=False)

sep25 = m.optimize_separated(25.0)
pd.DataFrame([sep25]).to_csv(R / "separated_reference_25km.csv", index=False)

# Trust-boundary sensitivity with identical total Alice-to-measurement gain.
trust_rows = []
for label, T, eta in [
    ("receiver_and_DSP_trusted", m.T_REF, m.ETA_B),
    ("receiver_trusted_DSP_in_channel", m.T_REF * m.ETA_DSP, m.ETA_REC * m.ETA_DET),
]:
    row = m.optimize_separated(25.0, eta_B=eta, T_override=T)
    row.update({"scenario": label, "T_security": T, "eta_B": eta, "total_gain": T * eta})
    trust_rows.append(row)
pd.DataFrame(trust_rows).to_csv(R / "trust_boundary_sensitivity.csv", index=False)

# Spectral sweep. Dense near the transition and coarser on the plateau.
deltas = np.unique(np.round(
    np.r_[np.arange(0.70, 2.051, 0.05), np.arange(2.2, 6.001, 0.2), 3.0], 10
))
spectral_rows = []
for i, delta in enumerate(deltas, 1):
    d = m.optimize_fixed_delta(float(delta), 25.0)
    d["SKR_sep_Mbps"] = sep25["SKR_sep_Mbps"]
    d["penalty_dB"] = 10.0 * np.log10(sep25["SKR_sep_Mbps"] / d["SCR_Mbps"]) \
        if d.get("SCR_Mbps", 0.0) > 0 else np.inf
    spectral_rows.append(d)
    print(f"spectral {i}/{len(deltas)} delta={delta:.3f} SCR={d.get('SCR_Mbps',0):.6f}", flush=True)
spectral = (pd.DataFrame(spectral_rows).sort_values("delta_f_GHz")
            .drop_duplicates(subset="delta_f_GHz", keep="first"))
spectral.to_csv(R / "spectral_sweep_25km.csv", index=False)

# Fixed platform at 4 GBd / 125 MBd / 3 GHz; optimize only V_A and use minimum SNR power.
pc_fixed = m.required_power_mW(4.0, 25.0)
def fixed_obj(va):
    return -m.evaluate(4.0, 0.125, 3.0, float(va), pc_fixed, 25.0)["SCR_Mbps"]
ropt = minimize_scalar(fixed_obj, bounds=m.VA_BOUNDS, method="bounded", options={"xatol": 2e-7})
fixed = m.evaluate(4.0, 0.125, 3.0, float(ropt.x), pc_fixed, 25.0)
# The separated comparator for this row is restricted to the same 125-MBd
# quantum symbol rate.  The independently optimized 269.23-MBd reference is
# retained in separated_reference_25km.csv for the co-design studies.
sep_fixed = m.BACKEND.optimize(m.T_ch(25.0), m.XI_A, m.ETA_B, m.XI_REC, m.BETA)
fixed["SKR_sep_Mbps"] = 1e3 * 0.125 * (1.0 - m.NU_PE) * max(sep_fixed.secret_fraction, 0.0)
fixed["penalty_dB"] = 10.0 * np.log10(fixed["SKR_sep_Mbps"] / fixed["SCR_Mbps"])
pd.DataFrame([fixed]).to_csv(R / "platform_configuration_25km.csv", index=False)

# Joint distance optimization.
distance_rows = []
for L in np.arange(25.0, 100.1, 5.0):
    d = m.optimize_full_distance(float(L))
    distance_rows.append(d)
    print(f"distance L={L:.0f} SCR={d['SCR_Mbps']:.6f} Pc={d['P_c_mW']:.6f}", flush=True)
distance = pd.DataFrame(distance_rows).sort_values("L_km")
distance.to_csv(R / "distance_joint_optimization.csv", index=False)
distance[["L_km", "xi_A_SNU", "xi_leak_SNU", "xi_Raman_SNU",
          "xi_excess_input_SNU"]].to_csv(R / "noise_components_distance.csv", index=False)

# Power sweeps at selected distances; all other variables are reoptimized.
from run_robust_power_sweep import robust
power_rows = []
for L in [25.0, 50.0, 75.0, 100.0]:
    pstar = float(distance.loc[np.isclose(distance["L_km"], L), "P_c_mW"].iloc[0])
    base_powers = np.geomspace(max(m.PC_BOUNDS_MW[0], pstar / 8), min(m.PC_BOUNDS_MW[1], pstar * 8), 25)
    base_powers = base_powers[~np.isclose(base_powers, pstar, rtol=0.0, atol=1e-10)]
    powers = np.sort(np.r_[base_powers, pstar])
    reference_row = distance.loc[np.isclose(distance["L_km"], L)].iloc[0].to_dict()
    for j, p in enumerate(powers, 1):
        if np.isclose(float(p), pstar, rtol=0.0, atol=1e-12):
            d = dict(reference_row)
        else:
            d = robust(float(p), L, seed=1000 + int(L) * 10 + j)
        power_rows.append(d)
        print(f"power L={L:.0f} {j}/{len(powers)} P={p:.6f} SCR={d.get('SCR_Mbps',0):.6f}", flush=True)
pd.DataFrame(power_rows).to_csv(R / "power_sweep_selected_distances.csv", index=False)

# Aggregate shared-noise boundaries at 25 km.  The shared-versus-separated
# hardware threshold is 10*log10(6/4) dB for six resources reduced to four.
sep_rate = sep25["SKR_sep_Mbps"]
def solve_rate_fraction(fraction: float) -> float:
    target = fraction * sep_rate
    return brentq(lambda x: m.shared_noise_rate(x) - target, 0.0, 0.25, xtol=1e-11)
xi_be = solve_rate_fraction(2.0 / 3.0)
xi_zero = brentq(lambda x: m.shared_noise_rate(x) - 1e-8, 0.0, 0.35, xtol=1e-11)
thresholds = pd.DataFrame([
    {"boundary": "shared_vs_separated_hardware_threshold", "penalty_dB": 10.0 * np.log10(6.0 / 4.0),
     "xi_shared_SNU": xi_be, "SCR_Mbps": m.shared_noise_rate(xi_be)},
    {"boundary": "zero_key", "penalty_dB": np.inf,
     "xi_shared_SNU": xi_zero, "SCR_Mbps": 0.0},
])
thresholds.to_csv(R / "shared_noise_thresholds_25km.csv", index=False)

# Raman-coefficient sensitivity at the longest distance.
raman_rows = []
for beta_R in [1.5e-9, 3.0e-9, 3.1e-9]:
    d = m.optimize_full_distance(100.0, beta_R=beta_R)
    d["beta_R"] = beta_R
    raman_rows.append(d)
pd.DataFrame(raman_rows).to_csv(R / "raman_sensitivity_100km.csv", index=False)

# Reference-informed receiver-bandwidth/electronic-noise co-design at 25 km.
from run_receiver_bandwidth_codesign import run as run_receiver_codesign
run_receiver_codesign()

print("Studies completed.")
