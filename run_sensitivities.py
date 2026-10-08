#!/usr/bin/env python3
"""Reoptimize the transmitter-noise and transport-efficiency scenarios."""

import numpy as np
import pandas as pd

import system_model as model


def transmitter_noise_sweep():
    rows = []
    try:
        for distance in (25.0, 100.0):
            for admitted_fraction in (0.0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
                model.OOB_ADMITTED_FRACTION = admitted_fraction
                result = model.optimize_full_distance(distance)
                result["oob_admitted_fraction"] = admitted_fraction
                result["floor_ratio_dB"] = (
                    10 * np.log10(admitted_fraction) if admitted_fraction else -np.inf
                )
                rows.append(result)
                print(f"transmitter noise: {distance:g} km, {admitted_fraction:g}", flush=True)
    finally:
        model.OOB_ADMITTED_FRACTION = 0.0

    pd.DataFrame(rows).to_csv(
        model.RESULTS / "transmitter_noise_sensitivity.csv", index=False
    )


def transport_efficiency_sweep():
    rows = []
    try:
        for distance in (25.0, 100.0):
            for efficiency in (0.90, 0.95, 1.00):
                model.CLASSICAL_NET_EFFICIENCY = efficiency
                result = model.optimize_full_distance(distance)
                result["classical_net_efficiency"] = efficiency
                rows.append(result)
                print(f"transport: {distance:g} km, efficiency {efficiency:g}", flush=True)
    finally:
        model.CLASSICAL_NET_EFFICIENCY = 0.95

    pd.DataFrame(rows).to_csv(
        model.RESULTS / "transport_efficiency_sensitivity.csv", index=False
    )


if __name__ == "__main__":
    transmitter_noise_sweep()
    transport_efficiency_sweep()
