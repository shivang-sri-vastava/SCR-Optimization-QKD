#!/usr/bin/env python3
"""Three-point receiver-bandwidth/electronic-noise co-design study.

The calculation reuses the manuscript's trusted-receiver SCR backend.  At
L=25 km, each selected receiver bandwidth is paired with the adopted
electronic-noise law xi_rec(B_RX)=xi_0+kappa_RX B_RX^2.  The shared link and its matched
separated reference are then optimized independently using the same receiver
pair, with eta_B and xi_A held fixed.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import system_model as m  # noqa: E402
from plot_style import (  # noqa: E402
    COLORS,
    PANEL_BOX,
    TWO_COLUMN_FIGSIZE,
    apply_publication_style,
)


RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

BANDWIDTHS_GHZ = np.array([0.35, 0.50, 0.75])
L_KM = 25.0


def run() -> pd.DataFrame:
    """Reoptimize both architectures at the three selected bandwidths."""
    original_b_rx = m.B_RX_GHZ
    original_fq_bounds = m.FQ_BOUNDS
    original_xi_rec = m.XI_REC
    rows: list[dict] = []
    try:
        for bandwidth in BANDWIDTHS_GHZ:
            xi_rec = m.receiver_electronic_noise(float(bandwidth))
            m.B_RX_GHZ = float(bandwidth)
            m.FQ_BOUNDS = (
                original_fq_bounds[0],
                min(1.0, float(bandwidth) / (1.0 + m.RHO_Q)),
            )
            m.XI_REC = xi_rec

            shared = m.optimize_full_distance(L_KM)
            rows.append({
                "L_km": L_KM,
                "B_RX_GHz": float(bandwidth),
                "xi_rec_SNU": xi_rec,
                "xi_model_intercept_SNU": m.XI_REC_BW_INTERCEPT,
                "xi_model_quadratic_SNU_per_GHz2": m.XI_REC_BW_QUADRATIC,
                "fq_max_GBd": float(m.FQ_BOUNDS[1]),
                "fq_opt_GBd": float(shared["fq_GBd"]),
                "fc_opt_GBd": float(shared["fc_GBd"]),
                "delta_f_opt_GHz": float(shared["delta_f_GHz"]),
                "V_A_opt_SNU": float(shared["V_A_SNU"]),
                "P_c_opt_mW": float(shared["P_c_mW"]),
                "xi_leak_SNU": float(shared["xi_leak_SNU"]),
                "xi_Raman_SNU": float(shared["xi_Raman_SNU"]),
                "R_PE_Mbps": float(shared["R_PE_Mbps"]),
                "R_EC_Mbps": float(shared["R_EC_Mbps"]),
                "R_c_net_Mbps": float(shared["R_c_net_Mbps"]),
                "R_return_net_Mbps": float(shared["R_return_net_Mbps"]),
                "SCR_shared_Mbps": float(shared["SCR_Mbps"]),
                "SKR_separated_Mbps": float(shared["SKR_sep_Mbps"]),
                "sharing_penalty_dB": float(shared["penalty_dB"]),
            })
            print(
                f"B_RX={1e3 * bandwidth:.0f} MHz, xi_rec={xi_rec:.6f} SNU: "
                f"SCR={shared['SCR_Mbps']:.6f} Mbit/s, "
                f"penalty={shared['penalty_dB']:.6f} dB",
                flush=True,
            )
    finally:
        m.B_RX_GHZ = original_b_rx
        m.FQ_BOUNDS = original_fq_bounds
        m.XI_REC = original_xi_rec

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "receiver_bandwidth_codesign_25km.csv", index=False)
    return frame


def make_figure(frame: pd.DataFrame) -> None:
    """Create the publication figure from the three-point result table."""
    apply_publication_style()
    frame = frame.sort_values("B_RX_GHz")
    bandwidth_mhz = 1e3 * frame["B_RX_GHz"].to_numpy()

    fig, axes = plt.subplots(
        1, 2, figsize=TWO_COLUMN_FIGSIZE, constrained_layout=True,
    )

    axes[0].plot(
        bandwidth_mhz, frame["xi_rec_SNU"], marker="o",
        color=COLORS["blue"],
    )
    axes[0].set_xlabel(r"Receiver bandwidth $B_{\rm RX}$ (MHz)")
    axes[0].set_ylabel(r"Trusted electronic noise $\xi_{\rm rec}$ (SNU)")
    axes[0].set_xticks(bandwidth_mhz)
    axes[0].set_ylim(0.08, 0.16)
    axes[0].grid(True, alpha=0.25)
    axes[0].text(
        0.02, 0.96, "(a)", transform=axes[0].transAxes, va="top",
        fontweight="bold", bbox=PANEL_BOX,
    )

    axes[1].plot(
        bandwidth_mhz, frame["SKR_separated_Mbps"], marker="o",
        color=COLORS["blue"], label="Separated SKR",
    )
    axes[1].plot(
        bandwidth_mhz, frame["SCR_shared_Mbps"], marker="s", linestyle="--",
        color=COLORS["orange"], markerfacecolor="none", markeredgewidth=1.1,
        label="Shared SCR",
    )
    axes[1].set_xlabel(r"Receiver bandwidth $B_{\rm RX}$ (MHz)")
    axes[1].set_ylabel("Optimized rate (Mbit/s)")
    axes[1].set_xticks(bandwidth_mhz)
    axes[1].set_ylim(10.0, 26.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.12, 0.98))
    axes[1].text(
        0.02, 0.96, "(b)", transform=axes[1].transAxes, va="top",
        fontweight="bold", bbox=PANEL_BOX,
    )

    fig.savefig(FIGURES / "fig_receiver_bandwidth_codesign.pdf", bbox_inches="tight")
    fig.savefig(
        FIGURES / "fig_receiver_bandwidth_codesign.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":
    make_figure(run())
