#!/usr/bin/env python3
"""Generate the publication distance-dependence figure from committed results."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from plot_style import (  # noqa: E402
    COLORS,
    PANEL_BOX,
    TWO_COLUMN_FIGSIZE,
    apply_publication_style,
)

R = ROOT / "results"
F = ROOT / "figures"
F.mkdir(exist_ok=True)

apply_publication_style()

distance = pd.read_csv(R / "distance_joint_optimization.csv").sort_values("L_km")

fig, axes = plt.subplots(
    1, 2, figsize=TWO_COLUMN_FIGSIZE, constrained_layout=True,
)

ax = axes[0]
ax.semilogy(distance.L_km, 1e3 * distance.xi_leak_SNU,
            marker="s", color=COLORS["blue"], label=r"$\xi_{\rm leak}$")
ax.semilogy(distance.L_km, 1e3 * distance.xi_Raman_SNU,
            marker="^", color=COLORS["orange"], label=r"$\xi_{\rm Raman}$")
ax.set_xlabel("Fiber length (km)")
ax.set_ylabel("Excess noise (mSNU)")
ax.set_xlim(24, 101)
ax.grid(True, which="both", alpha=0.25)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.12, 0.98))
ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, va="top",
        fontweight="bold", bbox=PANEL_BOX)

ax = axes[1]
ax.semilogy(distance.L_km, distance.SKR_sep_Mbps,
            marker="o", color=COLORS["blue"], label="Separated SKR")
ax.semilogy(distance.L_km, distance.SCR_Mbps,
            marker="s", color=COLORS["orange"], linestyle="--",
            label="Shared SCR")
ax.set_xlabel("Fiber length (km)")
ax.set_ylabel("Rate (Mbit/s)")
ax.set_xlim(24, 101)
ax.set_ylim(top=20.0)
ax.grid(True, which="both", alpha=0.25)
ax.legend(frameon=False, loc="upper right")
ax.text(0.02, 0.96, "(b)", transform=ax.transAxes, va="top",
        fontweight="bold", bbox=PANEL_BOX)

fig.savefig(F / "fig5_noise_and_rate_distance.pdf", bbox_inches="tight")
fig.savefig(F / "fig5_noise_and_rate_distance.png", dpi=300, bbox_inches="tight")
plt.close(fig)

from run_receiver_bandwidth_codesign import make_figure as make_receiver_figure  # noqa: E402


receiver = pd.read_csv(R / "receiver_bandwidth_codesign_25km.csv")
make_receiver_figure(receiver)


print("Figures generated.")
