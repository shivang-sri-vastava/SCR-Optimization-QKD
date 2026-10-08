#!/usr/bin/env python3
"""Build the explanatory frequency-layout schematic used in the manuscript."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from plot_style import COLORS, apply_publication_style  # noqa: E402


def _double_arrow(ax, x0, x1, y, label, dy=0.06, color="#333333"):
    arrow = FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="<->", mutation_scale=8,
        linewidth=0.9, color=color,
    )
    ax.add_patch(arrow)
    ax.text((x0 + x1) / 2, y + dy, label, ha="center", va="bottom",
            color=color)


def build_figure() -> None:
    """Create a compact two-panel vector schematic with manuscript styling."""
    apply_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.35), constrained_layout=True)

    # Panel (a): shared-transmitter equivalent-baseband allocation.
    ax = axes[0]
    q0, q1 = 0.45, 2.25
    c0, c1 = 3.55, 6.40
    outer0, outer1 = 0.15, 6.75
    ax.add_patch(Rectangle((q0, 0.22), q1 - q0, 0.70,
                           facecolor=COLORS["blue"], alpha=0.22,
                           edgecolor=COLORS["blue"], linewidth=1.2))
    ax.add_patch(Rectangle((c0, 0.22), c1 - c0, 0.70,
                           facecolor=COLORS["orange"], alpha=0.22,
                           edgecolor=COLORS["orange"], linewidth=1.2))
    ax.text((q0 + q1) / 2, 0.57, "Quantum band", ha="center", va="center")
    ax.text((c0 + c1) / 2, 0.57, "Classical band", ha="center", va="center")
    q_center, c_center = (q0 + q1) / 2, (c0 + c1) / 2
    ax.plot([q_center, q_center], [0.12, 1.04], color=COLORS["blue"], lw=0.8)
    ax.plot([c_center, c_center], [0.12, 1.04], color=COLORS["orange"], lw=0.8)
    _double_arrow(ax, q0, q1, 1.05, r"$B_q$", dy=0.03)
    _double_arrow(ax, c0, c1, 1.05, r"$B_c$", dy=0.03)
    _double_arrow(ax, q1, c0, -0.02, r"clearance $\geq B_{\rm guard}$",
                  dy=0.02, color="#555555")
    _double_arrow(ax, q_center, c_center, -0.24, r"$\Delta f$", dy=0.02)
    _double_arrow(ax, outer0, outer1, -0.57, r"available span $B_{\rm tot}$",
                  dy=0.02)
    ax.plot([outer0, outer0], [-0.63, 0.12], color="#777777", lw=0.7)
    ax.plot([outer1, outer1], [-0.63, 0.12], color="#777777", lw=0.7)
    ax.text(0.01, 0.98, "(a)", transform=ax.transAxes, ha="left", va="top",
            fontweight="bold")
    ax.set_xlim(0, 6.9)
    ax.set_ylim(-0.78, 1.36)
    ax.set_xlabel("Equivalent-baseband frequency")
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)

    # Panel (b): receiver-mode abstraction after heterodyne translation.
    ax = axes[1]
    rx0, rx1 = -3.0, 3.0
    bq0, bq1 = -1.55, 1.55
    ax.add_patch(Rectangle((rx0, 0.16), rx1 - rx0, 0.82,
                           facecolor="#8c8c8c", alpha=0.12,
                           edgecolor="#555555", linewidth=1.0,
                           linestyle="--"))
    ax.add_patch(Rectangle((bq0, 0.30), bq1 - bq0, 0.54,
                           facecolor=COLORS["blue"], alpha=0.25,
                           edgecolor=COLORS["blue"], linewidth=1.2))
    ax.text(0, 0.57, "Recovered quantum band", ha="center", va="center")
    _double_arrow(ax, rx0, rx1, 1.12, r"usable receive width $B_{\rm RX}$",
                  dy=0.03)
    _double_arrow(ax, bq0, bq1, 0.03, r"occupied width $B_q$", dy=0.02)
    ax.axvline(0, ymin=0.11, ymax=0.54, color="#555555", lw=0.8)
    ax.text(0, -0.19, "DSP-centered at 0", ha="center", va="top")
    ax.text(0.01, 0.98, "(b)", transform=ax.transAxes, ha="left", va="top",
            fontweight="bold")
    ax.set_xlim(-3.25, 3.25)
    ax.set_ylim(-0.48, 1.43)
    ax.set_xlabel("Quantum-receiver frequency")
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)

    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "fig_signal_frequency_layout.pdf", bbox_inches="tight")
    fig.savefig(figures / "fig_signal_frequency_layout.png", dpi=300,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
