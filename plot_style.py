#!/usr/bin/env python3
"""Shared publication style for the manuscript's numerical figures."""
from __future__ import annotations

import matplotlib.pyplot as plt


# Both two-panel plots are inserted at nearly the full IEEE two-column width.
# Keeping the same source dimensions prevents LaTeX from applying materially
# different scale factors to nominally identical font sizes.
TWO_COLUMN_FIGSIZE = (7.15, 2.80)

COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
}

PANEL_BOX = {
    "facecolor": "white",
    "edgecolor": "none",
    "alpha": 0.92,
    "pad": 1.0,
}


def apply_publication_style() -> None:
    """Apply one vector-safe, Times-like style to every numerical plot."""
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
