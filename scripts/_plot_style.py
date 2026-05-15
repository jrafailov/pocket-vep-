"""Shared figure styling for the pocket-vep report.

Modeled directly on the MUSiCaL paper's plot_*.py scripts (theme_pubr-style
clean publication look). One source of truth so visualize_results.py and
visualize_features.py can't drift apart visually.

Applying:
    from _plot_style import apply_style, ACCENT, NEUTRAL, TIER_PALETTE
    apply_style()
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Palette. ACCENT is the "this is the result" blue; the rest are graded
# toward neutral so the eye lands on accent without being told to.
ACCENT     = "#2E86AB"   # union / multi-modal winner
ACCENT_MID = "#7FB3D5"   # pairwise combos
NEUTRAL    = "#9A9A9A"   # single-modality baselines
CHANCE     = "#888888"

# Per-tier coloring for the headline plot. The seven feature configurations
# split cleanly into three tiers (single-modality, pairwise, union), and the
# colors reinforce that without needing a legend.
TIER_PALETTE = {
    "sequence":   NEUTRAL,
    "structure":  NEUTRAL,
    "evolution":  NEUTRAL,
    "seq_struct": ACCENT_MID,
    "seq_evo":    ACCENT_MID,
    "struct_evo": ACCENT_MID,
    "all":        ACCENT,
}

# Class colors used in EDA plots. Kept blue/orange so the report's two
# color uses (modality tier + class label) stay distinct.
CLASS_PALETTE = {"benign": "#4C9AFF", "oncogenic": "#E5573F"}


def pick_sans_font() -> str:
    """Match the MUSiCaL recipe. Helvetica if it exists on the system,
    otherwise walk the same fallback chain so the figures look consistent
    across machines."""
    available = {f.name for f in fm.fontManager.ttflist}
    for candidate in ("Helvetica", "Helvetica Neue", "Nimbus Sans",
                      "Liberation Sans", "Arial", "DejaVu Sans"):
        if candidate in available:
            return candidate
    return "sans-serif"


def apply_style() -> str:
    """Set matplotlib rcParams to the pocket-vep / MUSiCaL house style.
    Returns the font name actually chosen, so callers can log it."""
    font = pick_sans_font()
    # Sizes are tuned for a figure rendered at ~0.9 \linewidth in NeurIPS-style
    # two-column-ish layout, where LaTeX further downscales by ~0.6x. Anything
    # smaller than these at the source ends up squinted-at in the rendered PDF.
    plt.rcParams.update({
        "font.family": font,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "regular",
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#999",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.color": "#222",
        "ytick.color": "#222",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.fontsize": 10,
        # Embed TrueType so reviewers can copy text out of the PDF.
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.dpi": 130,
    })
    return font


def style_panel(ax) -> None:
    """Apply per-panel finishing touches: hide top/right spines, fade the
    remaining spines, light y-gridlines only. Matches MUSiCaL's _draw_panel
    body."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#999")
        ax.spines[side].set_linewidth(0.8)
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="#e6e6e6")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8)


def annotate_bars(ax, x_positions, heights, fmt: str = "{:.2f}",
                  offset: float = 0.018, fontsize: float = 9) -> None:
    """Bold value labels above each bar; MUSiCaL convention."""
    for xpos, h in zip(x_positions, heights):
        if h is None:
            continue
        ax.text(xpos, h + offset, fmt.format(h),
                ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold", color="#000")


def draw_chance_line(ax, y: float = 0.5, label: str | None = "chance",
                     label_x: float = 0.012) -> None:
    """Dashed gray reference line for the binary-classifier chance level.
    The label is drawn just inside the left edge of the axes so it reads
    once, not repeated across faceted panels (caller decides which panel
    gets the label)."""
    ax.axhline(y, color=CHANCE, linestyle="--", linewidth=0.7,
               alpha=0.7, zorder=0)
    if label is not None:
        ax.text(label_x, y + 0.005, label,
                transform=ax.get_yaxis_transform(),
                ha="left", va="bottom",
                fontsize=6.5, color="#666", style="italic")
