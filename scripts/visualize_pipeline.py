"""Pipeline diagram for the report.

Left-to-right schematic: ClinVar -> three feature lanes -> feature matrix
-> gene-grouped split -> classifier. Rectangles are FancyBboxPatch with
rounded corners; stages are joined by thin neutral connector lines (no
arrowheads — the left-to-right reading order carries the direction). Fonts
and palette come from _plot_style so this stays visually consistent with
the headline plots.

Logos: drop PNGs into figures/logos/<key>.png (keys: clinvar, sequence,
structure, evolution, matrix, split, classifier) and they get composited
just above the matching box. Missing logos silently no-op so the figure
builds before they land.

    python scripts/visualize_pipeline.py
    python scripts/visualize_pipeline.py --format png pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import FancyBboxPatch

from _plot_style import apply_style

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "report/figures"
LOGO_DIR = OUT_DIR / "logos"

# Two-tone fill: data + featurization stages are off-white, the post-join
# stages slide one shade toward accent blue so the eye picks up the
# train-and-evaluate end of the pipeline without needing arrows or
# colored borders.
BOX_EDGE      = "#5a5a5a"
CONNECTOR     = "#9a9a9a"
DATA_FILL     = "#ffffff"
LANE_FILL     = "#ffffff"
STAGE_FILL    = "#e3edf5"


def _box(ax, x, y, w, h, *, fill, edge=BOX_EDGE, lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.9",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2,
    ))


def _label(ax, cx, cy, *, title, body, footer, show_title=True,
           title_y=3.8, body_y=0.0, footer_y=-4.0):
    """Three-tier text block centered at (cx, cy). Vertical offsets are
    overridable so stages that need to fit both a logo AND a title (e.g.
    ClinVar) can lower the title and footer to make room. `show_title=
    False` suppresses the title when a logo is taking that role instead.
    Font sizes are intentionally on the larger side so the footer counts
    stay legible after the figure scales down to column width."""
    if show_title:
        ax.text(cx, cy + title_y, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color="#111", zorder=3)
    ax.text(cx, cy + body_y, body, ha="center", va="center",
            fontsize=10.5, color="#222", zorder=3, linespacing=1.3)
    ax.text(cx, cy + footer_y, footer, ha="center", va="center",
            fontsize=10, color="#111", zorder=3)


def _connector(ax, x0, y0, x1, y1, *, color=CONNECTOR, lw=0.9):
    ax.add_line(Line2D([x0, x1], [y0, y1],
                       color=color, linewidth=lw, zorder=1,
                       solid_capstyle="round"))


def _logo_path(key) -> Path | None:
    """Return the logo path for `key` if it exists on disk, else None.
    Caller uses the truthiness to decide whether the corresponding box's
    text title should be suppressed."""
    p = LOGO_DIR / f"{key}.png"
    return p if p.exists() else None


# Per-key zoom overrides. Source images vary by 10x in native pixel size
# (UCSC banner is 145x55, the DeepMind wordmark is 3810x512), so we pick
# zoom to match a roughly constant display height of ~2.5 data units.
# Formula: zoom = target_inches * fig.dpi / source_height_pixels, with
# target ~0.245 in at the current figsize and fig.dpi=130 from apply_style.
LOGO_ZOOM = {
    "clinvar":    0.07,   # 512x512 circular icon
    "sequence":   0.32,   # placeholder, no logo yet
    "structure":  0.035,  # 3810x512 wordmark (very wide)
    "evolution":  0.35,   # 145x55 banner (busier than the other marks, kept small)
    "matrix":     0.32,   # placeholder, no logo yet
    "split":      0.32,   # placeholder, no logo yet
    "classifier": 0.10,   # 512x176 logo
}


def _logo(ax, key, cx, cy):
    """Place the logo for `key` centered at (cx, cy). Silently no-ops if
    the PNG isn't on disk yet so the figure builds either way."""
    path = _logo_path(key)
    if path is None:
        return
    img = mpimg.imread(path)
    ax.add_artist(AnnotationBbox(
        OffsetImage(img, zoom=LOGO_ZOOM.get(key, 0.30)),
        (cx, cy),
        frameon=False, zorder=4, pad=0.0,
    ))


def _stage(ax, key, x, y, w, h, *, fill, title, body, footer,
           force_title_with_logo=False):
    """One pipeline stage. If figures/logos/<key>.png exists it replaces
    the text title at the top of the box; otherwise the title renders as
    bold text. `force_title_with_logo=True` keeps the title even when a
    logo is present, stacking logo above title (taller boxes need this)."""
    _box(ax, x, y, w, h, fill=fill)
    cx, cy = x + w / 2, y + h / 2
    has_logo = _logo_path(key) is not None

    if has_logo and force_title_with_logo and h >= 20:
        # Tall-box layout (used by ClinVar at h=40). Logo near the top,
        # title well below it so the two don't visually collide, body and
        # footer spaced to fill the rest of the box rather than crowding
        # mid-height.
        _label(ax, cx, cy,
               title=title, body=body, footer=footer,
               show_title=True,
               title_y=5.0, body_y=-2.0, footer_y=-9.0)
        _logo(ax, key, cx, cy + 13.0)
    elif has_logo and force_title_with_logo:
        # Compact lane layout (h=12). Title at the top of the box, small
        # logo just below it, body and footer in the lower half. Used by
        # the structure and evolution lanes so the headline reads even
        # though the source-provider logo is also present. Fonts are a
        # touch smaller than the default _label tier so all four text
        # rows + logo fit without the body crowding the footer.
        ax.text(cx, cy + 4.6, title, ha="center", va="center",
                fontsize=11.5, fontweight="bold", color="#111", zorder=3)
        _logo(ax, key, cx, cy + 1.9)
        ax.text(cx, cy - 1.0, body, ha="center", va="center",
                fontsize=9.5, color="#222", zorder=3, linespacing=1.25)
        ax.text(cx, cy - 4.6, footer, ha="center", va="center",
                fontsize=9.5, color="#111", zorder=3)
    elif has_logo:
        _label(ax, cx, cy,
               title=title, body=body, footer=footer,
               show_title=False)
        _logo(ax, key, cx, cy + 3.6)
    else:
        _label(ax, cx, cy,
               title=title, body=body, footer=footer)


def build_pipeline(out_dir: Path, formats: Iterable[str]) -> None:
    apply_style()

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_xlim(0, 138)
    ax.set_ylim(0, 42)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    mid_cy = 22  # all "trunk" stages share this y-center
    box_h  = 12
    box_y  = mid_cy - box_h / 2

    # --- Stage 1: ClinVar ---
    # Sized to match the full vertical span of the three-lane stack
    # (lanes at y=8, 22, 36 with box_h=12 give a span of ~40), so the
    # input "anchors" the pipeline visually rather than feeling shorter
    # than what flows out of it. Width 22 keeps the two-line body
    # ("cancer-gene missense" / "GRCh38 SNVs, dedup") inside the border.
    clinvar_x, clinvar_w = 3, 22
    clinvar_h = 40
    clinvar_y = mid_cy - clinvar_h / 2
    _stage(ax, "clinvar", clinvar_x, clinvar_y, clinvar_w, clinvar_h,
           fill=DATA_FILL,
           title="ClinVar",
           body="cancer-gene missense\nGRCh38 SNVs, dedup",
           footer="36,075 variants",
           force_title_with_logo=True)

    # --- Stage 2: three feature lanes ---
    lane_x, lane_w = 32, 30
    lanes = [
        ("sequence",  36, "Sequence",
         "Δmass, Δhydro, Δcharge\nBLOSUM62, WT/MT identity",
         "44 features"),
        ("structure", 22, "Structure",
         "AlphaFold via UniProt\nfpocket, DSSP, pLDDT",
         "8 features"),
        ("evolution",  8, "Evolution",
         "UCSC phyloP100way\nphastCons100way bigWigs",
         "2 features"),
    ]
    for key, ly, title, body, footer in lanes:
        # Structure and evolution lanes carry a provider logo (DeepMind,
        # UCSC); force the headline back on so the panel still reads as
        # "Structure" / "Evolution" at a glance. Sequence has no logo, so
        # its title renders through the default no-logo branch already.
        _stage(ax, key, lane_x, ly - box_h / 2, lane_w, box_h,
               fill=LANE_FILL, title=title, body=body, footer=footer,
               force_title_with_logo=True)
        _connector(ax, clinvar_x + clinvar_w, mid_cy, lane_x, ly)

    # --- Stage 3: feature matrix ---
    # Trunk row positions shifted right so the lanes-to-matrix gap matches
    # the clinvar-to-lanes gap (~7 units on each side), keeping the
    # figure visually balanced left-to-right.
    matrix_x, matrix_w = 69, 20
    _stage(ax, "matrix", matrix_x, box_y, matrix_w, box_h, fill=STAGE_FILL,
           title="Feature matrix",
           body="30,620 × 54\nrows × features",
           footer="1,040 genes")
    for _, ly, *_ in lanes:
        _connector(ax, lane_x + lane_w, ly, matrix_x, mid_cy)

    # --- Stage 4: gene-grouped split ---
    # Widened to 22 so the two-line body ("70% train" / "15% validate · 15% test")
    # fits without poking past the right border. Previous 17 still clipped
    # the validation/test row.
    split_x, split_w = 93, 22
    _stage(ax, "split", split_x, box_y, split_w, box_h, fill=STAGE_FILL,
           title="Gene-grouped split",
           body="70% train\n15% validate, 15% test",
           footer="5 seeds")
    _connector(ax, matrix_x + matrix_w, mid_cy, split_x, mid_cy)

    # --- Stage 5: classifier ---
    clf_x, clf_w = 119, 16
    _stage(ax, "classifier", clf_x, box_y, clf_w, box_h, fill=STAGE_FILL,
           title="ML classifier",
           body="Random Forest\nMLP, XGBoost",
           footer="F1, PR, ROC, ECE")
    _connector(ax, split_x + split_w, mid_cy, clf_x, mid_cy)

    fig.tight_layout(pad=0.2)
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = out_dir / f"pipeline.{fmt}"
        fig.savefig(path, format=fmt)
        print(f"wrote {path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=OUT_DIR)
    p.add_argument("--format", nargs="+", default=["png", "pdf"])
    args = p.parse_args()
    build_pipeline(args.out, args.format)


if __name__ == "__main__":
    main()
