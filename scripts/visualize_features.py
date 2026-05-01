"""EDA plots for data/processed/feature_matrix.parquet.

Each plot is a self-contained function registered with @register_plot("name").
Run all, run a subset (--plots), or skip some (--exclude).

    python scripts/visualize_features.py --list
    python scripts/visualize_features.py
    python scripts/visualize_features.py --plots class_balance pca_2d_by_class
    python scripts/visualize_features.py --exclude wt_to_mt_substitution_heatmap
    python scripts/visualize_features.py --sample 20000

Adding a new plot: define a function that takes (df, out_dir) and decorate it
with @register_plot("your_name"). It will appear in --list automatically and
run as part of the default set.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "data/processed/feature_matrix.parquet"
DEFAULT_OUT = ROOT / "results/eda"
LABEL_COL = "ML_Label"
CLASS_ORDER = ["benign", "oncogenic"]
CLASS_PALETTE = {"benign": "#4C9AFF", "oncogenic": "#E5573F"}

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
WT_AA_COLS = [f"WT_AA_{a}" for a in AA_ORDER]
MT_AA_COLS = [f"MT_AA_{a}" for a in AA_ORDER]
SS_COLS = ["ss_H", "ss_E", "ss_L"]

# Numeric features expected in the matrix. Plots tolerate missing columns
# (e.g. structure columns absent if the matrix was built sequence-only).
NUMERIC_FEATURES = [
    "Delta_Mass",
    "Delta_Hydro",
    "Delta_Charge",
    "BLOSUM62",
    "plddt",
    "sasa",
    "dist_to_nearest_pocket",
    "druggability",
]

PlotFn = Callable[[pd.DataFrame, Path], None]
PLOT_REGISTRY: Dict[str, PlotFn] = {}


def register_plot(name: str):
    def deco(fn: PlotFn) -> PlotFn:
        if name in PLOT_REGISTRY:
            raise ValueError(f"duplicate plot name: {name}")
        PLOT_REGISTRY[name] = fn
        return fn
    return deco


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] wrote {path}")


def _present(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

@register_plot("class_balance")
def class_balance(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(
        data=df, x=LABEL_COL, order=CLASS_ORDER,
        hue=LABEL_COL, palette=CLASS_PALETTE, legend=False, ax=ax,
    )
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    ax.set_title("Class balance")
    ax.set_xlabel("")
    _save(fig, out_dir, "class_balance")


@register_plot("numeric_distributions")
def numeric_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    feats = _present(df, NUMERIC_FEATURES)
    if not feats:
        print("[eda] numeric_distributions: no numeric features present, skipping")
        return
    n = len(feats)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, feat in zip(axes, feats):
        sns.kdeplot(
            data=df, x=feat, hue=LABEL_COL, hue_order=CLASS_ORDER,
            palette=CLASS_PALETTE, common_norm=False, fill=True, alpha=0.3,
            ax=ax,
        )
        ax.set_title(feat)
        ax.set_xlabel("")
    for ax in axes[len(feats):]:
        ax.set_visible(False)
    fig.suptitle("Numeric feature distributions by class", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "numeric_distributions")


@register_plot("correlation_heatmap")
def correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    feats = _present(df, NUMERIC_FEATURES)
    if len(feats) < 2:
        print("[eda] correlation_heatmap: need >=2 numeric features, skipping")
        return
    corr = df[feats].corr(method="pearson")
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(feats), 1 + 0.6 * len(feats)))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="vlag", center=0,
        vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title("Pearson correlation, numeric features")
    _save(fig, out_dir, "correlation_heatmap")


def _scatter_by_class(df: pd.DataFrame, x: str, y: str, out_dir: Path, name: str) -> None:
    if x not in df.columns or y not in df.columns:
        print(f"[eda] {name}: missing column, skipping")
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=df, x=x, y=y, hue=LABEL_COL, hue_order=CLASS_ORDER,
        palette=CLASS_PALETTE, alpha=0.3, s=10, edgecolor="none", ax=ax,
    )
    ax.set_title(f"{x} vs {y}, colored by class")
    _save(fig, out_dir, name)


@register_plot("delta_hydro_vs_charge")
def delta_hydro_vs_charge(df: pd.DataFrame, out_dir: Path) -> None:
    _scatter_by_class(df, "Delta_Hydro", "Delta_Charge", out_dir, "delta_hydro_vs_charge")


@register_plot("blosum_vs_delta_mass")
def blosum_vs_delta_mass(df: pd.DataFrame, out_dir: Path) -> None:
    _scatter_by_class(df, "BLOSUM62", "Delta_Mass", out_dir, "blosum_vs_delta_mass")


@register_plot("plddt_vs_sasa")
def plddt_vs_sasa(df: pd.DataFrame, out_dir: Path) -> None:
    _scatter_by_class(df, "plddt", "sasa", out_dir, "plddt_vs_sasa")


@register_plot("pocket_membership_by_class")
def pocket_membership_by_class(df: pd.DataFrame, out_dir: Path) -> None:
    if "in_pocket" not in df.columns:
        print("[eda] pocket_membership_by_class: in_pocket missing, skipping")
        return
    rates = df.groupby(LABEL_COL)["in_pocket"].mean().reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4))
    rates.plot.bar(
        color=[CLASS_PALETTE[c] for c in rates.index],
        ax=ax, edgecolor="black",
    )
    ax.set_ylabel("P(in_pocket = 1)")
    ax.set_xlabel("")
    ax.set_title("Pocket-membership rate by class")
    ax.set_ylim(0, max(0.05, rates.max() * 1.2))
    for i, v in enumerate(rates.values):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.setp(ax.get_xticklabels(), rotation=0)
    _save(fig, out_dir, "pocket_membership_by_class")


@register_plot("secondary_structure_by_class")
def secondary_structure_by_class(df: pd.DataFrame, out_dir: Path) -> None:
    cols = _present(df, SS_COLS)
    if not cols:
        print("[eda] secondary_structure_by_class: ss_* missing, skipping")
        return
    rates = df.groupby(LABEL_COL)[cols].mean().reindex(CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4))
    rates.T.plot.bar(
        ax=ax, color=[CLASS_PALETTE[c] for c in rates.index],
        edgecolor="black",
    )
    ax.set_ylabel("Fraction of residues")
    ax.set_xlabel("Secondary structure")
    ax.set_title("Secondary-structure usage by class")
    ax.legend(title="class")
    plt.setp(ax.get_xticklabels(), rotation=0)
    _save(fig, out_dir, "secondary_structure_by_class")


def _aa_frequency_by_class(
    df: pd.DataFrame, cols: list[str], side: str, out_dir: Path, name: str,
) -> None:
    cols = _present(df, cols)
    if not cols:
        print(f"[eda] {name}: {side} one-hot columns missing, skipping")
        return
    freqs = df.groupby(LABEL_COL)[cols].mean().reindex(CLASS_ORDER)
    freqs.columns = [c.split("_")[-1] for c in freqs.columns]
    fig, ax = plt.subplots(figsize=(10, 4))
    freqs.T.plot.bar(
        ax=ax, color=[CLASS_PALETTE[c] for c in freqs.index],
        edgecolor="black", width=0.8,
    )
    ax.set_ylabel("Frequency")
    ax.set_xlabel(f"{side} amino acid")
    ax.set_title(f"{side} amino-acid frequency by class")
    ax.legend(title="class")
    plt.setp(ax.get_xticklabels(), rotation=0)
    _save(fig, out_dir, name)


@register_plot("wt_aa_frequency_by_class")
def wt_aa_frequency_by_class(df: pd.DataFrame, out_dir: Path) -> None:
    _aa_frequency_by_class(df, WT_AA_COLS, "WT", out_dir, "wt_aa_frequency_by_class")


@register_plot("mt_aa_frequency_by_class")
def mt_aa_frequency_by_class(df: pd.DataFrame, out_dir: Path) -> None:
    _aa_frequency_by_class(df, MT_AA_COLS, "MT", out_dir, "mt_aa_frequency_by_class")


@register_plot("wt_to_mt_substitution_heatmap")
def wt_to_mt_substitution_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    wt_cols = _present(df, WT_AA_COLS)
    mt_cols = _present(df, MT_AA_COLS)
    if len(wt_cols) != 20 or len(mt_cols) != 20:
        print("[eda] wt_to_mt_substitution_heatmap: WT/MT one-hot incomplete, skipping")
        return
    wt_idx = df[wt_cols].values.argmax(axis=1)
    mt_idx = df[mt_cols].values.argmax(axis=1)
    classes = [c for c in CLASS_ORDER if c in df[LABEL_COL].unique()]
    fig, axes = plt.subplots(1, len(classes), figsize=(7 * len(classes), 6))
    axes = np.atleast_1d(axes)
    for ax, cls in zip(axes, classes):
        mask = (df[LABEL_COL].values == cls)
        mat = np.zeros((20, 20), dtype=int)
        np.add.at(mat, (wt_idx[mask], mt_idx[mask]), 1)
        sns.heatmap(
            mat, xticklabels=AA_ORDER, yticklabels=AA_ORDER,
            cmap="magma", ax=ax, cbar_kws={"shrink": 0.7},
        )
        ax.set_title(f"WT -> MT substitutions ({cls})")
        ax.set_xlabel("MT")
        ax.set_ylabel("WT")
    fig.tight_layout()
    _save(fig, out_dir, "wt_to_mt_substitution_heatmap")


@register_plot("pca_2d_by_class")
def pca_2d_by_class(df: pd.DataFrame, out_dir: Path) -> None:
    feats = _present(df, NUMERIC_FEATURES)
    if len(feats) < 2:
        print("[eda] pca_2d_by_class: need >=2 numeric features, skipping")
        return
    X = df[feats].to_numpy(dtype=float)
    keep = ~np.isnan(X).any(axis=1)
    if keep.sum() < 10:
        print("[eda] pca_2d_by_class: too few complete rows, skipping")
        return
    X = X[keep]
    y = df[LABEL_COL].to_numpy()[keep]
    Xs = StandardScaler().fit_transform(X)
    pcs = PCA(n_components=2, random_state=42).fit_transform(Xs)
    plot_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], LABEL_COL: y})
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(
        data=plot_df, x="PC1", y="PC2",
        hue=LABEL_COL, hue_order=CLASS_ORDER, palette=CLASS_PALETTE,
        alpha=0.3, s=10, edgecolor="none", ax=ax,
    )
    ax.set_title(f"PCA of {len(feats)} numeric features, colored by class")
    _save(fig, out_dir, "pca_2d_by_class")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX,
                   help=f"Feature matrix parquet (default: {DEFAULT_MATRIX})")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help=f"Output directory (default: {DEFAULT_OUT})")
    p.add_argument("--plots", nargs="*", default=None,
                   help="Plot names to run (default: all). Use --list to see names.")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Plot names to skip.")
    p.add_argument("--list", action="store_true",
                   help="Print registered plot names and exit.")
    p.add_argument("--sample", type=int, default=None,
                   help="Random subsample N rows for fast iteration.")
    args = p.parse_args()

    if args.list:
        for name in PLOT_REGISTRY:
            print(name)
        return

    print(f"[eda] loading {args.matrix}")
    df = pd.read_parquet(args.matrix)
    print(f"[eda]   {len(df):,} rows, {df.shape[1]} columns")
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"[eda]   subsampled to {len(df):,} rows")

    excluded = set(args.exclude)
    selected = args.plots if args.plots is not None else list(PLOT_REGISTRY)
    selected = [n for n in selected if n not in excluded]

    for name in selected:
        fn = PLOT_REGISTRY.get(name)
        if fn is None:
            print(f"[eda] WARN: unknown plot {name!r}, skipping")
            continue
        fn(df, args.out)


if __name__ == "__main__":
    main()
