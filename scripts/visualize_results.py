"""Plots for an experiments.csv emitted by run_experiments.py.

Each plot is a self-contained function registered with @register_plot("name").
Run all, run a subset (--plots), or skip some (--exclude). Output defaults to
a sibling `analysis_graphs/` folder next to the CSV.

    python scripts/visualize_results.py --list
    python scripts/visualize_results.py
    python scripts/visualize_results.py --plots headline_macro_f1 metric_heatmap
    python scripts/visualize_results.py --exclude val_vs_test_gap
    python scripts/visualize_results.py --split val
    python scripts/visualize_results.py --csv results/experiments.csv

Adding a new plot: define a function that takes (df, out_dir, split) and
decorate it with @register_plot("your_name"). It will appear in --list
automatically and run as part of the default set.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results/experiments.csv"

# Feature-set order matches ALL_FEATURE_SETS in run_experiments.py: the three
# orthogonal single-modality arms first, then pairwise combos, then the union.
FEATURE_SET_ORDER = [
    "sequence", "structure", "evolution",
    "seq_struct", "seq_evo", "struct_evo",
    "all",
]
MODEL_ORDER = ["random_forest", "mlp", "xgboost"]
SPLITS = ["val", "test"]

CLASS_PALETTE = {"benign": "#4C9AFF", "oncogenic": "#E5573F"}
MODEL_PALETTE = {"random_forest": "#5E81AC", "mlp": "#A3BE8C", "xgboost": "#BF616A"}
FEATURE_SET_PALETTE = {
    "sequence":   "#88C0D0",
    "structure":  "#D08770",
    "evolution":  "#A3BE8C",
    "seq_struct": "#B48EAD",
    "seq_evo":    "#81A1C1",
    "struct_evo": "#BF616A",
    "all":        "#2E3440",
}
MODEL_MARKERS = {"random_forest": "o", "mlp": "s", "xgboost": "^"}

PlotFn = Callable[[pd.DataFrame, Path, str], None]
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
    print(f"[results] wrote {path}")


def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    sub = df[df["split"] == split]
    if sub.empty:
        raise ValueError(f"no rows for split={split!r} (available: {sorted(df['split'].unique())})")
    return sub


def _annotate_bars(ax, fmt: str = "%.3f") -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, fontsize=8, padding=2)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

@register_plot("headline_macro_f1")
def headline_macro_f1(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    sub = _filter_split(df, split)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=sub, x="feature_set", y="macro_f1", hue="model_name",
        order=FEATURE_SET_ORDER, hue_order=MODEL_ORDER,
        palette=MODEL_PALETTE, ax=ax,
    )
    _annotate_bars(ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Macro-F1 by feature set and model ({split})")
    ax.legend(title="model", loc="lower right")
    _save(fig, out_dir, "headline_macro_f1")


@register_plot("metrics_grid")
def metrics_grid(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    sub = _filter_split(df, split)
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc"]
    metrics = [m for m in metrics if m in sub.columns]
    ncols = 3
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, metric in zip(axes, metrics):
        sns.barplot(
            data=sub, x="feature_set", y=metric, hue="model_name",
            order=FEATURE_SET_ORDER, hue_order=MODEL_ORDER,
            palette=MODEL_PALETTE, ax=ax, legend=False,
        )
        ax.set_ylim(0, 1)
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_PALETTE[m]) for m in MODEL_ORDER]
    fig.legend(handles, MODEL_ORDER, title="model", loc="lower center",
               ncol=len(MODEL_ORDER), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Metrics by feature set and model ({split})", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "metrics_grid")


@register_plot("metric_heatmap")
def metric_heatmap(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    sub = _filter_split(df, split)
    pivot = (
        sub.pivot(index="feature_set", columns="model_name", values="macro_f1")
        .reindex(index=FEATURE_SET_ORDER, columns=MODEL_ORDER)
    )
    fig, ax = plt.subplots(figsize=(5.5, 4))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="viridis",
        vmin=pivot.values.min() - 0.02, vmax=pivot.values.max() + 0.02,
        cbar_kws={"label": "macro_f1"}, ax=ax,
    )
    ax.set_title(f"Macro-F1 heatmap ({split})")
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature set")
    _save(fig, out_dir, "metric_heatmap")


@register_plot("per_class_f1")
def per_class_f1(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    sub = _filter_split(df, split).copy()
    if "f1_benign" not in sub.columns or "f1_oncogenic" not in sub.columns:
        print("[results] per_class_f1: per-class F1 columns missing, skipping")
        return
    long = sub.melt(
        id_vars=["feature_set", "model_name"],
        value_vars=["f1_benign", "f1_oncogenic"],
        var_name="class", value_name="f1",
    )
    long["class"] = long["class"].str.replace("f1_", "", regex=False)
    long["combo"] = long["feature_set"] + "\n" + long["model_name"]
    combo_order = [f"{fs}\n{m}" for fs in FEATURE_SET_ORDER for m in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=long, x="combo", y="f1", hue="class",
        order=combo_order, hue_order=["benign", "oncogenic"],
        palette=CLASS_PALETTE, ax=ax,
    )
    _annotate_bars(ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("F1")
    ax.set_title(f"Per-class F1 by (feature set, model) ({split})")
    ax.legend(title="class", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)
    _save(fig, out_dir, "per_class_f1")


@register_plot("precision_recall_oncogenic")
def precision_recall_oncogenic(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    sub = _filter_split(df, split)
    needed = {"precision_oncogenic", "recall_oncogenic"}
    if not needed.issubset(sub.columns):
        print("[results] precision_recall_oncogenic: missing columns, skipping")
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for fs in FEATURE_SET_ORDER:
        for m in MODEL_ORDER:
            row = sub[(sub["feature_set"] == fs) & (sub["model_name"] == m)]
            if row.empty:
                continue
            ax.scatter(
                row["recall_oncogenic"], row["precision_oncogenic"],
                color=FEATURE_SET_PALETTE[fs], marker=MODEL_MARKERS[m],
                s=140, edgecolor="black", linewidth=0.6,
                label=f"{fs} / {m}",
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall (oncogenic)")
    ax.set_ylabel("Precision (oncogenic)")
    ax.set_title(f"Precision vs recall on the oncogenic class ({split})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    _save(fig, out_dir, "precision_recall_oncogenic")


@register_plot("val_vs_test_gap")
def val_vs_test_gap(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    del split  # this plot uses both splits
    sub = df[df["split"].isin(SPLITS)].copy()
    sub["combo"] = sub["feature_set"] + "\n" + sub["model_name"]
    combo_order = [f"{fs}\n{m}" for fs in FEATURE_SET_ORDER for m in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(
        data=sub, x="combo", y="macro_f1", hue="split",
        order=combo_order, hue_order=SPLITS,
        palette={"val": "#A3BE8C", "test": "#5E81AC"}, ax=ax,
    )
    _annotate_bars(ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Macro F1")
    ax.set_title("Val vs test macro-F1 (overfitting check)")
    ax.legend(title="split")
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9)
    _save(fig, out_dir, "val_vs_test_gap")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help=f"Experiments CSV (default: {DEFAULT_CSV})")
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (default: <csv parent>/analysis_graphs)")
    p.add_argument("--split", choices=SPLITS, default="test",
                   help="Split to plot in single-split charts (default: test)")
    p.add_argument("--plots", nargs="*", default=None,
                   help="Plot names to run (default: all). Use --list to see names.")
    p.add_argument("--exclude", nargs="*", default=[],
                   help="Plot names to skip.")
    p.add_argument("--list", action="store_true",
                   help="Print registered plot names and exit.")
    args = p.parse_args()

    if args.list:
        for name in PLOT_REGISTRY:
            print(name)
        return

    out_dir = args.out or (args.csv.parent / "analysis_graphs")
    print(f"[results] loading {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"[results]   {len(df):,} rows, splits={sorted(df['split'].unique())}")

    excluded = set(args.exclude)
    selected = args.plots if args.plots is not None else list(PLOT_REGISTRY)
    selected = [n for n in selected if n not in excluded]

    for name in selected:
        fn = PLOT_REGISTRY.get(name)
        if fn is None:
            print(f"[results] WARN: unknown plot {name!r}, skipping")
            continue
        fn(df, out_dir, args.split)


if __name__ == "__main__":
    main()
