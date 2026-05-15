"""Plots and a results table for an experiments.csv emitted by run_experiments.py.

Reads the per-seed CSV directly and aggregates mean/std in-script, so error
bars come from the seed sweep. Each plot is registered with @register_plot and
takes (df, out_dir, split). Default output goes to a sibling
`analysis_graphs/` folder next to the CSV.

    python scripts/visualize_results.py --list
    python scripts/visualize_results.py
    python scripts/visualize_results.py --plots headline_pr_auc
    python scripts/visualize_results.py --exclude val_vs_test_gap
    python scripts/visualize_results.py --split val
    python scripts/visualize_results.py --csv path/to/experiments.csv

Adding a new plot: define a function (df, out_dir, split) and decorate it
with @register_plot("your_name").
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _plot_style import (
    apply_style, style_panel, annotate_bars, draw_chance_line,
    ACCENT, ACCENT_MID, NEUTRAL, TIER_PALETTE,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results_new_v2/experiments.csv"

# Output formats, set from --format in main(). Module-global so plot functions
# don't thread it through every call site.
FORMATS: List[str] = ["png"]

apply_style()

# Tiered: single-modality, pairwise combos, then the union.
FEATURE_SET_ORDER = [
    "sequence", "structure", "evolution",
    "seq_struct", "seq_evo", "struct_evo",
    "all",
]
TIER_BREAKS = [2.5, 5.5]  # vertical separators in headline plots

MODEL_ORDER = ["random_forest", "mlp", "xgboost"]
SPLITS = ["val", "test"]

# Display names — edit these to relabel without touching plot code.
FEATURE_SET_LABEL = {
    "sequence":   "Sequence",
    "structure":  "Structure",
    "evolution":  "Evolution",
    "seq_struct": "Sequence + Structure",
    "seq_evo":    "Sequence + Evolution",
    "struct_evo": "Structure + Evolution",
    "all":        "All Combined",
}
# Short labels for figure x-axes. Long names rotate awkwardly and clip the
# legend at the report's figsize; short names fit horizontally. Tables keep
# the long names via FEATURE_SET_LABEL.
FEATURE_SET_SHORT = {
    "sequence":   "Seq",
    "structure":  "Str",
    "evolution":  "Evo",
    # Two-line labels for the combos so they fit horizontally under each bar
    # without colliding with neighbors. Singles stay one line on purpose;
    # the height mismatch is invisible because bars share an x-axis baseline.
    "seq_struct": "Seq+\nStr",
    "seq_evo":    "Seq+\nEvo",
    "struct_evo": "Str+\nEvo",
    "all":        "All",
}
MODEL_LABEL = {
    "random_forest": "Random Forest",
    "mlp":           "MLP",
    "xgboost":       "XGBoost",
}
METRIC_LABEL = {
    "accuracy":          "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "macro_f1":          "Macro F1",
    "weighted_f1":       "Weighted F1",
    "f1_benign":         "F1 (Benign)",
    "f1_oncogenic":      "F1 (Oncogenic)",
    "precision_benign":  "Precision (Benign)",
    "recall_benign":     "Recall (Benign)",
    "precision_oncogenic": "Precision (Oncogenic)",
    "recall_oncogenic":    "Recall (Oncogenic)",
    "roc_auc":           "ROC-AUC",
    "pr_auc":            "PR-AUC",
    "brier_score":       "Brier Score",
    "ece":               "ECE",
}
SPLIT_LABEL = {"val": "Validation", "test": "Test"}


def _fs(name: str) -> str:
    return FEATURE_SET_LABEL.get(name, name)


def _fs_short(name: str) -> str:
    return FEATURE_SET_SHORT.get(name, name)


def _ml(name: str) -> str:
    return MODEL_LABEL.get(name, name)


def _mt(name: str) -> str:
    return METRIC_LABEL.get(name, name.replace("_", " ").title())


def _sp(name: str) -> str:
    return SPLIT_LABEL.get(name, name.capitalize())

HEADLINE_METRICS = ["macro_f1", "pr_auc", "roc_auc", "balanced_accuracy"]
TABLE_METRICS = ["macro_f1", "pr_auc", "roc_auc", "balanced_accuracy", "ece"]
GRID_METRICS = [
    "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1",
    "roc_auc", "pr_auc", "brier_score", "ece",
]
LOWER_IS_BETTER = {"brier_score", "ece"}

# Composition for the lift plot: each multi-modal set vs its constituent singles.
COMPONENT_OF = {
    "seq_struct": ["sequence", "structure"],
    "seq_evo":    ["sequence", "evolution"],
    "struct_evo": ["structure", "evolution"],
    "all":        ["sequence", "structure", "evolution"],
}

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in FORMATS:
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path)
        print(f"[results] wrote {path}")
    plt.close(fig)


def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    sub = df[df["split"] == split]
    if sub.empty:
        raise ValueError(f"no rows for split={split!r} (available: {sorted(df['split'].unique())})")
    return sub


def _agg(df: pd.DataFrame, split: str, metrics: list[str]) -> pd.DataFrame:
    """Mean/std per (feature_set, model_name) for the requested metrics.
    Returns a DataFrame with MultiIndex columns (metric, 'mean'|'std')."""
    sub = _filter_split(df, split)
    g = sub.groupby(["feature_set", "model_name"])[metrics]
    agg = pd.concat({"mean": g.mean(), "std": g.std()}, axis=1)
    agg.columns = agg.columns.swaplevel(0, 1)
    return agg.sort_index(axis=1)


def _grouped_bar(
    ax, agg: pd.DataFrame, metric: str,
    x_order: list[str], hue_order: list[str], palette: dict,
    width_total: float = 0.8, capsize: float = 2.5,
    annotate: bool = False, annotate_fmt: str = "{:.2f}",
    annotate_fontsize: float = 6.5,
) -> None:
    """Grouped bar with yerr; we already have aggregated data so seaborn would
    re-aggregate and lose the precomputed stds. If annotate=True, writes the
    mean above each bar so the reader can read values directly without
    inferring them from bar height (the failure mode that gets papered over by
    a truncated y-axis)."""
    x = np.arange(len(x_order))
    n = len(hue_order)
    width = width_total / n
    for i, h in enumerate(hue_order):
        means, stds = [], []
        for fs in x_order:
            try:
                means.append(agg.loc[(fs, h), (metric, "mean")])
                stds.append(agg.loc[(fs, h), (metric, "std")])
            except KeyError:
                means.append(np.nan)
                stds.append(np.nan)
        offset = (i - (n - 1) / 2) * width
        positions = x + offset
        ax.bar(
            positions, means, width, yerr=stds, capsize=capsize,
            label=_ml(h) if h in MODEL_LABEL else h,
            color=palette[h], edgecolor="black", linewidth=0.4,
            error_kw={"elinewidth": 0.7, "ecolor": "#333"},
        )
        if annotate:
            for xpos, mean, std in zip(positions, means, stds):
                if np.isnan(mean):
                    continue
                top = mean + (std if not np.isnan(std) else 0.0)
                ax.text(
                    xpos, top + 0.005, annotate_fmt.format(mean),
                    ha="center", va="bottom",
                    fontsize=annotate_fontsize, color="#222",
                )
    ax.set_xticks(x)
    ax.set_xticklabels([_fs(fs) if fs in FEATURE_SET_LABEL else fs for fs in x_order])


# Per-metric y-axis floor. The point is to anchor to a meaningful reference
# (chance line for binary classifiers, zero for calibration error) rather than
# the data window, which is what produced the distorted figure in the first
# draft.
METRIC_FLOOR = {
    "macro_f1":          0.5,
    "weighted_f1":       0.5,
    "accuracy":          0.5,
    "balanced_accuracy": 0.5,
    "roc_auc":           0.5,
    "pr_auc":            0.0,  # baseline is class prevalence, plotted separately
    "f1_benign":         0.5,
    "f1_oncogenic":      0.5,
    "brier_score":       0.0,
    "ece":               0.0,
}


def _metric_ylim(metric: str, means: np.ndarray, stds: np.ndarray,
                 headroom: float = 0.04) -> tuple[float, float]:
    """Anchor the lower bound at a meaningful reference (chance for AUC-ish
    metrics, zero for calibration); only the upper bound is data-driven."""
    floor = METRIC_FLOOR.get(metric, 0.0)
    hi = float(np.nanmax(means + np.nan_to_num(stds))) + headroom
    return floor, min(1.0, max(hi, floor + 0.05))


# ---------------------------------------------------------------------------
# Headline plots: one per metric (auto-registered below)
# ---------------------------------------------------------------------------

def _draw_panel_bars(
    ax, feature_sets: list[str], means: list[float], stds: list[float],
    palette: dict, *, annotate: bool = True, annotate_fmt: str = "{:.2f}",
) -> None:
    """One MUSiCaL-style bar panel. Bars are tier-colored via `palette`,
    error bars are thin, and value labels sit above each bar in bold.
    """
    x = np.arange(len(feature_sets))
    colors = [palette[fs] for fs in feature_sets]
    ax.bar(
        x, means, width=0.7, color=colors, alpha=0.95,
        edgecolor="#333", linewidth=0.6,
        yerr=stds, capsize=2.0,
        error_kw={"elinewidth": 0.7, "ecolor": "#333"},
    )
    if annotate:
        # Position labels above the upper end of the error bar so the
        # numeric value never gets clipped by the cap.
        tops = [m + (s if not np.isnan(s) else 0.0) for m, s in zip(means, stds)]
        annotate_bars(ax, x, tops, fmt=annotate_fmt, offset=0.012)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_fs_short(fs) if fs in FEATURE_SET_SHORT else fs for fs in feature_sets],
        rotation=0, ha="center",
    )
    ax.set_xlim(-0.7, len(feature_sets) - 0.3)


def _draw_headline(df: pd.DataFrame, out_dir: Path, split: str, metric: str) -> None:
    """Facet by model: three panels side-by-side, one per classifier.
    Within each panel, seven bars (one per feature configuration) colored
    by tier so the union bar (accent blue) jumps out against the
    single-modality bars (neutral gray) and pairwise combos (mid-blue).
    Mirrors the MUSiCaL Random / Hard negatives layout.
    """
    agg = _agg(df, split, [metric])
    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(
        1, n_models, figsize=(2.5 * n_models + 1.0, 2.6), sharey=True,
    )
    axes = np.atleast_1d(axes)

    floor = METRIC_FLOOR.get(metric, 0.0)
    ymax = 1.0 if metric != "ece" else 0.15

    for i, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        means = [float(agg.loc[(fs, model), (metric, "mean")])
                 for fs in FEATURE_SET_ORDER]
        stds = [float(agg.loc[(fs, model), (metric, "std")])
                for fs in FEATURE_SET_ORDER]
        _draw_panel_bars(ax, FEATURE_SET_ORDER, means, stds, TIER_PALETTE)

        ax.set_title(_ml(model), pad=4)
        ax.set_ylim(0.0 if floor < 0.5 else 0.0, ymax)
        style_panel(ax)
        if i == 0:
            ax.set_ylabel(f"Test {_mt(metric)}")
        else:
            ax.tick_params(axis="y", labelleft=False)

    # Tier legend lives outside the panels; three swatches, no frame.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=NEUTRAL),
        plt.Rectangle((0, 0), 1, 1, color=ACCENT_MID),
        plt.Rectangle((0, 0), 1, 1, color=ACCENT),
    ]
    fig.legend(
        handles, ["Single modality", "Pairwise combo", "Union (all)"],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06),
        fontsize=8, handlelength=1.0, columnspacing=1.4, frameon=False,
    )

    fig.tight_layout()
    _save(fig, out_dir, f"headline_{metric}")


def _make_headline_fn(metric: str) -> PlotFn:
    def fn(df: pd.DataFrame, out_dir: Path, split: str) -> None:
        _draw_headline(df, out_dir, split, metric)
    fn.__name__ = f"headline_{metric}"
    return fn


for _m in HEADLINE_METRICS:
    register_plot(f"headline_{_m}")(_make_headline_fn(_m))


@register_plot("headline_macro_f1_perconfig")
def headline_macro_f1_perconfig(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    """Same shape as headline_macro_f1 but bars are colored per feature
    configuration (FEATURE_SET_PALETTE) instead of by tier. Used to
    color-match the ROC curves underneath in the report's headline
    figure, so the eye can track a single config across both rows.
    """
    metric = "macro_f1"
    agg = _agg(df, split, [metric])
    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(
        1, n_models, figsize=(2.5 * n_models + 1.0, 2.6), sharey=True,
    )
    axes = np.atleast_1d(axes)

    floor = METRIC_FLOOR.get(metric, 0.0)
    ymax = 1.0

    for i, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        means = [float(agg.loc[(fs, model), (metric, "mean")])
                 for fs in FEATURE_SET_ORDER]
        stds = [float(agg.loc[(fs, model), (metric, "std")])
                for fs in FEATURE_SET_ORDER]
        _draw_panel_bars(ax, FEATURE_SET_ORDER, means, stds, FEATURE_SET_PALETTE)
        ax.set_title(_ml(model), pad=4)
        ax.set_ylim(0.0 if floor < 0.5 else 0.0, ymax)
        style_panel(ax)
        if i == 0:
            ax.set_ylabel(f"Test {_mt(metric)}")
        else:
            ax.tick_params(axis="y", labelleft=False)

    fig.tight_layout()
    _save(fig, out_dir, "headline_macro_f1_perconfig")


@register_plot("headline_f1_auc")
def headline_f1_auc(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    """Two-row variant of the headline figure for the report.

    Top row: macro-F1 across the 7 feature configurations per model.
    Bottom row: ROC-AUC over the same configurations / models.
    Same tier coloring, same bar order, shared y-axis within a row.
    """
    metrics = ["macro_f1", "roc_auc"]
    agg = _agg(df, split, metrics)
    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(
        2, n_models, figsize=(2.5 * n_models + 1.0, 4.8),
        sharey="row",
    )

    for row, metric in enumerate(metrics):
        floor = METRIC_FLOOR.get(metric, 0.0)
        ymax = 1.0
        for i, model in enumerate(MODEL_ORDER):
            ax = axes[row, i]
            means = [float(agg.loc[(fs, model), (metric, "mean")])
                     for fs in FEATURE_SET_ORDER]
            stds = [float(agg.loc[(fs, model), (metric, "std")])
                    for fs in FEATURE_SET_ORDER]
            _draw_panel_bars(ax, FEATURE_SET_ORDER, means, stds, TIER_PALETTE)
            ax.set_ylim(0.0 if floor < 0.5 else 0.0, ymax)
            style_panel(ax)
            if row == 0:
                ax.set_title(_ml(model), pad=4)
            if i == 0:
                ax.set_ylabel(f"Test {_mt(metric)}")
            else:
                ax.tick_params(axis="y", labelleft=False)
            # Only show x-tick labels on the bottom row.
            if row == 0:
                ax.tick_params(axis="x", labelbottom=False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=NEUTRAL),
        plt.Rectangle((0, 0), 1, 1, color=ACCENT_MID),
        plt.Rectangle((0, 0), 1, 1, color=ACCENT),
    ]
    fig.legend(
        handles, ["Single modality", "Pairwise combo", "Union (all)"],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03),
        fontsize=8, handlelength=1.0, columnspacing=1.4, frameon=False,
    )

    fig.tight_layout()
    _save(fig, out_dir, "headline_f1_auc")


# ---------------------------------------------------------------------------
# Other plots
# ---------------------------------------------------------------------------

@register_plot("metrics_grid")
def metrics_grid(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    metrics = [m for m in GRID_METRICS if m in df.columns]
    agg = _agg(df, split, metrics)
    ncols = 4
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, metric in zip(axes, metrics):
        _grouped_bar(ax, agg, metric, FEATURE_SET_ORDER, MODEL_ORDER, MODEL_PALETTE, capsize=2)
        suffix = " (lower = better)" if metric in LOWER_IS_BETTER else ""
        ax.set_title(_mt(metric) + suffix, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=7, ha="right")
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    for ax in axes[len(metrics):]:
        ax.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_PALETTE[m]) for m in MODEL_ORDER]
    fig.legend(handles, [_ml(m) for m in MODEL_ORDER], title="Model",
               loc="lower center", ncol=len(MODEL_ORDER), bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Metrics by feature set and model ({_sp(split)})", y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "metrics_grid")


@register_plot("metric_heatmap")
def metric_heatmap(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    agg = _agg(df, split, ["macro_f1"])
    mean = agg[("macro_f1", "mean")].unstack("model_name").reindex(FEATURE_SET_ORDER, columns=MODEL_ORDER)
    std = agg[("macro_f1", "std")].unstack("model_name").reindex(FEATURE_SET_ORDER, columns=MODEL_ORDER)
    annot = np.array([
        [f"{mean.values[i, j]:.3f}\n±{std.values[i, j]:.3f}" for j in range(mean.shape[1])]
        for i in range(mean.shape[0])
    ])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(
        mean, annot=annot, fmt="", cmap="viridis",
        cbar_kws={"label": "Macro F1 (mean)"}, ax=ax,
    )
    ax.set_title(f"Macro-F1 heatmap ({_sp(split)})")
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature set")
    ax.set_xticklabels([_ml(t.get_text()) for t in ax.get_xticklabels()], rotation=0)
    ax.set_yticklabels([_fs(t.get_text()) for t in ax.get_yticklabels()], rotation=0)
    _save(fig, out_dir, "metric_heatmap")


@register_plot("metric_heatmap_grid")
def metric_heatmap_grid(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    metrics = [m for m in HEADLINE_METRICS if m in df.columns]
    agg = _agg(df, split, metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 5))
    axes = np.atleast_1d(axes)
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        mean = agg[(metric, "mean")].unstack("model_name").reindex(FEATURE_SET_ORDER, columns=MODEL_ORDER)
        sns.heatmap(
            mean, annot=True, fmt=".3f", cmap="viridis",
            cbar=(i == len(metrics) - 1), ax=ax,
        )
        ax.set_title(_mt(metric))
        ax.set_xlabel("")
        ax.set_xticklabels([_ml(t.get_text()) for t in ax.get_xticklabels()], rotation=20, ha="right")
        if i == 0:
            ax.set_ylabel("Feature set")
            ax.set_yticklabels([_fs(t.get_text()) for t in ax.get_yticklabels()], rotation=0)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    fig.suptitle(f"Headline metrics by feature set × model ({_sp(split)})", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "metric_heatmap_grid")


@register_plot("per_class_f1")
def per_class_f1(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    if "f1_benign" not in df.columns or "f1_oncogenic" not in df.columns:
        print("[results] per_class_f1: per-class F1 columns missing, skipping")
        return
    agg = _agg(df, split, ["f1_benign", "f1_oncogenic"])
    combos = [(fs, m) for fs in FEATURE_SET_ORDER for m in MODEL_ORDER]
    x = np.arange(len(combos))
    width = 0.4
    b_mean = [agg.loc[c, ("f1_benign", "mean")] for c in combos]
    b_std = [agg.loc[c, ("f1_benign", "std")] for c in combos]
    o_mean = [agg.loc[c, ("f1_oncogenic", "mean")] for c in combos]
    o_std = [agg.loc[c, ("f1_oncogenic", "std")] for c in combos]
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(x - width / 2, b_mean, width, yerr=b_std, capsize=2,
           label="benign", color=CLASS_PALETTE["benign"], edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, o_mean, width, yerr=o_std, capsize=2,
           label="oncogenic", color=CLASS_PALETTE["oncogenic"], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_fs(fs)}\n{_ml(m)}" for fs, m in combos],
                       rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title(f"Per-class F1 by (feature set, model) ({_sp(split)})")
    ax.legend(title="Class", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    _save(fig, out_dir, "per_class_f1")


@register_plot("precision_recall_oncogenic")
def precision_recall_oncogenic(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    needed = {"precision_oncogenic", "recall_oncogenic"}
    if not needed.issubset(df.columns):
        print("[results] precision_recall_oncogenic: missing columns, skipping")
        return
    agg = _agg(df, split, ["precision_oncogenic", "recall_oncogenic"])
    fig, ax = plt.subplots(figsize=(7, 6))
    for fs in FEATURE_SET_ORDER:
        for m in MODEL_ORDER:
            try:
                rx = agg.loc[(fs, m), ("recall_oncogenic", "mean")]
                py = agg.loc[(fs, m), ("precision_oncogenic", "mean")]
                ex = agg.loc[(fs, m), ("recall_oncogenic", "std")]
                ey = agg.loc[(fs, m), ("precision_oncogenic", "std")]
            except KeyError:
                continue
            ax.errorbar(
                rx, py, xerr=ex, yerr=ey,
                fmt=MODEL_MARKERS[m], color=FEATURE_SET_PALETTE[fs],
                markersize=10, markeredgecolor="black", markeredgewidth=0.5,
                capsize=3, elinewidth=0.7, label=f"{_fs(fs)} / {_ml(m)}",
            )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall (Oncogenic)")
    ax.set_ylabel("Precision (Oncogenic)")
    ax.set_title(f"Precision vs Recall on the oncogenic class ({_sp(split)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir, "precision_recall_oncogenic")


@register_plot("val_vs_test_gap")
def val_vs_test_gap(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    del split  # this plot uses both
    val_agg = _agg(df, "val", ["macro_f1"])
    test_agg = _agg(df, "test", ["macro_f1"])
    combos = [(fs, m) for fs in FEATURE_SET_ORDER for m in MODEL_ORDER]
    x = np.arange(len(combos))
    width = 0.4
    v_mean = [val_agg.loc[c, ("macro_f1", "mean")] for c in combos]
    v_std = [val_agg.loc[c, ("macro_f1", "std")] for c in combos]
    t_mean = [test_agg.loc[c, ("macro_f1", "mean")] for c in combos]
    t_std = [test_agg.loc[c, ("macro_f1", "std")] for c in combos]
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(x - width / 2, v_mean, width, yerr=v_std, capsize=2,
           label="val", color="#A3BE8C", edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2, t_mean, width, yerr=t_std, capsize=2,
           label="test", color="#5E81AC", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{_fs(fs)}\n{_ml(m)}" for fs, m in combos],
                       rotation=30, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Macro F1")
    ax.set_title("Validation vs Test macro-F1 (overfitting check)")
    ax.legend(title="Split",
              labels=[_sp(s) for s in SPLITS],
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    _save(fig, out_dir, "val_vs_test_gap")


@register_plot("stability_per_seed")
def stability_per_seed(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    sub = _filter_split(df, split)
    combos = [(fs, m) for fs in FEATURE_SET_ORDER for m in MODEL_ORDER]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(15, 5))
    means = []
    for i, (fs, m) in enumerate(combos):
        vals = sub[(sub.feature_set == fs) & (sub.model_name == m)]["macro_f1"].to_numpy()
        if not len(vals):
            continue
        mean = float(vals.mean())
        means.append(mean)
        ax.bar(i, mean, 0.65, color=MODEL_PALETTE[m], alpha=0.35,
               edgecolor="black", linewidth=0.5)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i, dtype=float) + jitter, vals,
                   color="black", s=18, zorder=3, edgecolor="white", linewidth=0.5)
    means_arr = np.asarray(means)
    ax.set_ylim(max(0, means_arr.min() - 0.07), min(1, means_arr.max() + 0.07))
    ax.set_xticks(np.arange(len(combos)))
    ax.set_xticklabels([f"{_fs(fs)}\n{_ml(m)}" for fs, m in combos],
                       rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Macro F1 (per seed)")
    ax.set_title(f"Per-seed Macro-F1 stability across {sub['seed'].nunique()} seeds ({_sp(split)})")
    handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_PALETTE[m], alpha=0.35) for m in MODEL_ORDER]
    ax.legend(handles, [_ml(m) for m in MODEL_ORDER], title="Model", loc="upper left",
              bbox_to_anchor=(1.01, 1.0), fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "stability_per_seed")


@register_plot("calibration_metrics")
def calibration_metrics(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    metrics = [m for m in ["brier_score", "ece"] if m in df.columns]
    if not metrics:
        print("[results] calibration_metrics: brier_score/ece missing, skipping")
        return
    agg = _agg(df, split, metrics)
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5))
    axes = np.atleast_1d(axes)
    for i, (ax, metric) in enumerate(zip(axes, metrics)):
        _grouped_bar(ax, agg, metric, FEATURE_SET_ORDER, MODEL_ORDER, MODEL_PALETTE, capsize=2)
        ax.set_title(f"{_mt(metric)} (lower = better)")
        ax.set_xlabel("")
        ax.set_ylabel(_mt(metric))
        plt.setp(ax.get_xticklabels(), rotation=30, fontsize=8, ha="right")
        if i == 0:
            ax.legend(title="Model", fontsize=8)
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    fig.suptitle(f"Calibration metrics ({_sp(split)})", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "calibration_metrics")


@register_plot("feature_lift")
def feature_lift(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    """Facet by model: three panels, one per classifier. Each panel shows
    four bars (the three pairwise combos and the union) measuring lift
    over the best single-modality constituent. Same MUSiCaL recipe as
    headline: tier coloring, value labels in bold, no in-panel legend.
    """
    agg = _agg(df, split, ["macro_f1"])
    multi = list(COMPONENT_OF.keys())

    n_models = len(MODEL_ORDER)
    fig, axes = plt.subplots(
        1, n_models, figsize=(2.2 * n_models + 0.6, 2.6), sharey=True,
    )
    axes = np.atleast_1d(axes)

    all_lifts: list[float] = []
    panel_lifts: dict[str, list[float]] = {}
    for model in MODEL_ORDER:
        lifts = []
        for fs in multi:
            target = float(agg.loc[(fs, model), ("macro_f1", "mean")])
            best_single = max(
                float(agg.loc[(c, model), ("macro_f1", "mean")])
                for c in COMPONENT_OF[fs]
            )
            lifts.append(target - best_single)
        panel_lifts[model] = lifts
        all_lifts.extend(lifts)

    ymax = max(0.14, max(all_lifts) + 0.025)
    x = np.arange(len(multi))
    colors = [TIER_PALETTE[fs] for fs in multi]

    for i, (ax, model) in enumerate(zip(axes, MODEL_ORDER)):
        lifts = panel_lifts[model]
        ax.bar(x, lifts, width=0.7, color=colors, alpha=0.95,
               edgecolor="#333", linewidth=0.6)
        annotate_bars(ax, x, lifts, fmt="+{:.02f}", offset=0.004)
        ax.axhline(0, color="#222", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([_fs_short(fs) for fs in multi], rotation=0, ha="center")
        ax.set_xlim(-0.7, len(multi) - 0.3)
        ax.set_ylim(0, ymax)
        ax.set_title(_ml(model), pad=4)
        style_panel(ax)
        if i == 0:
            ax.set_ylabel(r"$\Delta$ Macro-F1 vs best single modality")
        else:
            ax.tick_params(axis="y", labelleft=False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=ACCENT_MID),
        plt.Rectangle((0, 0), 1, 1, color=ACCENT),
    ]
    fig.legend(
        handles, ["Pairwise combo", "Union (all)"],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06),
        fontsize=8, handlelength=1.0, columnspacing=1.4, frameon=False,
    )

    fig.tight_layout()
    _save(fig, out_dir, "feature_lift")


@register_plot("results_table")
def results_table(df: pd.DataFrame, out_dir: Path, split: str) -> None:
    metrics = [m for m in TABLE_METRICS if m in df.columns]
    agg = _agg(df, split, metrics)
    # Identify best feature_set per (model, metric) — per-model bolding.
    best: dict[tuple[str, str], str] = {}
    for m in MODEL_ORDER:
        for metric in metrics:
            scores = {fs: agg.loc[(fs, m), (metric, "mean")] for fs in FEATURE_SET_ORDER}
            picker = min if metric in LOWER_IS_BETTER else max
            best[(m, metric)] = picker(scores, key=scores.get)

    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV (best-cell marked with trailing '*')
    csv_path = out_dir / "results_table.csv"
    with csv_path.open("w") as f:
        f.write("feature_set,model," + ",".join(metrics) + "\n")
        for fs in FEATURE_SET_ORDER:
            for m in MODEL_ORDER:
                cells = [fs, m]
                for metric in metrics:
                    mean = agg.loc[(fs, m), (metric, "mean")]
                    std = agg.loc[(fs, m), (metric, "std")]
                    s = f"{mean:.3f} ± {std:.3f}"
                    if best[(m, metric)] == fs:
                        s += "*"
                    cells.append(s)
                f.write(",".join(cells) + "\n")
    print(f"[results] wrote {csv_path}")

    # Markdown — pretty labels for human reading; best cell wrapped in **bold**.
    md_path = out_dir / "results_table.md"
    with md_path.open("w") as f:
        f.write(f"{_sp(split)} split, mean ± std across {df['seed'].nunique()} seeds. "
                f"**Bold** = best feature set within each model (per metric); "
                f"lower = better for {_mt('ece')}.\n\n")
        f.write("| Feature set | Model | " + " | ".join(_mt(m) for m in metrics) + " |\n")
        f.write("|" + "---|" * (2 + len(metrics)) + "\n")
        for fs in FEATURE_SET_ORDER:
            for m in MODEL_ORDER:
                cells = [_fs(fs), _ml(m)]
                for metric in metrics:
                    mean = agg.loc[(fs, m), (metric, "mean")]
                    std = agg.loc[(fs, m), (metric, "std")]
                    s = f"{mean:.3f} ± {std:.3f}"
                    if best[(m, metric)] == fs:
                        s = f"**{s}**"
                    cells.append(s)
                f.write("| " + " | ".join(cells) + " |\n")
    print(f"[results] wrote {md_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help=f"Per-seed experiments CSV (default: {DEFAULT_CSV})")
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
    p.add_argument("--format", choices=["png", "pdf", "both"], default="png",
                   help="Output file format. Use 'pdf' for vector report figures.")
    args = p.parse_args()

    if args.list:
        for name in PLOT_REGISTRY:
            print(name)
        return

    global FORMATS
    FORMATS = ["png", "pdf"] if args.format == "both" else [args.format]

    out_dir = args.out or (args.csv.parent / "analysis_graphs")
    print(f"[results] loading {args.csv}")
    df = pd.read_csv(args.csv)
    n_seeds = df["seed"].nunique() if "seed" in df.columns else "n/a"
    print(f"[results]   {len(df):,} rows, splits={sorted(df['split'].unique())}, seeds={n_seeds}")

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
