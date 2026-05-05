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
from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results_new_v2/experiments.csv"

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
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[results] wrote {path}")


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
    width_total: float = 0.8, capsize: float = 3.0,
) -> None:
    """Manual grouped bar with yerr, since we already have aggregated data."""
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
        ax.bar(
            x + offset, means, width, yerr=stds, capsize=capsize,
            label=_ml(h) if h in MODEL_LABEL else h,
            color=palette[h], edgecolor="black", linewidth=0.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_fs(fs) if fs in FEATURE_SET_LABEL else fs for fs in x_order])


def _zoom_ylim(means: np.ndarray, stds: np.ndarray, pad: float = 0.03) -> tuple[float, float]:
    lo = float(np.nanmin(means - stds)) - pad
    hi = float(np.nanmax(means + stds)) + pad
    return max(0.0, lo), min(1.0, hi)


# ---------------------------------------------------------------------------
# Headline plots: one per metric (auto-registered below)
# ---------------------------------------------------------------------------

def _draw_headline(df: pd.DataFrame, out_dir: Path, split: str, metric: str) -> None:
    agg = _agg(df, split, [metric])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    _grouped_bar(ax, agg, metric, FEATURE_SET_ORDER, MODEL_ORDER, MODEL_PALETTE)
    for sep in TIER_BREAKS:
        ax.axvline(sep, color="#888", linestyle="--", alpha=0.4, linewidth=0.8)
    means = agg[(metric, "mean")].to_numpy()
    stds = agg[(metric, "std")].to_numpy()
    ax.set_ylim(*_zoom_ylim(means, stds))
    n_seeds = df["seed"].nunique()
    ax.set_ylabel(_mt(metric))
    ax.set_xlabel("")
    ax.set_title(f"{_sp(split)} {_mt(metric)} by feature set and model (mean ± std, {n_seeds} seeds)")
    ax.legend(title="Model", loc="lower right", framealpha=0.9, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8.5)
    fig.tight_layout()
    _save(fig, out_dir, f"headline_{metric}")


def _make_headline_fn(metric: str) -> PlotFn:
    def fn(df: pd.DataFrame, out_dir: Path, split: str) -> None:
        _draw_headline(df, out_dir, split, metric)
    fn.__name__ = f"headline_{metric}"
    return fn


for _m in HEADLINE_METRICS:
    register_plot(f"headline_{_m}")(_make_headline_fn(_m))


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
    agg = _agg(df, split, ["macro_f1"])
    multi = list(COMPONENT_OF.keys())
    x = np.arange(len(multi))
    width = 0.8 / len(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(MODEL_ORDER):
        lifts = []
        for fs in multi:
            target = agg.loc[(fs, m), ("macro_f1", "mean")]
            best_single = max(agg.loc[(c, m), ("macro_f1", "mean")] for c in COMPONENT_OF[fs])
            lifts.append(target - best_single)
        offset = (i - (len(MODEL_ORDER) - 1) / 2) * width
        ax.bar(x + offset, lifts, width, label=_ml(m), color=MODEL_PALETTE[m],
               edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_fs(fs) for fs in multi], rotation=15, ha="right")
    ax.set_xlabel("Multi-modal feature set")
    ax.set_ylabel("Δ Macro-F1 vs best constituent single-modality")
    ax.set_title(f"Lift from combining feature sets ({_sp(split)})")
    ax.legend(title="Model")
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
    args = p.parse_args()

    if args.list:
        for name in PLOT_REGISTRY:
            print(name)
        return

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
