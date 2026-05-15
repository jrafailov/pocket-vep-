"""Plot ROC curves for the 7 feature configurations on a single seed.

Two output modes (both run by default):

- best: one panel, seven curves. For each feature set we use the classifier
  with the highest mean test ROC-AUC in the 5-seed sweep. Compact summary.

- per-model: three panels (RF, MLP, XGBoost), each showing all seven
  configs as separate curves. Trains 21 models on the chosen seed. Lines
  up cleanly under the 3-panel macro-F1 bar plot.

    python scripts/plot_roc_curves.py
    python scripts/plot_roc_curves.py --mode best
    python scripts/plot_roc_curves.py --mode per-model --seed 43
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data import make_splits
from src.features import load_feature_matrix
from src.models.trainer import Trainer

from _plot_style import apply_style

# Mirror run_experiments.py — keep the canonical block keys in one place.
ALL_FEATURE_SETS: dict[str, list[str]] = {
    "sequence": ["sequence"],
    "structure": ["structure"],
    "evolution": ["evolution"],
    "seq_struct": ["sequence", "structure"],
    "seq_evo": ["sequence", "evolution"],
    "struct_evo": ["structure", "evolution"],
    "all": ["sequence", "structure", "evolution"],
}

FEATURE_ORDER = [
    "sequence", "structure", "evolution",
    "seq_struct", "seq_evo", "struct_evo",
    "all",
]

# Per-config colors. Distinct enough that 7 overlapping curves stay
# readable, with the union curve in near-black so it pops.
CURVE_COLOR = {
    "sequence":   "#88C0D0",
    "structure":  "#D08770",
    "evolution":  "#A3BE8C",
    "seq_struct": "#B48EAD",
    "seq_evo":    "#81A1C1",
    "struct_evo": "#BF616A",
    "all":        "#2E3440",
}

LABEL = {
    "sequence":   "Sequence",
    "structure":  "Structure",
    "evolution":  "Evolution",
    "seq_struct": "Seq + Str",
    "seq_evo":    "Seq + Evo",
    "struct_evo": "Str + Evo",
    "all":        "All combined",
}

MODEL_LABEL = {"random_forest": "RF", "mlp": "MLP", "xgboost": "XGB"}


def best_model_per_config(csv_path: Path) -> dict[str, str]:
    """For each feature_set, pick the classifier with the highest mean test
    ROC-AUC across the 5-seed sweep."""
    df = pd.read_csv(csv_path)
    df = df[df.split == "test"]
    grouped = (
        df.groupby(["feature_set", "model_name"])["roc_auc"]
        .mean()
        .reset_index()
    )
    out: dict[str, str] = {}
    for fs, sub in grouped.groupby("feature_set"):
        out[fs] = sub.loc[sub["roc_auc"].idxmax(), "model_name"]
    return out


MODEL_ORDER = ["random_forest", "mlp", "xgboost"]


def _train_one(trainer: Trainer, fs: str, model_name: str, seed: int):
    keys = ALL_FEATURE_SETS[fs]
    result = trainer.run(keys, model_name, seed=seed)
    model = result["model"]
    X_test = result["X_test"]
    y_test = result["y_test_enc"]
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)
    return fpr, tpr, auc


def _draw_roc_panel(ax, curves: dict, *, with_legend: bool, with_ylabel: bool):
    """Render the seven configs' (fpr, tpr, auc) tuples on one axis.

    `curves` is a dict feature_set -> (fpr, tpr, auc); union arm gets a
    thicker line so it pops against the others.
    """
    for fs in FEATURE_ORDER:
        fpr, tpr, auc = curves[fs]
        lw = 2.6 if fs == "all" else 1.6
        ax.plot(
            fpr, tpr,
            color=CURVE_COLOR[fs], lw=lw,
            # Drop the :<14 padding (it added trailing whitespace inside
            # each legend label) and use a single space + the AUC value.
            # With panels now 2.83" wide, padding makes the legend overflow.
            label=f"{LABEL[fs]} {auc:.3f}",
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.005)
    # Font sizes match _plot_style.apply_style defaults (axes.labelsize=11,
    # xtick.labelsize=9) so this figure reads consistently with the top
    # headline figure in the stacked panel in the report.
    ax.set_xlabel("False positive rate")
    if with_ylabel:
        ax.set_ylabel("True positive rate")
    if with_legend:
        # Per-config AUC table. Legend sits inside the panel (2.83 in
        # wide after the figsize match), so size is tight enough to fit
        # while staying readable after the 0.95 \linewidth downscale.
        ax.legend(loc="lower right", fontsize=9, frameon=False,
                  handlelength=1.2, borderpad=0.25, labelspacing=0.25,
                  handletextpad=0.45)


def plot_best(trainer: Trainer, best: dict, seed: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for fs in FEATURE_ORDER:
        model_name = best[fs]
        print(f"[roc:best] training {fs} / {model_name}")
        fpr, tpr, auc = _train_one(trainer, fs, model_name, seed)
        lw = 2.4 if fs == "all" else 1.5
        ax.plot(
            fpr, tpr,
            color=CURVE_COLOR[fs], lw=lw,
            label=f"{LABEL[fs]:<14}  AUC = {auc:.3f}  ({MODEL_LABEL[model_name]})",
        )
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", lw=0.8, label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.005)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=7.5, frameon=False,
              handlelength=1.6, borderpad=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[roc:best] wrote {out_path}")


def plot_per_model(trainer: Trainer, seed: int, out_path: Path) -> None:
    """3 panels (RF / MLP / XGBoost), each with the 7 feature configs as
    separate curves. Trains 21 models on the requested seed."""
    curves: dict[str, dict] = {m: {} for m in MODEL_ORDER}
    for model_name in MODEL_ORDER:
        for fs in FEATURE_ORDER:
            print(f"[roc:per-model] training {fs} / {model_name}")
            curves[model_name][fs] = _train_one(trainer, fs, model_name, seed)

    # Figsize width matches the top headline_macro_f1_perconfig figure
    # (2.5 * n_models + 1.0 = 8.5 wide) so the RF / MLP / XGBoost columns
    # line up between the two stacked figures in the report. Height kept
    # so the source aspect ratio (~3:1) matches what the old wider figure
    # had — taller panels would push the body over the 6-page limit.
    fig, axes = plt.subplots(1, 3, figsize=(2.5 * 3 + 1.0, 2.4), sharey=True)
    for i, (ax, model_name) in enumerate(zip(axes, MODEL_ORDER)):
        _draw_roc_panel(
            ax, curves[model_name],
            with_legend=True,
            with_ylabel=(i == 0),
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"))
    print(f"[roc:per-model] wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--feature-matrix", type=Path,
                   default=ROOT / "data/processed/feature_matrix.parquet")
    p.add_argument("--experiments-csv", type=Path,
                   default=ROOT / "results/full_run/experiments.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", choices=["best", "per-model", "both"], default="both",
                   help="Which output to generate.")
    p.add_argument("--out-best", type=Path,
                   default=ROOT / "report/figures/roc_curves_best.pdf")
    p.add_argument("--out-per-model", type=Path,
                   default=ROOT / "report/figures/roc_curves_per_model.pdf")
    args = p.parse_args()

    apply_style()

    print(f"[roc] loading {args.feature_matrix}")
    df, schema = load_feature_matrix(args.feature_matrix)
    train_df, val_df, test_df = make_splits(df, seed=args.seed)
    trainer = Trainer(train_df, val_df, test_df, schema=schema)

    if args.mode in ("best", "both"):
        best = best_model_per_config(args.experiments_csv)
        print("[roc:best] best classifier per config (by mean test ROC-AUC):")
        for fs in FEATURE_ORDER:
            print(f"  {fs:<10} -> {best[fs]}")
        plot_best(trainer, best, args.seed, args.out_best)

    if args.mode in ("per-model", "both"):
        plot_per_model(trainer, args.seed, args.out_per_model)


if __name__ == "__main__":
    main()
