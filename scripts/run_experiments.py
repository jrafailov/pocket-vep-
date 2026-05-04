"""Run the N-models x N-feature-sets x N-seeds experiment grid.

Feature sets cover the three orthogonal modalities (sequence, structure,
evolution) plus all pairwise combos and the full union, so the ablation can
isolate the contribution of each modality.

Each --seed re-splits the data and re-fits the models. Within a single seed
every (feature_set, model) combo trains on the same train/val/test, so
arm-vs-arm deltas are paired and split variance cancels. Across seeds we
average to capture variance from "which genes happened to land in test".

Prereqs:
    python scripts/download_clinvar.py        # data/interim/clinvar_labeled.parquet
    python scripts/build_structure_cache.py   # data/processed/{plddt,structure_features}.parquet
    python scripts/build_conservation_cache.py # data/processed/conservation_cache.parquet
    python scripts/build_feature_matrix.py    # data/processed/feature_matrix.parquet

The materialized feature_matrix.parquet contains every block's columns inner-
joined to the rows with full coverage. Splitting + training operate on this
single file so every feature-set variant trains on the SAME row set
(otherwise restricted feature sets would use fewer rows and the comparison
would be unfair).

Run:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --models xgboost              # subset of models
    python scripts/run_experiments.py --feature-sets sequence       # subset of feature sets
    python scripts/run_experiments.py --seeds 42 43 44              # custom seed list
    python scripts/run_experiments.py --interpret-methods native permutation
    python scripts/run_experiments.py --no-interpret                # skip interpretation
    python scripts/run_experiments.py --out-dir results/run_a/      # redirect outputs
    python scripts/run_experiments.py \\
        --feature-matrix data/processed/feature_matrix.debug.parquet  # debug subset

Output layout (rooted at --out-dir, default `results/`):
    {out_dir}/experiments.csv          # raw, one row per (seed, fs, model, split)
    {out_dir}/experiments_summary.csv  # mean / std across seeds, one row per (fs, model, split)
    {out_dir}/interpretations/{feature_set}_{model_name}.csv  # first seed only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make `src` importable when this script is run directly (no editable install needed).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import make_splits
from src.eval import interpret_model
from src.features import load_feature_matrix
from src.models.trainer import Trainer

ALL_FEATURE_SETS: dict[str, list[str]] = {
    # Single-modality arms.
    "sequence": ["sequence"],
    "structure": ["structure"],
    "evolution": ["evolution"],
    # Pairwise combos.
    "seq_struct": ["sequence", "structure"],
    "seq_evo": ["sequence", "evolution"],
    "struct_evo": ["structure", "evolution"],
    # Full union.
    "all": ["sequence", "structure", "evolution"],
}

ALL_MODELS = ["random_forest", "mlp", "xgboost"]
ALL_INTERPRET_METHODS = ["native", "permutation", "shap"]

EXPERIMENTS_CSV_NAME = "experiments.csv"
SUMMARY_CSV_NAME = "experiments_summary.csv"
INTERPRETATIONS_SUBDIR = "interpretations"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
SUMMARY_METRICS = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--feature-matrix",
        default="data/processed/feature_matrix.parquet",
        type=Path,
        help="Path to the materialized feature matrix "
             "(produced by scripts/build_feature_matrix.py).",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=ALL_MODELS,
        choices=ALL_MODELS,
        help="Subset of models to run.",
    )
    ap.add_argument(
        "--feature-sets",
        nargs="+",
        default=list(ALL_FEATURE_SETS),
        choices=list(ALL_FEATURE_SETS),
        help="Subset of feature-set names to run.",
    )
    ap.add_argument(
        "--out-dir",
        default="results",
        type=Path,
        help="Root directory for outputs. Layout inside is fixed: "
             f"{EXPERIMENTS_CSV_NAME} and {INTERPRETATIONS_SUBDIR}/.",
    )
    ap.add_argument(
        "--interpret-methods",
        nargs="+",
        default=ALL_INTERPRET_METHODS,
        choices=ALL_INTERPRET_METHODS,
        help="Which interpretation methods to run per model.",
    )
    ap.add_argument(
        "--no-interpret",
        action="store_true",
        help="Skip interpretation entirely.",
    )
    ap.add_argument(
        "--shap-sample-size",
        default=500,
        type=int,
        help="Rows to sample for SHAP (caps KernelExplainer cost on MLP).",
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Seeds to sweep over. Each seed re-splits the data and re-fits the "
             "models, yielding one row per (seed, feature_set, model, split) in "
             "experiments.csv. Default: 5 seeds (42-46).",
    )
    return ap.parse_args()


def _write_interpretation(interp_dir: Path, fs_name: str, model_name: str, importances: dict) -> None:
    if not importances:
        return
    interp_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    for method, df in importances.items():
        d = df.copy()
        d.insert(0, "method", method)
        parts.append(d)
    out_df = pd.concat(parts, ignore_index=True)
    out_path = interp_dir / f"{fs_name}_{model_name}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"  wrote interpretation: {out_path}")


def _summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw per-seed rows into mean / std across seeds."""
    metric_cols = [c for c in SUMMARY_METRICS if c in raw.columns]
    grouped = raw.groupby(["feature_set", "model_name", "split"], sort=False)
    agg_spec = {f"{m}_mean": (m, "mean") for m in metric_cols}
    agg_spec.update({f"{m}_std": (m, "std") for m in metric_cols})
    agg_spec["n_seeds"] = ("accuracy", "count")
    return grouped.agg(**agg_spec).reset_index()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.feature_matrix}")
    df, schema = load_feature_matrix(args.feature_matrix)
    print(f"  {len(df):,} rows with full feature coverage")
    print(df["ML_Label"].value_counts().to_string())

    out_dir = args.out_dir
    interp_dir = out_dir / INTERPRETATIONS_SUBDIR
    first_seed = args.seeds[0]

    rows: list[dict] = []
    for seed in args.seeds:
        print(f"\n########## seed={seed} ##########")
        train_df, val_df, test_df = make_splits(df, seed=seed)
        trainer = Trainer(train_df, val_df, test_df, schema=schema)

        for fs_name in args.feature_sets:
            feature_keys = ALL_FEATURE_SETS[fs_name]
            for model_name in args.models:
                tag = f"[seed={seed} | {fs_name} | {model_name}]"
                print(f"\n=== {tag} ===")
                try:
                    result = trainer.run(feature_keys, model_name, seed=seed)
                except NotImplementedError as e:
                    print(f"  skipped: {e}")
                    continue
                except Exception as e:
                    print(f"  FAILED: {type(e).__name__}: {e}")
                    continue

                for metrics in result["metrics"]:
                    metrics["feature_set"] = fs_name
                    metrics["seed"] = seed
                    rows.append(metrics)
                    print(
                        f"  [{metrics['split']:<4}] "
                        f"accuracy={metrics['accuracy']:.3f}  "
                        f"balanced_acc={metrics['balanced_accuracy']:.3f}  "
                        f"macro_f1={metrics['macro_f1']:.3f}"
                        + (f"  roc_auc={metrics['roc_auc']:.3f}" if "roc_auc" in metrics else "")
                    )

                # Interpretation only on the first seed; otherwise we'd write 5x
                # the importance files with no good way to merge them.
                if not args.no_interpret and seed == first_seed:
                    try:
                        importances = interpret_model(
                            model_name,
                            result["model"],
                            result["X_train"],
                            result["X_val"],
                            result["y_val_enc"],
                            methods=args.interpret_methods,
                            shap_sample_size=args.shap_sample_size,
                            seed=seed,
                        )
                        _write_interpretation(interp_dir, fs_name, model_name, importances)
                    except Exception as e:
                        print(f"  interpretation FAILED: {type(e).__name__}: {e}")

    if not rows:
        print("\nNo successful runs. Nothing to save.")
        return

    results_df = pd.DataFrame(rows)
    front = [
        "seed",
        "feature_set",
        "model_name",
        "split",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
    ]
    cols = front + [c for c in results_df.columns if c not in front]
    results_df = results_df[[c for c in cols if c in results_df.columns]]

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / EXPERIMENTS_CSV_NAME
    results_df.to_csv(raw_path, index=False)
    print(f"\nWrote {raw_path}  ({len(results_df)} rows across {len(args.seeds)} seed(s))")

    summary = _summarize(results_df)
    summary_path = out_dir / SUMMARY_CSV_NAME
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")

    print("\n=== Summary (mean +/- std across seeds) ===")
    headline_cols = ["feature_set", "model_name", "split"]
    pretty = summary[headline_cols].copy()
    for m in ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]:
        if f"{m}_mean" in summary.columns:
            pretty[m] = [
                f"{mu:.3f} +/- {sd:.3f}"
                for mu, sd in zip(summary[f"{m}_mean"], summary[f"{m}_std"].fillna(0))
            ]
    pretty["n"] = summary["n_seeds"]
    print(pretty.to_string(index=False))


if __name__ == "__main__":
    main()
