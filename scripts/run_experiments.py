"""Run the 3-models x 3-feature-sets experiment grid.

Prereqs:
    python scripts/download_clinvar.py        # data/interim/clinvar_labeled.parquet
    python scripts/build_structure_cache.py   # data/processed/{plddt,structure_features}.parquet
    python scripts/build_feature_matrix.py    # data/processed/feature_matrix.parquet

The materialized feature_matrix.parquet contains every block's columns inner-
joined to the rows with full coverage. Splitting + training operate on this
single file so every feature-set variant trains on the SAME row set
(otherwise structure-only / combined runs would use fewer rows than
sequence-only, and the comparison would be unfair).

Run:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --models xgboost              # subset of models
    python scripts/run_experiments.py --feature-sets sequence       # subset of feature sets
    python scripts/run_experiments.py --interpret-methods native permutation
    python scripts/run_experiments.py --no-interpret                # skip interpretation
    python scripts/run_experiments.py --out-dir results/run_a/      # redirect outputs
    python scripts/run_experiments.py \\
        --feature-matrix data/processed/feature_matrix.debug.parquet  # debug subset

Output layout (rooted at --out-dir, default `results/`):
    {out_dir}/experiments.csv
    {out_dir}/interpretations/{feature_set}_{model_name}.csv
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
    "sequence": ["sequence"],
    "structure": ["structure"],
    "evolution": ["evolution"],
    "combined": ["sequence", "structure"],
}

ALL_MODELS = ["random_forest", "mlp", "xgboost"]
ALL_INTERPRET_METHODS = ["native", "permutation", "shap"]

EXPERIMENTS_CSV_NAME = "experiments.csv"
INTERPRETATIONS_SUBDIR = "interpretations"


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
    ap.add_argument("--seed", default=42, type=int)
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


def main() -> None:
    args = parse_args()

    print(f"Loading {args.feature_matrix}")
    df, schema = load_feature_matrix(args.feature_matrix)
    print(f"  {len(df):,} rows with full feature coverage")
    print(df["ML_Label"].value_counts().to_string())

    train_df, val_df, test_df = make_splits(df, seed=args.seed)
    print(
        f"Splits: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}"
    )

    trainer = Trainer(train_df, val_df, test_df, schema=schema)

    out_dir = args.out_dir
    interp_dir = out_dir / INTERPRETATIONS_SUBDIR

    rows: list[dict] = []
    for fs_name in args.feature_sets:
        feature_keys = ALL_FEATURE_SETS[fs_name]
        for model_name in args.models:
            tag = f"[{fs_name} | {model_name}]"
            print(f"\n=== {tag} ===")
            try:
                result = trainer.run(feature_keys, model_name)
            except NotImplementedError as e:
                print(f"  skipped: {e}")
                continue
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                continue

            for metrics in result["metrics"]:
                metrics["feature_set"] = fs_name
                rows.append(metrics)
                print(
                    f"  [{metrics['split']:<4}] "
                    f"accuracy={metrics['accuracy']:.3f}  "
                    f"balanced_acc={metrics['balanced_accuracy']:.3f}  "
                    f"macro_f1={metrics['macro_f1']:.3f}"
                    + (f"  roc_auc={metrics['roc_auc']:.3f}" if "roc_auc" in metrics else "")
                )

            if not args.no_interpret:
                try:
                    importances = interpret_model(
                        model_name,
                        result["model"],
                        result["X_train"],
                        result["X_val"],
                        result["y_val_enc"],
                        methods=args.interpret_methods,
                        shap_sample_size=args.shap_sample_size,
                        seed=args.seed,
                    )
                    _write_interpretation(interp_dir, fs_name, model_name, importances)
                except Exception as e:
                    print(f"  interpretation FAILED: {type(e).__name__}: {e}")

    if not rows:
        print("\nNo successful runs. Nothing to save.")
        return

    results_df = pd.DataFrame(rows)
    # stable column ordering
    front = [
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
    out_path = out_dir / EXPERIMENTS_CSV_NAME
    results_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print("\n=== Summary ===")
    print(
        results_df[
            ["feature_set", "model_name", "split", "accuracy", "balanced_accuracy", "macro_f1"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
