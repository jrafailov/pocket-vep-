"""Run the 3-models x 3-feature-sets experiment grid.

Prereq:
    python scripts/download_clinvar.py   # writes data/interim/clinvar_labeled.parquet

Run:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --eval-on test        # final headline numbers
    python scripts/run_experiments.py --models xgboost      # subset
    python scripts/run_experiments.py --feature-sets sequence
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make `src` importable when this script is run directly (no editable install needed).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_clinvar_labeled, make_splits
from src.models.trainer import Trainer

ALL_FEATURE_SETS: dict[str, list[str]] = {
    "sequence": ["sequence"],
    # "structure": ["structure"],
    # "combined": ["sequence", "structure"],
}

ALL_MODELS = ["decision_tree", "mlp", "xgboost"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        default="data/interim/clinvar_labeled.parquet",
        type=Path,
        help="Path to the cleaned ClinVar parquet.",
    )
    ap.add_argument(
        "--eval-on",
        choices=["val", "test"],
        default="val",
        help="Evaluate on val (iteration) or test (final numbers).",
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
        "--out",
        default="results/experiments.csv",
        type=Path,
        help="Where to write the results table.",
    )
    ap.add_argument("--seed", default=42, type=int)
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading {args.data}")
    df = load_clinvar_labeled(args.data)
    print(f"  {len(df):,} labeled rows")
    print(df["ML_Label"].value_counts().to_string())

    train_df, val_df, test_df = make_splits(df, seed=args.seed)
    print(
        f"Splits: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}"
    )

    trainer = Trainer(train_df, val_df, test_df)

    rows: list[dict] = []
    for fs_name in args.feature_sets:
        feature_keys = ALL_FEATURE_SETS[fs_name]
        for model_name in args.models:
            tag = f"[{fs_name} | {model_name}]"
            print(f"\n=== {tag} ===")
            try:
                metrics = trainer.run(feature_keys, model_name, eval_on=args.eval_on)
            except NotImplementedError as e:
                print(f"  skipped: {e}")
                continue
            except Exception as e:
                print(f"  FAILED: {type(e).__name__}: {e}")
                continue

            metrics["feature_set"] = fs_name
            rows.append(metrics)
            print(
                f"  accuracy={metrics['accuracy']:.3f}  "
                f"balanced_acc={metrics['balanced_accuracy']:.3f}  "
                f"macro_f1={metrics['macro_f1']:.3f}"
            )
            for k, v in metrics.items():
                if k.startswith(("precision_", "recall_", "f1_")):
                    print(f"    {k}: {v:.3f}")
            if "roc_auc" in metrics:
                print(f"    roc_auc: {metrics['roc_auc']:.3f}")

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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print("\n=== Summary ===")
    print(
        results_df[
            ["feature_set", "model_name", "accuracy", "balanced_accuracy", "macro_f1"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
