"""Find variants where the full-features model rescues a call that the
sequence-only model would have missed (or vice versa). These are candidate
case studies for the report.

Re-fits two XGBoost models on the seed=42 gene-grouped split using the same
hyperparameters as scripts/run_experiments.py, predicts on test, and reports
the top flipped variants by probability swing. Output is a small CSV at
results/case_study_candidates.csv plus a printed shortlist.

Usage
    python scripts/case_study_finder.py
    python scripts/case_study_finder.py --top 20 --gene BRAF
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.splits import make_splits  # noqa: E402

MATRIX_PATH = ROOT / "data/processed/feature_matrix.parquet"
OUT_PATH = ROOT / "results/case_study_candidates.csv"

# Column groupings match scripts/run_experiments.py.
SEQ_COLS = (
    ["Delta_Mass", "Delta_Hydro", "Delta_Charge", "BLOSUM62"]
    + [f"WT_AA_{a}" for a in "ACDEFGHIKLMNPQRSTVWY"]
    + [f"MT_AA_{a}" for a in "ACDEFGHIKLMNPQRSTVWY"]
)
STRUCT_COLS = [
    "dist_to_nearest_pocket", "in_pocket", "druggability", "sasa",
    "plddt", "ss_H", "ss_E", "ss_L",
]
EVO_COLS = ["phylop100way", "phastcons100way"]
ALL_COLS = SEQ_COLS + STRUCT_COLS + EVO_COLS

# XGBoost hyperparameters from scripts/run_experiments.py / src/models/classifiers.py.
XGB_PARAMS = dict(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    eval_metric="logloss", verbosity=0, tree_method="hist",
)


LABEL_MAP = {"benign": 0, "oncogenic": 1}


def _y(df, label_col):
    return df[label_col].map(LABEL_MAP).astype(int).values


def fit_predict(train, test, feat_cols, label_col="ML_Label", seed=42):
    """Fit XGBoost with the class-imbalance-aware scale_pos_weight that the
    experiments script uses, then return test-set predicted probabilities
    for the oncogenic class."""
    y_train = _y(train, label_col)
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    spw = neg / pos
    clf = XGBClassifier(scale_pos_weight=spw, random_state=seed, **XGB_PARAMS)
    clf.fit(train[feat_cols].values, y_train)
    return clf.predict_proba(test[feat_cols].values)[:, 1]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--top", type=int, default=15,
                   help="Number of top candidates to print (default 15).")
    p.add_argument("--gene", default=None,
                   help="Filter shortlist to this gene symbol.")
    p.add_argument("--seed", type=int, default=42,
                   help="Split seed (default 42, matching seed-1 of the experiments).")
    args = p.parse_args()

    print(f"[case-study] loading {MATRIX_PATH}")
    df = pd.read_parquet(MATRIX_PATH)
    print(f"[case-study]   {len(df):,} rows, {df['GeneSymbol'].nunique():,} genes")

    train, _val, test = make_splits(df, seed=args.seed)

    print("[case-study] fitting sequence-only XGBoost")
    p_seq = fit_predict(train, test, SEQ_COLS, seed=args.seed)
    print("[case-study] fitting all-features XGBoost")
    p_all = fit_predict(train, test, ALL_COLS, seed=args.seed)

    out = test[["GeneSymbol", "protein_change_clean", "ML_Label",
                "dist_to_nearest_pocket", "in_pocket", "druggability",
                "sasa", "plddt", "phylop100way", "phastcons100way"]].copy()
    out["y"] = _y(test, "ML_Label")  # int-coded label for boolean filters
    out["prob_seq"] = p_seq
    out["prob_all"] = p_all
    out["pred_seq"] = (p_seq >= 0.5).astype(int)
    out["pred_all"] = (p_all >= 0.5).astype(int)
    out["delta_prob"] = p_all - p_seq

    # "Rescued": truly oncogenic, seq missed it, all caught it.
    rescued = out[(out.y == 1) & (out.pred_seq == 0) & (out.pred_all == 1)]
    # "Cleared": truly benign, seq false-flagged it, all cleared it.
    cleared = out[(out.y == 0) & (out.pred_seq == 1) & (out.pred_all == 0)]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rescued.assign(flip="rescued").reindex(columns=list(out.columns) + ["flip"]) \
        .to_csv(OUT_PATH, index=False)
    cleared.assign(flip="cleared").to_csv(OUT_PATH, mode="a", index=False, header=False)
    print(f"[case-study] wrote {len(rescued)} rescued + {len(cleared)} cleared rows to {OUT_PATH.relative_to(ROOT)}")

    def _print(group, label):
        view = group.sort_values("delta_prob", key=abs, ascending=False)
        if args.gene:
            view = view[view.GeneSymbol == args.gene]
        view = view.head(args.top)
        print(f"\n=== Top {len(view)} {label} variants (|Δprob| sorted) ===")
        cols = ["GeneSymbol", "protein_change_clean", "prob_seq", "prob_all",
                "delta_prob", "dist_to_nearest_pocket", "in_pocket",
                "druggability", "plddt", "sasa", "phylop100way"]
        with pd.option_context("display.max_colwidth", 30,
                               "display.width", 180,
                               "display.float_format", "{:.3f}".format):
            print(view[cols].to_string(index=False))

    _print(rescued, "rescued (truth=onc, seq=ben, all=onc)")
    _print(cleared, "cleared (truth=ben, seq=onc, all=ben)")


if __name__ == "__main__":
    main()
