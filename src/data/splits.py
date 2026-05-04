from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def make_splits(
    df: pd.DataFrame,
    label_col: str = "ML_Label",
    group_col: str = "GeneSymbol",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group-aware train/val/test split. Default 70/15/15.

    Splits by group_col (default GeneSymbol) so no gene's variants
    straddle splits. Without this the model can learn protein-identity
    shortcuts (a gene whose training variants are mostly oncogenic
    teaches the model to predict oncogenic for that gene at test time)
    which inflates metrics past what the features actually buy us.

    sklearn has no group + stratified single-split, but with ~1,200 genes
    and a mild label imbalance random group assignment preserves the
    label balance well enough for our purposes. Per-split balance is
    printed so any drift is visible.
    """
    if group_col not in df.columns:
        raise KeyError(
            f"{group_col!r} not in df.columns. Make sure it is in "
            f"build_feature_matrix.py PASSTHROUGH_COLS so the materialized "
            f"matrix carries it through."
        )

    groups = df[group_col].astype(str)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(df, groups=groups))
    trainval = df.iloc[trainval_idx]
    test = df.iloc[test_idx]

    rel_val = val_size / (1 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
    train_idx, val_idx = next(
        gss_val.split(trainval, groups=trainval[group_col].astype(str))
    )
    train = trainval.iloc[train_idx]
    val = trainval.iloc[val_idx]

    _report_splits(train, val, test, label_col, group_col)
    return train, val, test


def _report_splits(train, val, test, label_col, group_col):
    print(f"Splits grouped by {group_col}")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        balance = split[label_col].value_counts(normalize=True).sort_index()
        bal_str = "  ".join(f"{k}={v:.1%}" for k, v in balance.items())
        print(
            f"  {name:5s} rows={len(split):>6,}  "
            f"genes={split[group_col].nunique():>4,}  {bal_str}"
        )
