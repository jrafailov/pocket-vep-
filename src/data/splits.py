from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(
    df: pd.DataFrame,
    label_col: str = "ML_Label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train/val/test split. Default 70/15/15."""
    trainval, test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=seed
    )
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(
        trainval, test_size=rel_val, stratify=trainval[label_col], random_state=seed
    )
    return train, val, test
