from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true,
    y_pred,
    label_names: Iterable[str],
    feature_keys: Iterable[str],
    model_name: str,
    split: str,
    y_proba=None,
) -> dict:
    label_names = list(label_names)
    feature_keys = list(feature_keys)

    result = {
        "model_name": model_name,
        "feature_keys": feature_keys,
        "split": split,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # per-class precision / recall / f1
    per_class_p = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, name in enumerate(label_names):
        result[f"precision_{name}"] = float(per_class_p[i])
        result[f"recall_{name}"] = float(per_class_r[i])
        result[f"f1_{name}"] = float(per_class_f[i])

    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            result["roc_auc"] = float("nan")

    return result


def format_report(y_true, y_pred, label_names: Iterable[str]) -> str:
    return classification_report(
        y_true, y_pred, target_names=list(label_names), zero_division=0
    )
