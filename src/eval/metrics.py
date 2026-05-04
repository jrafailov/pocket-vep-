from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """Bin predictions by predicted probability and weight the gap between
    bin-mean confidence and bin-mean accuracy by bin size. Lower is better,
    0 means perfectly calibrated. n_bins=10 is the literature default.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(y_proba, bin_edges[1:-1]), 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        confidence = y_proba[mask].mean()
        accuracy = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(confidence - accuracy)
    return float(ece)


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
        y_true_arr = np.asarray(y_true)
        y_proba_arr = np.asarray(y_proba)
        try:
            result["roc_auc"] = float(roc_auc_score(y_true_arr, y_proba_arr))
        except ValueError:
            result["roc_auc"] = float("nan")

        # PR-AUC is asymmetric in which class is positive. y_proba comes in as
        # P(class 1), and the encoder puts oncogenic at index 1, so this is
        # PR-AUC with oncogenic as positive (the call we care about clinically).
        # The baseline is positive-class prevalence -- a no-skill classifier
        # that just outputs the prior gets PR-AUC equal to it, so anything
        # above that is real lift.
        try:
            result["pr_auc"] = float(average_precision_score(y_true_arr, y_proba_arr))
        except ValueError:
            result["pr_auc"] = float("nan")
        result["pr_auc_baseline"] = float(y_true_arr.mean())

        # Calibration. Brier captures overall probability quality (lower better);
        # ECE captures pure calibration miscalibration independent of sharpness.
        try:
            result["brier_score"] = float(brier_score_loss(y_true_arr, y_proba_arr))
        except ValueError:
            result["brier_score"] = float("nan")
        result["ece"] = expected_calibration_error(y_true_arr, y_proba_arr)

    return result


def format_report(y_true, y_pred, label_names: Iterable[str]) -> str:
    return classification_report(
        y_true, y_pred, target_names=list(label_names), zero_division=0
    )
