"""Per-model interpretation functions.

Each model exposes its own attribution quirks (trees have native importance,
MLP has none), so each model gets its own function. They share a common
return shape: ``dict[method_name, DataFrame[feature, importance, rank]]``.

Method opt-in/out: pass ``methods=("native", "permutation", "shap")`` or any
subset. Methods unsupported by a model (e.g. ``native`` on MLP) are skipped
with a warning instead of raising.
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


SUPPORTED_METHODS = ("native", "permutation", "shap")


def _to_importance_df(values, feature_names: Iterable[str]) -> pd.DataFrame:
    feature_names = list(feature_names)
    values = np.asarray(values, dtype=float)
    df = pd.DataFrame({"feature": feature_names, "importance": values})
    df = df.sort_values("importance", ascending=False, kind="mergesort").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def _subsample(X: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n is None or n >= len(X):
        return X
    return X.sample(n=n, random_state=seed)


def _normalize_methods(methods: Iterable[str]) -> list[str]:
    methods = list(methods)
    unknown = [m for m in methods if m not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(
            f"Unknown interpretation method(s): {unknown}. "
            f"Supported: {list(SUPPORTED_METHODS)}"
        )
    return methods


def _permutation(model, X_eval, y_eval, feature_names, seed) -> pd.DataFrame:
    result = permutation_importance(
        model, X_eval, y_eval, n_repeats=5, random_state=seed, n_jobs=-1
    )
    return _to_importance_df(result.importances_mean, feature_names)


def _collapse_shap_values(shap_values) -> np.ndarray:
    """Reduce SHAP output to a 2D (n_samples, n_features) array for the positive class.

    SHAP returns different shapes depending on version and explainer:
    - older API: list of (n_samples, n_features) arrays, one per class
    - newer API: 3D ndarray of shape (n_samples, n_features, n_classes)
    - regression / single-class: 2D ndarray
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[..., -1]
    return arr


def _shap_tree(model, X_sample, feature_names) -> pd.DataFrame:
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    arr = _collapse_shap_values(shap_values)
    mean_abs = np.abs(arr).mean(axis=0)
    return _to_importance_df(mean_abs, feature_names)


def _shap_kernel(model, X_train, X_eval, feature_names, sample_size, seed) -> pd.DataFrame:
    import shap

    background = _subsample(X_train, min(100, sample_size), seed)
    explained = _subsample(X_eval, sample_size, seed)
    predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(explained, nsamples="auto", silent=True)
    arr = _collapse_shap_values(shap_values)
    mean_abs = np.abs(arr).mean(axis=0)
    return _to_importance_df(mean_abs, feature_names)


def interpret_decision_tree(
    model,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_eval,
    methods: Iterable[str] = SUPPORTED_METHODS,
    shap_sample_size: int = 500,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    methods = _normalize_methods(methods)
    feature_names = list(X_train.columns)
    out: dict[str, pd.DataFrame] = {}

    if "native" in methods:
        out["native"] = _to_importance_df(model.feature_importances_, feature_names)
    if "permutation" in methods:
        out["permutation"] = _permutation(model, X_eval, y_eval, feature_names, seed)
    if "shap" in methods:
        sample = _subsample(X_eval, shap_sample_size, seed)
        out["shap"] = _shap_tree(model, sample, feature_names)
    return out


def interpret_xgboost(
    model,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_eval,
    methods: Iterable[str] = SUPPORTED_METHODS,
    shap_sample_size: int = 500,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    methods = _normalize_methods(methods)
    feature_names = list(X_train.columns)
    out: dict[str, pd.DataFrame] = {}

    if "native" in methods:
        out["native"] = _to_importance_df(model.feature_importances_, feature_names)
    if "permutation" in methods:
        out["permutation"] = _permutation(model, X_eval, y_eval, feature_names, seed)
    if "shap" in methods:
        sample = _subsample(X_eval, shap_sample_size, seed)
        out["shap"] = _shap_tree(model, sample, feature_names)
    return out


def interpret_mlp(
    model,
    X_train: pd.DataFrame,
    X_eval: pd.DataFrame,
    y_eval,
    methods: Iterable[str] = SUPPORTED_METHODS,
    shap_sample_size: int = 500,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    methods = _normalize_methods(methods)
    feature_names = list(X_train.columns)
    out: dict[str, pd.DataFrame] = {}

    if "native" in methods:
        warnings.warn(
            "MLP has no native feature_importances_; skipping 'native' method.",
            stacklevel=2,
        )
    if "permutation" in methods:
        out["permutation"] = _permutation(model, X_eval, y_eval, feature_names, seed)
    if "shap" in methods:
        out["shap"] = _shap_kernel(
            model, X_train, X_eval, feature_names, shap_sample_size, seed
        )
    return out
