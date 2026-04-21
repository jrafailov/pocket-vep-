from __future__ import annotations

from typing import Iterable, Literal

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from ..eval.metrics import compute_metrics
from ..features import build_feature_matrix
from . import get_model


class Trainer:
    """In-Python wrapper that behaves like a CLI: pick features + model, run one experiment."""

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_col: str = "ML_Label",
    ) -> None:
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_col = label_col

    def _split_df(self, eval_on: str) -> pd.DataFrame:
        if eval_on == "val":
            return self.val_df
        if eval_on == "test":
            return self.test_df
        raise ValueError(f"eval_on must be 'val' or 'test', got {eval_on!r}")

    def run(
        self,
        feature_keys: Iterable[str],
        model_name: str,
        eval_on: Literal["val", "test"] = "val",
    ) -> dict:
        feature_keys = list(feature_keys)

        X_tr = build_feature_matrix(self.train_df, feature_keys)
        eval_df = self._split_df(eval_on)
        X_ev = build_feature_matrix(eval_df, feature_keys)

        # align columns in case one-hot categories differ across splits
        X_ev = X_ev.reindex(columns=X_tr.columns, fill_value=0)

        y_tr = self.train_df.loc[X_tr.index, self.label_col]
        y_ev = eval_df.loc[X_ev.index, self.label_col]

        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_tr)
        y_ev_enc = le.transform(y_ev)

        pos = int((y_tr_enc == 1).sum())
        neg = int((y_tr_enc == 0).sum())
        spw = neg / max(pos, 1)

        model = get_model(model_name, scale_pos_weight=spw)

        if model_name == "mlp":
            sw = compute_sample_weight("balanced", y_tr_enc)
            model.fit(X_tr, y_tr_enc)  # MLP fit signature varies by version; keep baseline simple
            # Note: to apply sample weighting to MLP, wrap via BaggingClassifier or use a
            # custom loop. Skipped here to keep the baseline close to the notebook.
            del sw
        else:
            model.fit(X_tr, y_tr_enc)

        preds = model.predict(X_ev)
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_ev)[:, 1]
            except Exception:
                proba = None

        return compute_metrics(
            y_true=y_ev_enc,
            y_pred=preds,
            y_proba=proba,
            label_names=list(le.classes_),
            feature_keys=feature_keys,
            model_name=model_name,
            split=eval_on,
        )
