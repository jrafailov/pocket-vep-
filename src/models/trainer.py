from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from ..eval.metrics import compute_metrics
from ..features import build_feature_matrix, select_features
from . import get_model


class Trainer:
    """In-Python wrapper that behaves like a CLI: pick features + model, run one experiment.

    Two construction modes:

    * Materialized matrix (preferred): pass `schema` from load_feature_matrix().
      train_df/val_df/test_df already contain every feature column from every
      block; run() just slices the columns belonging to the requested blocks.
      Every feature-set variant trains on the same row set.

    * Legacy / on-the-fly: pass `schema=None`. run() falls back to
      build_feature_matrix(), which inner-joins per-experiment. Row counts
      will differ across feature sets (sequence > evolution > structure ~
      union), so cross-arm metric deltas in this mode are confounded with
      row-budget differences.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        label_col: str = "ML_Label",
        schema: dict | None = None,
    ) -> None:
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.label_col = label_col
        self.schema = schema

    def run(
        self,
        feature_keys: Iterable[str],
        model_name: str,
        seed: int = 42,
    ) -> dict:
        feature_keys = list(feature_keys)

        if self.schema is not None:
            X_tr = select_features(self.train_df, feature_keys, self.schema)
            X_val = select_features(self.val_df, feature_keys, self.schema)
            X_test = select_features(self.test_df, feature_keys, self.schema)
        else:
            X_tr = build_feature_matrix(self.train_df, feature_keys)
            X_val = build_feature_matrix(self.val_df, feature_keys).reindex(
                columns=X_tr.columns, fill_value=0
            )
            X_test = build_feature_matrix(self.test_df, feature_keys).reindex(
                columns=X_tr.columns, fill_value=0
            )

        y_tr = self.train_df.loc[X_tr.index, self.label_col]
        y_val = self.val_df.loc[X_val.index, self.label_col]
        y_test = self.test_df.loc[X_test.index, self.label_col]

        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_tr)
        y_val_enc = le.transform(y_val)
        y_test_enc = le.transform(y_test)
        print("Label encoder: ", le.classes_)

        pos = int((y_tr_enc == 1).sum())
        neg = int((y_tr_enc == 0).sum())
        spw = neg / max(pos, 1)

        model = get_model(model_name, seed=seed, scale_pos_weight=spw)

        if model_name == "mlp":
            sw = compute_sample_weight("balanced", y_tr_enc)
            model.fit(X_tr, y_tr_enc)  # MLP fit signature varies by version; keep baseline simple
            # Note: to apply sample weighting to MLP, wrap via BaggingClassifier or use a
            # custom loop. Skipped here to keep the baseline close to the notebook.
            del sw
        else:
            model.fit(X_tr, y_tr_enc)

        metrics = [
            self._evaluate(model, X_val, y_val_enc, le.classes_, feature_keys, model_name, "val"),
            self._evaluate(model, X_test, y_test_enc, le.classes_, feature_keys, model_name, "test"),
        ]

        return {
            "metrics": metrics,
            "model": model,
            "feature_names": list(X_tr.columns),
            "X_train": X_tr,
            "X_val": X_val,
            "X_test": X_test,
            "y_val_enc": y_val_enc,
            "y_test_enc": y_test_enc,
            "label_names": list(le.classes_),
        }

    @staticmethod
    def _evaluate(model, X, y_enc, label_names, feature_keys, model_name, split):
        preds = model.predict(X)
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)[:, 1]
            except Exception:
                proba = None
        return compute_metrics(
            y_true=y_enc,
            y_pred=preds,
            y_proba=proba,
            label_names=list(label_names),
            feature_keys=feature_keys,
            model_name=model_name,
            split=split,
        )
