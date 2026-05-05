from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def random_forest(seed: int = 42, **_: object):
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=1,
        random_state=seed,
    )


def mlp(seed: int = 42, **_: object):
    # sklearn MLP doesn't support class_weight; Trainer passes sample_weight at fit time.
    # StandardScaler is fit on train only via Pipeline semantics — required because
    # input features span 0/1 one-hots, 0–100 (pLDDT), and unbounded floats.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16), 
                    max_iter=1000, 
                    early_stopping=True, 
                    n_iter_no_change=20, 
                    random_state=seed
                ),
            ),
        ]
    )


def xgboost(seed: int = 42, scale_pos_weight: float = 1.0, **_: object):
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
        eval_metric="logloss",
    )
