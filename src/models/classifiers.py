from __future__ import annotations

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def decision_tree(class_weight: str | dict | None = "balanced", **_: object):
    return DecisionTreeClassifier(
        max_depth=5, class_weight=class_weight, random_state=42
    )


def mlp(**_: object):
    # sklearn MLP doesn't support class_weight; Trainer passes sample_weight at fit time.
    return MLPClassifier(
        hidden_layer_sizes=(32, 16), max_iter=500, random_state=42
    )


def xgboost(scale_pos_weight: float = 1.0, **_: object):
    return XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
