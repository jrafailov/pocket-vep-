from __future__ import annotations

from typing import Callable

from .classifiers import decision_tree, mlp, xgboost

MODEL_REGISTRY: dict[str, Callable] = {
    "decision_tree": decision_tree,
    "mlp": mlp,
    "xgboost": xgboost,
}


def get_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


__all__ = ["MODEL_REGISTRY", "get_model", "decision_tree", "mlp", "xgboost"]
