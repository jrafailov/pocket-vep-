from .interpret import (
    interpret_decision_tree,
    interpret_mlp,
    interpret_xgboost,
)
from .metrics import compute_metrics, format_report

INTERPRET_REGISTRY = {
    "decision_tree": interpret_decision_tree,
    "xgboost": interpret_xgboost,
    "mlp": interpret_mlp,
}


def interpret_model(model_name: str, model, X_train, X_eval, y_eval, methods, **kwargs):
    if model_name not in INTERPRET_REGISTRY:
        raise KeyError(
            f"No interpreter for model '{model_name}'. "
            f"Available: {sorted(INTERPRET_REGISTRY)}"
        )
    return INTERPRET_REGISTRY[model_name](
        model, X_train, X_eval, y_eval, methods=methods, **kwargs
    )


__all__ = [
    "compute_metrics",
    "format_report",
    "INTERPRET_REGISTRY",
    "interpret_model",
    "interpret_decision_tree",
    "interpret_xgboost",
    "interpret_mlp",
]
