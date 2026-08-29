from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score


def binary_metrics(y_true: list[float] | np.ndarray, y_prob: list[float] | np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int64)

    result: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }

    result["roc_auc"] = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    return result
