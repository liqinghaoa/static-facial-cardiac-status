"""Metrics for direct-label R3DPR binary classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def compute_binary_metrics(y_true: Any, y_prob: Any) -> dict[str, Any]:
    """Compute protocol metrics using Patient probability and argmax decisions."""
    true = np.asarray(y_true, dtype=np.int64)
    probability = np.asarray(y_prob, dtype=np.float64)
    if true.ndim != 1 or len(true) == 0 or not np.isin(true, [0, 1]).all():
        raise ValueError("y_true must be a non-empty vector of binary labels")
    if probability.shape != (len(true), 2):
        raise ValueError(f"y_prob must have shape [N, 2], got {probability.shape}")
    if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to one")
    if len(np.unique(true)) != 2:
        raise ValueError("both classes are required for binary ROC-AUC")
    prediction = probability.argmax(axis=1)
    matrix = confusion_matrix(true, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "macro_auc": float(roc_auc_score(true, probability[:, 1])),
        "accuracy": float(accuracy_score(true, prediction)),
        "macro_precision": float(precision_score(true, prediction, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(true, prediction, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(true, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "confusion_matrix": matrix,
    }


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items() if np.isscalar(value)}
