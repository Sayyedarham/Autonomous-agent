"""Evaluation utilities for labelled telemetry only; never infer labels on unlabelled data."""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_predictions(labels: list[int] | np.ndarray, predictions: list[int] | np.ndarray, scores: list[float] | np.ndarray | None = None) -> dict[str, Any]:
    truth, predicted = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("labels and predictions must be one-dimensional arrays of equal length.")
    if not set(np.unique(truth)).issubset({0, 1}) or not set(np.unique(predicted)).issubset({0, 1}):
        raise ValueError("labels and predictions must contain only 0 and 1.")
    matrix = confusion_matrix(truth, predicted, labels=[0, 1])
    result: dict[str, Any] = {"precision": float(precision_score(truth, predicted, zero_division=0)), "recall": float(recall_score(truth, predicted, zero_division=0)), "f1": float(f1_score(truth, predicted, zero_division=0)), "confusion_matrix": matrix.tolist(), "false_positives": int(matrix[0, 1]), "false_negatives": int(matrix[1, 0])}
    if scores is not None:
        numeric_scores = np.asarray(scores, dtype=float)
        if numeric_scores.shape != truth.shape:
            raise ValueError("scores must have the same length as labels.")
        result["pr_auc"] = float(average_precision_score(truth, numeric_scores)) if truth.any() else None
    return result
