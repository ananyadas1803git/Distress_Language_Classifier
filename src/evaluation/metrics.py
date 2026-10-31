from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)

from src.uncertainty.calibration import compute_expected_calibration_error


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Computes comprehensive classification and uncertainty metrics.

    Args:
        y_true: True binary labels (0 or 1).
        y_prob: Predicted probabilities for class 1.
        threshold: Decision threshold for classification.
        n_bins: Number of bins for ECE calculation.

    Returns:
        Dict of metric names to values.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "ece": float(compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    return metrics


def evaluate_all_models(
    results_dict: Dict[str, Dict[str, np.ndarray]],
    y_test: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Generates comparison summary table across all models.

    Args:
        results_dict: Mapping model_name -> {'y_prob': np.ndarray}.
        y_test: Ground truth test labels.
        n_bins: Bins for ECE.

    Returns:
        DataFrame summarizing metrics across all models.
    """
    rows = []
    for model_name, preds in results_dict.items():
        y_prob = preds["y_prob"]
        m = compute_classification_metrics(y_test, y_prob, n_bins=n_bins)
        m["model"] = model_name
        rows.append(m)

    df_metrics = pd.DataFrame(rows)
    # Order columns nicely
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier_score", "ece"]
    return df_metrics[[c for c in cols if c in df_metrics.columns]]
