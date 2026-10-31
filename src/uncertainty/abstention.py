from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.utils.logger import get_logger

logger = get_logger("abstention")


class ConfidenceAbstainer:
    """Confidence-based abstention mechanism to prevent false overclaiming."""

    def __init__(self, confidence_threshold: float = 0.70, strategy: str = "max_probability"):
        """Args:
            confidence_threshold: Minimum confidence required to issue a prediction.
                                  Instances with confidence below this threshold are abstained.
            strategy: 'max_probability' (max(p, 1-p)), 'margin' (|p - 0.5| * 2), or 'entropy'.
        """
        self.confidence_threshold = confidence_threshold
        self.strategy = strategy

    def compute_confidence(self, probs: np.ndarray) -> np.ndarray:
        """Computes certainty metric for binary probabilities in [0, 1]."""
        probs = np.asarray(probs)
        if probs.ndim == 2:
            probs = probs[:, 1]

        if self.strategy == "max_probability":
            return np.maximum(probs, 1.0 - probs)
        elif self.strategy == "margin":
            return 2.0 * np.abs(probs - 0.5)
        elif self.strategy == "entropy":
            eps = 1e-7
            p = np.clip(probs, eps, 1.0 - eps)
            ent = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
            return 1.0 - ent  # normalized certainty
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def predict_with_abstention(
        self, probs: np.ndarray, threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Classifies instances or flags them as abstained (-1).

        Args:
            probs: Predicted probabilities of class 1.
            threshold: Decision threshold for positive class.

        Returns:
            Tuple of:
              - final_predictions: array where 0 = Non-distress, 1 = Distress, -1 = Abstained
              - abstained_mask: boolean array (True if abstained)
              - confidences: certainty score per instance
        """
        confidences = self.compute_confidence(probs)
        probs_1d = probs[:, 1] if probs.ndim == 2 else probs

        standard_preds = (probs_1d >= threshold).astype(int)
        abstained_mask = confidences < self.confidence_threshold

        final_preds = standard_preds.copy()
        final_preds[abstained_mask] = -1

        return final_preds, abstained_mask, confidences


def evaluate_coverage_vs_accuracy(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold_sweep: List[float],
) -> pd.DataFrame:
    """Evaluates accuracy, F1, and selective risk across different abstention thresholds.

    Args:
        y_true: True binary labels.
        probs: Positive class probabilities.
        threshold_sweep: List of confidence thresholds to test.

    Returns:
        DataFrame containing threshold, coverage, accuracy, f1, and abstention_rate.
    """
    records = []
    y_true = np.asarray(y_true)
    probs_1d = probs[:, 1] if probs.ndim == 2 else probs

    for thresh in threshold_sweep:
        abstainer = ConfidenceAbstainer(confidence_threshold=thresh)
        final_preds, abstained_mask, confs = abstainer.predict_with_abstention(probs_1d)

        retained_mask = ~abstained_mask
        coverage = np.mean(retained_mask)
        abstention_rate = np.mean(abstained_mask)

        if np.sum(retained_mask) > 0:
            sel_acc = accuracy_score(y_true[retained_mask], final_preds[retained_mask])
            sel_f1 = f1_score(y_true[retained_mask], final_preds[retained_mask], zero_division=0)
            sel_risk = 1.0 - sel_acc
        else:
            sel_acc = np.nan
            sel_f1 = np.nan
            sel_risk = np.nan

        records.append({
            "confidence_threshold": thresh,
            "coverage": coverage,
            "abstention_rate": abstention_rate,
            "retained_count": int(np.sum(retained_mask)),
            "abstained_count": int(np.sum(abstained_mask)),
            "selective_accuracy": sel_acc,
            "selective_f1": sel_f1,
            "selective_risk": sel_risk,
        })

    return pd.DataFrame(records)
