from typing import Tuple, Dict, Any, Optional
import numpy as np
import scipy.optimize as opt
from sklearn.calibration import calibration_curve

from src.utils.logger import get_logger

logger = get_logger("calibration")


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE).

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted positive class probabilities in [0, 1].
        n_bins: Number of equal-width probability bins.

    Returns:
        Scalar ECE value in [0, 1].
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_weight = np.sum(mask) / total_samples
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_reliability_diagram(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """Computes points for plotting a reliability curve.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins.

    Returns:
        Dictionary with 'prob_true', 'prob_pred', and 'ece'.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "ece": ece,
    }


class TemperatureScaler:
    """Post-hoc temperature scaling calibration for deep learning logits."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        """Optimizes temperature T to minimize cross-entropy loss on validation logits."""
        def loss_fn(t: float) -> float:
            scaled = logits / max(t, 1e-4)
            # Binary cross entropy with numerical stability
            p = 1.0 / (1.0 + np.exp(-scaled))
            p = np.clip(p, 1e-7, 1.0 - 1e-7)
            bce = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
            return bce

        res = opt.minimize(loss_fn, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        logger.info(f"Optimal calibration temperature found: {self.temperature:.4f}")
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Applies learned temperature to scale logits and returns calibrated probabilities."""
        scaled = logits / self.temperature
        return 1.0 / (1.0 + np.exp(-scaled))
