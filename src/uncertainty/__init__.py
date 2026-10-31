"""Uncertainty quantification, probability calibration, and confidence-based abstention."""

from src.uncertainty.calibration import (
    compute_expected_calibration_error,
    compute_reliability_diagram,
    TemperatureScaler,
)
from src.uncertainty.abstention import (
    ConfidenceAbstainer,
    evaluate_coverage_vs_accuracy,
)

__all__ = [
    "compute_expected_calibration_error",
    "compute_reliability_diagram",
    "TemperatureScaler",
    "ConfidenceAbstainer",
    "evaluate_coverage_vs_accuracy",
]
