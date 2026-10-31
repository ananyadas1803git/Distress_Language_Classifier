"""SHAP-based explanation and interpretability module with safe, non-diagnostic framing."""

from src.explainability.shap_explainer import (
    LinearSHAPExplainer,
    BlackBoxSHAPExplainer,
    format_responsible_explanation,
)

__all__ = [
    "LinearSHAPExplainer",
    "BlackBoxSHAPExplainer",
    "format_responsible_explanation",
]
