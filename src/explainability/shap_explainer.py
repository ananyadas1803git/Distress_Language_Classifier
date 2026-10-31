from typing import List, Dict, Any, Optional
import numpy as np
import shap

from src.utils.logger import get_logger

logger = get_logger("shap_explainer")

ETHICAL_DISCLAIMER = (
    "RESPONSIBLE AI & ETHICAL NOTICE:\n"
    "Highlighted linguistic cues indicate statistical token associations with distress-related vocabulary "
    "in anonymized social media text. These attributions DO NOT represent clinical evidence, psychological "
    "assessment, or depression diagnosis."
)


def format_responsible_explanation(
    text: str,
    top_tokens: List[Dict[str, Any]],
    probability: float,
    abstained: bool = False,
) -> Dict[str, Any]:
    """Wraps token-level attributions in a safe, uncertainty-aware format.

    Args:
        text: Input post text.
        top_tokens: List of dicts with 'token', 'shap_value', and 'direction'.
        probability: Calibrated distress probability.
        abstained: Whether the model abstained due to uncertainty.

    Returns:
        Structured explanation dict.
    """
    status = "Uncertain / Abstained from decision" if abstained else (
        "Distress-related language detected" if probability >= 0.5 else "No significant distress language detected"
    )

    return {
        "text_preview": text[:200] + ("..." if len(text) > 200 else ""),
        "calibrated_distress_probability": round(float(probability), 4),
        "status": status,
        "abstained": abstained,
        "top_contributing_tokens": top_tokens,
        "disclaimer": ETHICAL_DISCLAIMER,
    }


class LinearSHAPExplainer:
    """SHAP explainer for linear models (Logistic Regression, Linear SVM) using TF-IDF features."""

    def __init__(self, model: Any, background_data: Any, feature_names: List[str]):
        """Args:
            model: Fitted scikit-learn linear estimator.
            background_data: Sample of training TF-IDF matrix.
            feature_names: List of TF-IDF vocabulary terms.
        """
        self.model = model
        self.feature_names = np.array(feature_names)
        logger.info("Initializing SHAP LinearExplainer...")
        self.explainer = shap.LinearExplainer(model, background_data)

    def explain_instance(self, sample_tfidf: Any, top_k: int = 10) -> List[Dict[str, Any]]:
        """Computes top-k token contributions for a single TF-IDF sample."""
        shap_values = self.explainer.shap_values(sample_tfidf)
        if isinstance(shap_values, list):
            vals = shap_values[1]  # positive class
        elif shap_values.ndim == 2:
            vals = shap_values[0]
        else:
            vals = shap_values

        non_zero_indices = sample_tfidf.nonzero()[1]
        token_attributions = []

        for idx in non_zero_indices:
            token = self.feature_names[idx]
            val = float(vals[idx])
            token_attributions.append({
                "token": token,
                "shap_value": round(val, 5),
                "direction": "supports_distress" if val > 0 else "supports_neutral",
            })

        token_attributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return token_attributions[:top_k]


class BlackBoxSHAPExplainer:
    """Model-agnostic SHAP explainer for complex models (LSTM, BiLSTM)."""

    def __init__(self, predict_fn: Any, background_samples: List[str]):
        """Args:
            predict_fn: Callable taking list of raw texts and returning probabilities.
            background_samples: Reference text samples for baseline distribution.
        """
        self.predict_fn = predict_fn
        self.background_samples = background_samples
        logger.info("Initializing text-based SHAP Explainer...")
        # Uses shap.Explainer with standard text masker
        masker = shap.maskers.Text(tokenizer=r"\W+")
        self.explainer = shap.Explainer(predict_fn, masker=masker)

    def explain_text(self, text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Computes phrase/word attribution for a given post."""
        shap_values = self.explainer([text])
        values = shap_values.values[0]
        tokens = shap_values.data[0]

        attributions = []
        for token, val in zip(tokens, values):
            token_clean = token.strip()
            if token_clean:
                attributions.append({
                    "token": token_clean,
                    "shap_value": round(float(val), 5),
                    "direction": "supports_distress" if val > 0 else "supports_neutral",
                })

        attributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return attributions[:top_k]
