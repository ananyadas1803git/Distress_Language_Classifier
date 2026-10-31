from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("error_analysis")


def categorize_prediction(true_label: int, pred_label: int, is_abstained: bool = False) -> str:
    """Categorizes prediction outcome with uncertainty awareness."""
    if is_abstained or pred_label == -1:
        return "Abstained (Uncertain)"
    if true_label == 1 and pred_label == 1:
        return "True Positive"
    if true_label == 0 and pred_label == 0:
        return "True Negative"
    if true_label == 0 and pred_label == 1:
        return "False Positive (False Alarm / Stigma Risk)"
    if true_label == 1 and pred_label == 0:
        return "False Negative (Missed Distress Signal)"
    return "Unknown"


def perform_error_analysis(
    texts: List[str],
    y_true: np.ndarray,
    y_prob: np.ndarray,
    abstained_mask: Optional[np.ndarray] = None,
    output_csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """Conducts rigorous error analysis on model predictions.

    Args:
        texts: Raw or cleaned post texts.
        y_true: Ground truth binary labels.
        y_prob: Calibrated probabilities of positive class.
        abstained_mask: Optional boolean array marking abstained instances.
        output_csv_path: Optional path to save detailed error breakdown.

    Returns:
        DataFrame with error categories and diagnostic properties.
    """
    logger.info("Performing detailed error analysis...")
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    std_preds = (y_prob >= 0.5).astype(int)
    if abstained_mask is None:
        abstained_mask = np.zeros(len(y_true), dtype=bool)

    categories = [
        categorize_prediction(t, p, a)
        for t, p, a in zip(y_true, std_preds, abstained_mask)
    ]

    confidences = np.maximum(y_prob, 1.0 - y_prob)
    text_lengths = [len(str(t).split()) for t in texts]

    df_errors = pd.DataFrame({
        "text": texts,
        "true_label": y_true,
        "predicted_prob": y_prob.round(4),
        "confidence": confidences.round(4),
        "standard_pred": std_preds,
        "abstained": abstained_mask,
        "error_category": categories,
        "word_count": text_lengths,
    })

    summary = df_errors["error_category"].value_counts()
    logger.info("Error Category Distribution:\n" + str(summary))

    if output_csv_path:
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_errors.to_csv(output_csv_path, index=False)
        logger.info(f"Detailed error audit saved to {output_csv_path}")

    return df_errors
