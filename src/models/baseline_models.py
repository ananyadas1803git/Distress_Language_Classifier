from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from src.utils.logger import get_logger

logger = get_logger("baseline_models")


class LogisticRegressionModel:
    """Algorithm 1: TF-IDF + Logistic Regression."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config.get("models", {}).get("logistic_regression", {}) if config else {}
        self.model = LogisticRegression(
            penalty=cfg.get("penalty", "l2"),
            C=cfg.get("C", 1.0),
            solver=cfg.get("solver", "lbfgs"),
            max_iter=cfg.get("max_iter", 1000),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=cfg.get("random_state", 42),
        )
        self.is_fitted = False

    def fit(self, X: Any, y: Any) -> "LogisticRegressionModel":
        """Fits Logistic Regression classifier."""
        logger.info("Training Logistic Regression model...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Logistic Regression training complete.")
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predicts class labels (0: No distress language, 1: Distress-related language)."""
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Returns class probabilities."""
        return self.model.predict_proba(X)

    def save(self, filepath: str) -> None:
        """Serializes model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Logistic Regression saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Loads serialized model from disk."""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Logistic Regression loaded from {filepath}")


class LinearSVMModel:
    """Algorithm 2: TF-IDF + Linear SVM (with calibrated probability output)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, calibrate: bool = True):
        cfg = config.get("models", {}).get("linear_svm", {}) if config else {}
        base_svm = LinearSVC(
            C=cfg.get("C", 1.0),
            loss=cfg.get("loss", "squared_hinge"),
            max_iter=cfg.get("max_iter", 2000),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=cfg.get("random_state", 42),
        )
        self.calibrate = calibrate
        if calibrate:
            # Calibrate SVM output to yield valid probabilities for uncertainty handling
            self.model = CalibratedClassifierCV(estimator=base_svm, cv=5)
        else:
            self.model = base_svm
        self.is_fitted = False

    def fit(self, X: Any, y: Any) -> "LinearSVMModel":
        """Fits Linear SVM classifier."""
        logger.info(f"Training Linear SVM model (calibrated={self.calibrate})...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Linear SVM training complete.")
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predicts class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Returns class probabilities (requires calibration=True)."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # Fallback: sigmoid of decision function
            df = self.model.decision_function(X)
            probs = 1.0 / (1.0 + np.exp(-df))
            return np.vstack([1.0 - probs, probs]).T

    def save(self, filepath: str) -> None:
        """Serializes model to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Linear SVM saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Loads serialized model from disk."""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Linear SVM loaded from {filepath}")
