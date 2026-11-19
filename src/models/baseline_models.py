from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV

from src.utils.logger import get_logger

logger = get_logger("baseline_models")


class LogisticRegressionModel:
    """Algorithm 1: TF-IDF + Logistic Regression."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config.get("models", {}).get("logistic_regression", {}) if config else {}
        penalty = cfg.get("penalty", "l2")
        lr_kwargs = {
            "C": cfg.get("C", 1.0),
            "solver": cfg.get("solver", "lbfgs"),
            "max_iter": cfg.get("max_iter", 1000),
            "class_weight": cfg.get("class_weight", "balanced"),
            "random_state": cfg.get("random_state", 42),
        }
        if penalty != "l2":
            lr_kwargs["penalty"] = penalty

        self.model = LogisticRegression(**lr_kwargs)
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


class RandomForestModel:
    """Algorithm 3: TF-IDF + Random Forest."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config.get("models", {}).get("random_forest", {}) if config else {}
        self.model = RandomForestClassifier(
            n_estimators=cfg.get("n_estimators", 100),
            max_depth=cfg.get("max_depth", None),
            min_samples_split=cfg.get("min_samples_split", 2),
            min_samples_leaf=cfg.get("min_samples_leaf", 1),
            class_weight=cfg.get("class_weight", "balanced"),
            random_state=cfg.get("random_state", 42),
            n_jobs=cfg.get("n_jobs", -1),
        )
        self.is_fitted = False

    def fit(self, X: Any, y: Any) -> "RandomForestModel":
        """Fits Random Forest classifier."""
        logger.info("Training Random Forest model...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Random Forest training complete.")
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
        logger.info(f"Random Forest saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Loads serialized model from disk."""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Random Forest loaded from {filepath}")


class NaiveBayesModel:
    """Algorithm 4: TF-IDF + Multinomial Naive Bayes."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, calibrate: bool = False):
        cfg = config.get("models", {}).get("naive_bayes", {}) if config else {}
        base_nb = MultinomialNB(
            alpha=cfg.get("alpha", 1.0),
            fit_prior=cfg.get("fit_prior", True),
        )
        self.calibrate = cfg.get("calibrate", calibrate)
        if self.calibrate:
            self.model = CalibratedClassifierCV(estimator=base_nb, cv=5)
        else:
            self.model = base_nb
        self.is_fitted = False

    def fit(self, X: Any, y: Any) -> "NaiveBayesModel":
        """Fits Multinomial Naive Bayes classifier."""
        logger.info(f"Training Naive Bayes model (calibrated={self.calibrate})...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Naive Bayes training complete.")
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
        logger.info(f"Naive Bayes saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Loads serialized model from disk."""
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Naive Bayes loaded from {filepath}")


# Convenient alias
MultinomialNBModel = NaiveBayesModel
