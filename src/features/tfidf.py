from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.logger import get_logger

logger = get_logger("tfidf_extractor")


class TFIDFExtractor:
    """Encapsulates TF-IDF vectorization configuration, fitting, transformation, and serialization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config.get("features", {}).get("tfidf", {}) if config else {}
        ngram_range = tuple(cfg.get("ngram_range", (1, 2)))
        max_features = cfg.get("max_features", 10000)
        min_df = cfg.get("min_df", 2)
        max_df = cfg.get("max_df", 0.95)
        sublinear_tf = cfg.get("sublinear_tf", True)
        norm = cfg.get("norm", "l2")

        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            norm=norm,
        )
        self.is_fitted = False

    def fit_transform(self, raw_texts: Any) -> spmatrix:
        """Fits vectorizer on raw_texts and returns feature matrix."""
        logger.info("Fitting TF-IDF vectorizer...")
        X = self.vectorizer.fit_transform(raw_texts)
        self.is_fitted = True
        logger.info(f"TF-IDF matrix built with shape {X.shape}")
        return X

    def transform(self, raw_texts: Any) -> spmatrix:
        """Transforms raw_texts using fitted vectorizer."""
        if not self.is_fitted:
            raise RuntimeError("TFIDFExtractor is not fitted yet. Call fit_transform first.")
        return self.vectorizer.transform(raw_texts)

    def save(self, filepath: str) -> None:
        """Saves fitted vectorizer to disk."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, filepath)
        logger.info(f"TF-IDF vectorizer saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Loads fitted vectorizer from disk."""
        self.vectorizer = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"TF-IDF vectorizer loaded from {filepath}")
