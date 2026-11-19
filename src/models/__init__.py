"""Models module containing baseline (Logistic Regression, Linear SVM, Random Forest, Naive Bayes) and neural (LSTM, BiLSTM) architectures."""

from src.models.baseline_models import (
    LogisticRegressionModel,
    LinearSVMModel,
    RandomForestModel,
    NaiveBayesModel,
    MultinomialNBModel,
)

try:
    from src.models.recurrent_models import RecurrentClassifier
    from src.models.trainer import RecurrentModelTrainer
except ImportError:
    RecurrentClassifier = None  # type: ignore
    RecurrentModelTrainer = None  # type: ignore

__all__ = [
    "LogisticRegressionModel",
    "LinearSVMModel",
    "RandomForestModel",
    "NaiveBayesModel",
    "MultinomialNBModel",
    "RecurrentClassifier",
    "RecurrentModelTrainer",
]
