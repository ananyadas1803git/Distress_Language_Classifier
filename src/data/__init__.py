"""Data loading, cleaning, anonymization, and PyTorch Dataset structures."""

from src.data.preprocess import clean_text, load_and_preprocess_dataset

try:
    from src.data.dataset import Vocabulary, TextClassificationDataset
except ImportError:
    Vocabulary = None  # type: ignore
    TextClassificationDataset = None  # type: ignore

__all__ = [
    "clean_text",
    "load_and_preprocess_dataset",
    "Vocabulary",
    "TextClassificationDataset",
]
