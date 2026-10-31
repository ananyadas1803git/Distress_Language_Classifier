import re
import html
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger("data_preprocess")


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_urls: bool = True,
    remove_handles: bool = True,
    strip_html: bool = True,
    remove_extra_whitespace: bool = True,
) -> str:
    """Cleans and anonymizes online post text.

    Args:
        text: Raw text string.
        lowercase: Whether to convert text to lowercase.
        remove_urls: Whether to strip web URLs.
        remove_handles: Whether to strip usernames/subreddits.
        strip_html: Whether to unescape and strip HTML tags.
        remove_extra_whitespace: Whether to collapse multiple whitespaces.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    if strip_html:
        text = html.unescape(text)
        text = re.sub(r"<[^>]+>", " ", text)

    if remove_urls:
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    if remove_handles:
        # Mask or remove Reddit handles/user tags to preserve anonymity
        text = re.sub(r"(?:^|(?<=\s))(?:/?r/|/?u/|@)[\w\d_-]+", " ", text)

    if lowercase:
        text = text.lower()

    if remove_extra_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    return text


def load_and_preprocess_dataset(
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw dataset, applies preprocessing/anonymization, and creates train/val/test splits.

    Args:
        config: Configuration dictionary loaded from config.yaml.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    raw_path = config["data"]["raw_path"]
    logger.info(f"Loading raw dataset from {raw_path}")
    df = pd.read_csv(raw_path)

    text_col = config["data"].get("text_column", "text")
    label_col = config["data"].get("label_column", "label")

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Expected columns '{text_col}' and '{label_col}' in dataset, but found {df.columns.tolist()}"
        )

    # Drop nulls in text or label
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[label_col] = df[label_col].astype(int)

    prep_cfg = config.get("preprocessing", {})
    logger.info("Cleaning and anonymizing text...")
    df["cleaned_text"] = df[text_col].apply(
        lambda t: clean_text(
            t,
            lowercase=prep_cfg.get("lowercase", True),
            remove_urls=prep_cfg.get("remove_urls", True),
            remove_handles=prep_cfg.get("remove_handles", True),
            strip_html=prep_cfg.get("strip_html", True),
            remove_extra_whitespace=prep_cfg.get("remove_extra_whitespace", True),
        )
    )

    # Filter out empty texts after cleaning
    min_len = prep_cfg.get("min_token_length", 2)
    df = df[df["cleaned_text"].str.split().str.len() >= min_len].copy()

    random_seed = config.get("project", {}).get("random_seed", 42)

    # If dataset already has a 'split' column (e.g. Dreaddit train/test)
    if "split" in df.columns and "test" in df["split"].unique():
        logger.info("Found existing 'split' column in dataset.")
        train_val_df = df[df["split"] == "train"].copy()
        test_df = df[df["split"] == "test"].copy()

        # Split train into train and validation
        val_size = config["data"].get("val_size", 0.1)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=random_seed,
            stratify=train_val_df[label_col] if config["data"].get("stratify", True) else None,
        )
    else:
        logger.info("Performing stratified train/val/test split from scratch.")
        test_size = config["data"].get("test_size", 0.2)
        val_size = config["data"].get("val_size", 0.1)

        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed,
            stratify=df[label_col] if config["data"].get("stratify", True) else None,
        )

        val_ratio_adjusted = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio_adjusted,
            random_state=random_seed,
            stratify=train_val_df[label_col] if config["data"].get("stratify", True) else None,
        )

    processed_dir = Path(config["data"].get("processed_dir", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(
        f"Preprocessing complete. Splits saved: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )
    return train_df, val_df, test_df
