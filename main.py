import argparse
import json
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.data.preprocess import load_and_preprocess_dataset
from src.features.tfidf import TFIDFExtractor
from src.models.baseline_models import (
    LogisticRegressionModel,
    LinearSVMModel,
    RandomForestModel,
    NaiveBayesModel,
)
from src.evaluation.metrics import evaluate_all_models
from src.uncertainty.abstention import ConfidenceAbstainer, evaluate_coverage_vs_accuracy
from src.evaluation.error_analysis import perform_error_analysis

logger = get_logger("main")


def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Loads YAML configuration file with standard PyYAML or fallback parser."""
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        root: dict = {}
        stack = [(0, root)]
        with open(config_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.split("#")[0].rstrip()
                if not line.strip():
                    continue
                indent = len(line) - len(line.lstrip())
                content = line.strip()
                while len(stack) > 1 and indent <= stack[-1][0]:
                    stack.pop()
                current_dict = stack[-1][1]
                if ":" in content:
                    key, val = content.split(":", 1)
                    key = key.strip()
                    val = val.strip()
                    if not val:
                        new_dict: dict = {}
                        current_dict[key] = new_dict
                        stack.append((indent, new_dict))
                    else:
                        if (val.startswith("[") and val.endswith("]")) or (val.startswith("{") and val.endswith("}")):
                            val = json.loads(val.replace("'", '"'))
                        elif val.lower() == "true":
                            val = True
                        elif val.lower() == "false":
                            val = False
                        elif val.lower() in ("null", "none", "~"):
                            val = None
                        else:
                            try:
                                val = int(val)
                            except ValueError:
                                try:
                                    val = float(val)
                                except ValueError:
                                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                                        val = val[1:-1]
                        current_dict[key] = val
        return root


def run_pipeline(config_path: str = "configs/config.yaml") -> None:
    """Runs the end-to-end distress language classification pipeline."""
    cfg = load_config(config_path)
    seed = cfg.get("project", {}).get("random_seed", 42)
    set_seed(seed)

    logger.info("=" * 60)
    logger.info(cfg.get("project", {}).get("name", "Distress Language Classifier"))
    logger.info("=" * 60)

    # 1. Data Preprocessing & Anonymization
    logger.info("Step 1: Data Preprocessing & Splitting...")
    train_df, val_df, test_df = load_and_preprocess_dataset(cfg)

    # 2. Feature Extraction (TF-IDF)
    logger.info("Step 2: TF-IDF Feature Extraction...")
    tfidf = TFIDFExtractor(cfg)
    X_train = tfidf.fit_transform(train_df["cleaned_text"])
    X_val = tfidf.transform(val_df["cleaned_text"])
    X_test = tfidf.transform(test_df["cleaned_text"])

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    # 3. Model Training: TF-IDF + Baseline Classifiers (Logistic Regression, Linear SVM, Random Forest, Naive Bayes)
    logger.info("Step 3: Training Baseline Classifiers...")
    lr_model = LogisticRegressionModel(cfg).fit(X_train, y_train)
    svm_model = LinearSVMModel(cfg, calibrate=True).fit(X_train, y_train)
    rf_model = RandomForestModel(cfg).fit(X_train, y_train)
    nb_model = NaiveBayesModel(cfg).fit(X_train, y_train)

    # Predictions & Probabilities
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    svm_probs = svm_model.predict_proba(X_test)[:, 1]
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    nb_probs = nb_model.predict_proba(X_test)[:, 1]

    results = {
        "TF-IDF + Logistic Regression": {"y_prob": lr_probs},
        "TF-IDF + Linear SVM": {"y_prob": svm_probs},
        "TF-IDF + Random Forest": {"y_prob": rf_probs},
        "TF-IDF + Naive Bayes": {"y_prob": nb_probs},
    }

    # 4. Evaluation Across Baseline Models
    logger.info("Step 4: Evaluating Baseline Performance & Calibration (ECE)...")
    summary_df = evaluate_all_models(results, y_test)
    logger.info("\n" + summary_df.to_string(index=False))

    # 5. Confidence-based Abstention Analysis
    logger.info("Step 5: Confidence-Based Abstention & Uncertainty Handling...")
    sweep_thresholds = cfg.get("uncertainty", {}).get("abstention", {}).get(
        "threshold_sweep", [0.5, 0.6, 0.7, 0.8, 0.9]
    )
    abstention_df = evaluate_coverage_vs_accuracy(y_test, lr_probs, sweep_thresholds)
    logger.info("\nAbstention Coverage vs Selective Performance (Logistic Regression):\n" + abstention_df.to_string(index=False))

    # 6. Error Analysis
    logger.info("Step 6: Rigorous Error Analysis & Stigma Risk Audit...")
    abstainer = ConfidenceAbstainer(confidence_threshold=0.70)
    final_preds, abstained_mask, _ = abstainer.predict_with_abstention(lr_probs)
    perform_error_analysis(
        texts=test_df["text"].tolist(),
        y_true=y_test,
        y_prob=lr_probs,
        abstained_mask=abstained_mask,
        output_csv_path="reports/metrics/error_analysis_lr.csv",
    )

    logger.info("=" * 60)
    logger.info("Pipeline demonstration completed successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distress Language Classification Pipeline")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    run_pipeline(args.config)
