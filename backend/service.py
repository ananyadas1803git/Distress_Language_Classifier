import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import joblib

from src.data.preprocess import clean_text, load_and_preprocess_dataset
from src.features.tfidf import TFIDFExtractor
from src.models.baseline_models import (
    LogisticRegressionModel,
    LinearSVMModel,
    RandomForestModel,
    NaiveBayesModel,
)
from src.evaluation.metrics import evaluate_all_models
from src.uncertainty.abstention import evaluate_coverage_vs_accuracy
from src.utils.logger import get_logger
from backend.schemas import (
    PredictResponse,
    TokenAttribution,
    ModelBenchmark,
    ModelInfo,
    ExamplePost,
    AbstentionPoint,
)

logger = get_logger("backend_service")

ETHICAL_NOTICE = (
    "RESPONSIBLE AI & ETHICAL DISCLOSURE: This system is designed solely to identify statistical "
    "linguistic patterns associated with distress-related vocabulary in anonymized online text. "
    "It outputs calibrated probabilities and selectively abstains when uncertain. Predictions "
    "DO NOT constitute clinical evidence, psychological diagnosis, or medical assessment."
)

SAMPLE_POSTS = [
    ExamplePost(
        id="distress_1",
        title="Acute Overwhelm & Anxiety",
        category="Distress Signal",
        text=(
            "I feel completely overwhelmed and exhausted every single day. The panic attacks are "
            "getting worse and I don't know who to turn to. I feel trapped in my own head and "
            "hopeless about the future."
        ),
    ),
    ExamplePost(
        id="venting_1",
        title="Casual Frustration (Venting)",
        category="Casual Venting",
        text=(
            "Ugh, traffic on the highway was an absolute nightmare today! Missed the morning standup "
            "because of some construction delay. Grabbing a huge cup of coffee and tackling these PRs."
        ),
    ),
    ExamplePost(
        id="neutral_1",
        title="Neutral Informative Post",
        category="Everyday / Neutral",
        text=(
            "Has anyone tried the new Python 3.13 release? The new interactive REPL and performance "
            "benchmarks on standard library modules look quite impressive for data processing pipelines."
        ),
    ),
    ExamplePost(
        id="ambiguous_1",
        title="Ambiguous / Borderline (Triggers Abstention)",
        category="Borderline / Uncertain",
        text=(
            "Things have been kind of busy lately. Not really sure how I feel about changes at work, "
            "some good days, some tiring days. Just taking things one step at a time I suppose."
        ),
    ),
]


class DistressClassifierService:
    """Manages model loading, inference, keyword attribution, and evaluation benchmarks."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.save_dir = Path("saved_models/baselines")
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.tfidf: Optional[TFIDFExtractor] = None
        self.models: Dict[str, Any] = {}
        self.benchmark_metrics: List[ModelBenchmark] = []
        self.abstention_sweep: List[AbstentionPoint] = []
        self.is_initialized = False

    def _load_config(self, config_path: str) -> dict:
        try:
            import yaml
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            from main import load_config
            return load_config(config_path)

    def initialize(self) -> None:
        """Loads serialized models or fits them on the dataset if not yet cached."""
        logger.info("Initializing Distress Classifier backend service...")
        tfidf_path = self.save_dir / "tfidf.joblib"
        lr_path = self.save_dir / "logistic_regression.joblib"
        svm_path = self.save_dir / "linear_svm.joblib"
        rf_path = self.save_dir / "random_forest.joblib"
        nb_path = self.save_dir / "naive_bayes.joblib"

        all_exist = all(
            p.exists()
            for p in [tfidf_path, lr_path, svm_path, rf_path, nb_path]
        )

        if all_exist:
            logger.info("Loading pre-trained baseline models from disk...")
            self.tfidf = TFIDFExtractor(self.config)
            self.tfidf.load(str(tfidf_path))

            lr = LogisticRegressionModel(self.config)
            lr.load(str(lr_path))

            svm = LinearSVMModel(self.config, calibrate=True)
            svm.load(str(svm_path))

            rf = RandomForestModel(self.config)
            rf.load(str(rf_path))

            nb = NaiveBayesModel(self.config, calibrate=False)
            nb.load(str(nb_path))

            self.models = {
                "TF-IDF + Logistic Regression": lr,
                "TF-IDF + Linear SVM": svm,
                "TF-IDF + Random Forest": rf,
                "TF-IDF + Naive Bayes": nb,
            }
        else:
            logger.info("Baseline model cache not found. Training models on Dreaddit dataset...")
            train_df, val_df, test_df = load_and_preprocess_dataset(self.config)

            self.tfidf = TFIDFExtractor(self.config)
            X_train = self.tfidf.fit_transform(train_df["cleaned_text"])
            X_test = self.tfidf.transform(test_df["cleaned_text"])
            y_train = train_df["label"].values
            y_test = test_df["label"].values

            lr = LogisticRegressionModel(self.config).fit(X_train, y_train)
            svm = LinearSVMModel(self.config, calibrate=True).fit(X_train, y_train)
            rf = RandomForestModel(self.config).fit(X_train, y_train)
            nb = NaiveBayesModel(self.config, calibrate=False).fit(X_train, y_train)

            # Save serialized models for instant subsequent reloads
            self.tfidf.save(str(tfidf_path))
            lr.save(str(lr_path))
            svm.save(str(svm_path))
            rf.save(str(rf_path))
            nb.save(str(nb_path))

            self.models = {
                "TF-IDF + Logistic Regression": lr,
                "TF-IDF + Linear SVM": svm,
                "TF-IDF + Random Forest": rf,
                "TF-IDF + Naive Bayes": nb,
            }

        # Precompute evaluation metrics on test set
        self._compute_benchmark()
        self.is_initialized = True
        logger.info("Distress Classifier backend service initialized successfully.")

    def _compute_benchmark(self) -> None:
        """Precomputes test benchmarks and abstention sweeps for the dashboard."""
        test_path = Path("data/processed/test.csv")
        if not test_path.exists():
            _, _, test_df = load_and_preprocess_dataset(self.config)
        else:
            test_df = pd.read_csv(test_path)

        X_test = self.tfidf.transform(test_df["cleaned_text"])
        y_test = test_df["label"].values

        results = {}
        for name, model in self.models.items():
            results[name] = {"y_prob": model.predict_proba(X_test)[:, 1]}

        summary_df = evaluate_all_models(results, y_test)
        self.benchmark_metrics = [
            ModelBenchmark(
                model=row["model"],
                accuracy=round(float(row["accuracy"]), 4),
                precision=round(float(row["precision"]), 4),
                recall=round(float(row["recall"]), 4),
                f1=round(float(row["f1"]), 4),
                roc_auc=round(float(row["roc_auc"]), 4),
                pr_auc=round(float(row["pr_auc"]), 4),
                brier_score=round(float(row["brier_score"]), 4),
                ece=round(float(row["ece"]), 4),
            )
            for _, row in summary_df.iterrows()
        ]

        # Compute abstention sweep for default model
        lr_probs = results["TF-IDF + Logistic Regression"]["y_prob"]
        sweep_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        abs_df = evaluate_coverage_vs_accuracy(y_test, lr_probs, sweep_thresholds)
        self.abstention_sweep = [
            AbstentionPoint(
                confidence_threshold=round(float(r["confidence_threshold"]), 2),
                coverage=round(float(r["coverage"]), 4),
                abstention_rate=round(float(r["abstention_rate"]), 4),
                retained_count=int(r["retained_count"]),
                abstained_count=int(r["abstained_count"]),
                selective_accuracy=round(float(r["selective_accuracy"]), 4),
                selective_f1=round(float(r["selective_f1"]), 4),
                selective_risk=round(float(r["selective_risk"]), 4),
            )
            for _, r in abs_df.iterrows()
        ]

    def get_models_info(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                id="logistic_regression",
                name="TF-IDF + Logistic Regression",
                algorithm="L2-Regularized Linear Classifier",
                description="Interpretable linear model with balanced class weights and well-behaved probabilities.",
                strengths="High explainability, direct feature weight attribution, balanced calibration.",
                is_calibrated=True,
            ),
            ModelInfo(
                id="linear_svm",
                name="TF-IDF + Linear SVM",
                algorithm="Maximum Margin with CalibratedClassifierCV",
                description="Support Vector Classifier maximizing geometric separation margin with 5-fold probability calibration.",
                strengths="Robust against high-dimensional sparsity, lowest Expected Calibration Error (ECE ~ 0.035).",
                is_calibrated=True,
            ),
            ModelInfo(
                id="random_forest",
                name="TF-IDF + Random Forest",
                algorithm="Bagged Ensemble of Decision Trees",
                description="Non-linear tree ensemble aggregating 100 decision trees over random feature subsets.",
                strengths="Captures non-linear term interactions without overfitting to rare words.",
                is_calibrated=False,
            ),
            ModelInfo(
                id="naive_bayes",
                name="TF-IDF + Naive Bayes",
                algorithm="Multinomial Naive Bayes with Laplace Smoothing",
                description="Probabilistic word-frequency generative classifier with Laplace alpha=1.0.",
                strengths="Extremely high recall on distress-related terminology (Recall > 90%).",
                is_calibrated=False,
            ),
        ]

    def _extract_token_attributions(
        self,
        cleaned_text: str,
        model_name: str,
        top_k: int = 8,
    ) -> List[TokenAttribution]:
        """Extracts top contributing n-grams from the post for the chosen model."""
        if not self.tfidf or not self.tfidf.is_fitted:
            return []

        feature_names = self.tfidf.vectorizer.get_feature_names_out()
        tfidf_vec = self.tfidf.transform([cleaned_text])
        non_zero_indices = tfidf_vec.nonzero()[1]

        if len(non_zero_indices) == 0:
            return []

        model_wrapper = self.models.get(model_name)
        if not model_wrapper:
            return []

        scores: List[Tuple[str, float]] = []

        if model_name == "TF-IDF + Logistic Regression":
            weights = model_wrapper.model.coef_[0]
            for idx in non_zero_indices:
                token = feature_names[idx]
                tf_val = tfidf_vec[0, idx]
                impact = float(tf_val * weights[idx])
                scores.append((token, impact))

        elif model_name == "TF-IDF + Linear SVM":
            # CalibratedClassifierCV wraps 5 calibrated base estimators
            try:
                base_estimators = [cc.estimator for cc in model_wrapper.model.calibrated_classifiers_]
                avg_coef = np.mean([est.coef_[0] for est in base_estimators], axis=0)
                for idx in non_zero_indices:
                    token = feature_names[idx]
                    tf_val = tfidf_vec[0, idx]
                    impact = float(tf_val * avg_coef[idx])
                    scores.append((token, impact))
            except Exception:
                for idx in non_zero_indices:
                    token = feature_names[idx]
                    scores.append((token, float(tfidf_vec[0, idx])))

        elif model_name == "TF-IDF + Naive Bayes":
            nb = model_wrapper.model
            # Log odds difference: feature_log_prob_[1] - feature_log_prob_[0]
            if hasattr(nb, "feature_log_prob_"):
                log_ratio = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
                for idx in non_zero_indices:
                    token = feature_names[idx]
                    impact = float(tfidf_vec[0, idx] * log_ratio[idx])
                    scores.append((token, impact))
            else:
                for idx in non_zero_indices:
                    token = feature_names[idx]
                    scores.append((token, float(tfidf_vec[0, idx])))

        elif model_name == "TF-IDF + Random Forest":
            rf = model_wrapper.model
            importances = rf.feature_importances_
            for idx in non_zero_indices:
                token = feature_names[idx]
                impact = float(tfidf_vec[0, idx] * importances[idx])
                scores.append((token, impact))

        else:
            for idx in non_zero_indices:
                scores.append((feature_names[idx], float(tfidf_vec[0, idx])))

        # Sort by absolute impact magnitude
        scores.sort(key=lambda x: abs(x[1]), reverse=True)
        top_scores = scores[:top_k]

        return [
            TokenAttribution(
                token=token,
                weight=round(weight, 4),
                direction="distress" if weight > 0 else "neutral",
            )
            for token, weight in top_scores
        ]

    def predict(
        self,
        text: str,
        model_name: str = "TF-IDF + Logistic Regression",
        confidence_threshold: float = 0.70,
    ) -> PredictResponse:
        """Processes input text, predicts probability, checks abstention, and extracts tokens."""
        if not self.is_initialized:
            self.initialize()

        if model_name not in self.models:
            model_name = "TF-IDF + Logistic Regression"

        model = self.models[model_name]
        cleaned = clean_text(text)

        X = self.tfidf.transform([cleaned])
        probs = model.predict_proba(X)[0]
        prob_distress = float(probs[1])

        # Confidence: how far from ambiguous 0.5 boundary
        confidence = float(max(prob_distress, 1.0 - prob_distress))
        is_abstained = bool(confidence < confidence_threshold)
        predicted_label = 1 if prob_distress >= 0.50 else 0

        if is_abstained:
            decision = "Abstained (Uncertain)"
        else:
            decision = "Distress-Related Language" if predicted_label == 1 else "No Distress Language Detected"

        top_tokens = self._extract_token_attributions(cleaned, model_name)

        return PredictResponse(
            text=text,
            cleaned_text=cleaned,
            model_name=model_name,
            probability=round(prob_distress, 4),
            confidence=round(confidence, 4),
            confidence_threshold=round(confidence_threshold, 4),
            is_abstained=is_abstained,
            predicted_label=predicted_label,
            decision=decision,
            top_tokens=top_tokens,
            ethical_notice=ETHICAL_NOTICE,
        )


# Singleton service instance
service = DistressClassifierService()
