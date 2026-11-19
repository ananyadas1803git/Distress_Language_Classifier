# Responsible Classifier for Detecting Distress-Related Language in Anonymized Online Posts

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-brightgreen.svg)](https://shap.readthedocs.io/)

---

## 📌 Problem Statement & Ethical Scope

Online posts frequently contain expressions associated with acute stress, emotional distress, social isolation, or low mood. While manual review of such posts at scale is infeasible, automated models deployed in sensitive contexts often suffer from severe limitations:

1. **Clinical Overclaiming:** Treating text classification as automated "depression diagnosis" or clinical profiling, when text alone cannot establish medical diagnoses.
2. **Uncalibrated Predictions & Absence of Uncertainty Handling:** Models outputting high-confidence labels on ambiguous language without communicating uncertainty.
3. **Opaque & Unsafe Explanations:** Overstating feature attribution as clinical symptoms rather than statistical language cues.
4. **False-Positive Stigmatization:** Misinterpreting venting or colloquial emotional expression as distress.
5. **Privacy and Bias Concerns:** Exposure of personally identifiable handles/subreddits in social-media data.
6. **Superficial Error Audits:** Evaluating only aggregate accuracy without dissecting false positives, false negatives, or failure modes.

> **Responsible AI Mandate:**  
> This system is **strictly non-diagnostic**. It identifies statistical distress-related language signals in anonymized online text, provides calibrated probabilities, abstains when uncertain, and explains token attributions with cautious linguistic disclaimers.

---

## 🔬 Algorithms & Methodological Pipeline

The project implements a rigorous comparative framework spanning traditional interpretable baselines, deep recurrent networks, uncertainty quantification, and safe explainability:

1. **TF-IDF + Logistic Regression:** Interpretable linear baseline with L2 regularization and balanced class weighting.
2. **TF-IDF + Linear SVM:** Maximum-margin baseline with calibrated decision probabilities.
3. **TF-IDF + Random Forest:** Non-linear tree ensemble baseline leveraging bagging over feature subsets.
4. **TF-IDF + Multinomial Naive Bayes:** Probabilistic frequency-based generative baseline with Laplace smoothing.
5. **LSTM (Unidirectional):** Sequential neural representation capturing long-range linguistic context and sentiment trajectory.
6. **BiLSTM (Bidirectional):** Dual-direction sequence modeling capturing forward and backward context across posts.
7. **SHAP-Based Explanations:** Token and phrase-level attributions (via `LinearExplainer` and model-agnostic `Explainer`) equipped with ethical, non-diagnostic disclaimers.
8. **Probability Calibration & Confidence-Based Abstention:**
   - Expected Calibration Error (ECE) and reliability diagrams.
   - Platt scaling, Isotonic regression, and Temperature scaling.
   - Confidence-based abstention (deferral) to prevent decisions on ambiguous posts and plot Coverage vs. Selective Risk curves.

---

## 📂 Project Directory Structure

```text
Distress Language Classifier/
├── configs/
│   └── config.yaml               # Centralized hyperparameters, paths, and thresholds
├── data/
│   ├── raw/
│   │   └── dreaddit.csv          # Raw Dreaddit social media distress dataset
│   └── processed/                # Anonymized, cleaned, and stratified splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── notebooks/                    # Interactive step-by-step experiment notebooks
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_recurrent_models.ipynb
│   ├── 04_calibration_and_abstention.ipynb
│   └── 05_shap_explanations.ipynb
├── reports/
│   ├── figures/                  # Reliability curves, SHAP plots, and coverage graphs
│   └── metrics/                  # Performance benchmarks and error audit logs
├── saved_models/
│   ├── baselines/                # Serialized TF-IDF vectorizers and linear models
│   ├── neural/                   # PyTorch checkpoints for LSTM & BiLSTM
│   └── scalers/                  # Calibration scalers & temperature models
├── src/
│   ├── __init__.py
│   ├── download_dataset.py       # Dataset fetcher script
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocess.py         # Text cleaning, anonymization (handles/URLs), splitting
│   │   └── dataset.py            # PyTorch Vocabulary & Dataset for recurrent models
│   ├── features/
│   │   ├── __init__.py
│   │   └── tfidf.py              # TF-IDF n-gram feature extraction & persistence
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_models.py    # TF-IDF + Logistic Regression, Linear SVM, Random Forest & Naive Bayes
│   │   ├── recurrent_models.py   # PyTorch LSTM & BiLSTM architectures
│   │   └── trainer.py            # PyTorch training loop with early stopping & evaluation
│   ├── uncertainty/
│   │   ├── __init__.py
│   │   ├── calibration.py        # ECE, reliability diagrams, temperature scaling
│   │   └── abstention.py         # Confidence thresholding, selective risk-coverage curves
│   ├── explainability/
│   │   ├── __init__.py
│   │   └── shap_explainer.py     # SHAP local/global attributions with ethical disclaimers
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier, ECE
│   │   └── error_analysis.py     # Categorization into TP, TN, FP (stigma risk), FN, Abstained
│   └── utils/
│       ├── __init__.py
│       ├── logger.py             # Formatted console and file logging
│       └── seed.py               # Deterministic seed setting for reproducibility
├── main.py                       # End-to-end execution pipeline
├── requirements.txt              # Project dependencies
├── .gitignore                    # Git tracking rules
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 Getting Started

### 1. Environment Setup

Clone or open the workspace and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify or Download Dataset

The raw dataset (`dreaddit.csv`) is placed under `data/raw/dreaddit.csv`. If you need to re-download:

```bash
python src/download_dataset.py
```

### 3. Run End-to-End Pipeline

Execute the full baseline, evaluation, abstention, and error analysis pipeline:

```bash
python main.py --config configs/config.yaml
```

---

## 🛡️ Responsible AI & Ethical Design

- **Privacy & Anonymization:** Preprocessing strips Reddit handles (`r/...`, `u/...`), user tags, and external hyperlinks before modeling.
- **Abstention over Guessing:** If model confidence falls below a configured threshold (e.g., $P \in [0.4, 0.6]$), the classifier explicitly abstains (`status = Abstained`) rather than outputting a forced classification.
- **Transparent Language Attribution:** SHAP attributions are accompanied by standardized non-diagnostic disclosures to prevent clinical misinterpretation.
