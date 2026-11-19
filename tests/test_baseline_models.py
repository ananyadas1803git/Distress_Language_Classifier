import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix

from src.models.baseline_models import (
    LogisticRegressionModel,
    LinearSVMModel,
    RandomForestModel,
    NaiveBayesModel,
    MultinomialNBModel,
)
from src.evaluation.metrics import evaluate_all_models


class TestBaselineModels(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "models": {
                "logistic_regression": {"max_iter": 200, "random_state": 42},
                "linear_svm": {"max_iter": 500, "random_state": 42},
                "random_forest": {"n_estimators": 10, "random_state": 42},
                "naive_bayes": {"alpha": 1.0, "fit_prior": True},
            }
        }
        # Synthetic sparse feature matrix (10 samples, 8 features)
        rng = np.random.RandomState(42)
        dense_X = rng.exponential(scale=1.0, size=(10, 8))
        dense_X[dense_X < 0.5] = 0.0  # Sparsify
        self.X_train = csr_matrix(dense_X)
        self.y_train = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0])

        dense_X_test = rng.exponential(scale=1.0, size=(4, 8))
        dense_X_test[dense_X_test < 0.5] = 0.0
        self.X_test = csr_matrix(dense_X_test)
        self.y_test = np.array([0, 1, 1, 0])

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_random_forest_fit_predict(self):
        rf = RandomForestModel(self.config)
        self.assertFalse(rf.is_fitted)

        rf.fit(self.X_train, self.y_train)
        self.assertTrue(rf.is_fitted)

        preds = rf.predict(self.X_test)
        self.assertEqual(preds.shape, (4,))
        self.assertTrue(set(preds).issubset({0, 1}))

        probs = rf.predict_proba(self.X_test)
        self.assertEqual(probs.shape, (4, 2))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_random_forest_save_and_load(self):
        rf = RandomForestModel(self.config).fit(self.X_train, self.y_train)
        save_path = str(Path(self.temp_dir) / "rf_model.joblib")
        rf.save(save_path)
        self.assertTrue(Path(save_path).exists())

        loaded_rf = RandomForestModel(self.config)
        loaded_rf.load(save_path)
        self.assertTrue(loaded_rf.is_fitted)

        orig_preds = rf.predict(self.X_test)
        loaded_preds = loaded_rf.predict(self.X_test)
        np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_naive_bayes_fit_predict(self):
        nb = NaiveBayesModel(self.config)
        self.assertFalse(nb.is_fitted)

        nb.fit(self.X_train, self.y_train)
        self.assertTrue(nb.is_fitted)

        preds = nb.predict(self.X_test)
        self.assertEqual(preds.shape, (4,))
        self.assertTrue(set(preds).issubset({0, 1}))

        probs = nb.predict_proba(self.X_test)
        self.assertEqual(probs.shape, (4, 2))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_naive_bayes_calibrated(self):
        nb_cal = NaiveBayesModel(self.config, calibrate=True)
        nb_cal.fit(self.X_train, self.y_train)
        self.assertTrue(nb_cal.is_fitted)

        probs = nb_cal.predict_proba(self.X_test)
        self.assertEqual(probs.shape, (4, 2))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_naive_bayes_alias(self):
        self.assertIs(MultinomialNBModel, NaiveBayesModel)

    def test_naive_bayes_save_and_load(self):
        nb = NaiveBayesModel(self.config).fit(self.X_train, self.y_train)
        save_path = str(Path(self.temp_dir) / "nb_model.joblib")
        nb.save(save_path)
        self.assertTrue(Path(save_path).exists())

        loaded_nb = NaiveBayesModel(self.config)
        loaded_nb.load(save_path)
        self.assertTrue(loaded_nb.is_fitted)

        orig_preds = nb.predict(self.X_test)
        loaded_preds = loaded_nb.predict(self.X_test)
        np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_evaluate_all_four_baselines(self):
        lr = LogisticRegressionModel(self.config).fit(self.X_train, self.y_train)
        svm = LinearSVMModel(self.config, calibrate=True).fit(self.X_train, self.y_train)
        rf = RandomForestModel(self.config).fit(self.X_train, self.y_train)
        nb = NaiveBayesModel(self.config).fit(self.X_train, self.y_train)

        results = {
            "TF-IDF + Logistic Regression": {"y_prob": lr.predict_proba(self.X_test)[:, 1]},
            "TF-IDF + Linear SVM": {"y_prob": svm.predict_proba(self.X_test)[:, 1]},
            "TF-IDF + Random Forest": {"y_prob": rf.predict_proba(self.X_test)[:, 1]},
            "TF-IDF + Naive Bayes": {"y_prob": nb.predict_proba(self.X_test)[:, 1]},
        }

        df_metrics = evaluate_all_models(results, self.y_test)
        self.assertEqual(len(df_metrics), 4)
        expected_models = {
            "TF-IDF + Logistic Regression",
            "TF-IDF + Linear SVM",
            "TF-IDF + Random Forest",
            "TF-IDF + Naive Bayes",
        }
        self.assertEqual(set(df_metrics["model"]), expected_models)


if __name__ == "__main__":
    unittest.main()
