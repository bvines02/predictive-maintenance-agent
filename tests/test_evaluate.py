"""Tests for src/evaluate.py.

Uses a tiny hand-built "model" (not scikit-learn) so the test is fast and
checks the metric wiring itself, not RandomForest's behavior.
"""

import numpy as np

from src.evaluate import evaluate_model


class PerfectModel:
    """A fake model that always predicts the true label correctly."""

    def __init__(self, y_true):
        self._y_true = np.asarray(y_true)

    def predict(self, X):
        return self._y_true

    def predict_proba(self, X):
        # Confident probabilities matching the true label.
        return np.column_stack([1 - self._y_true, self._y_true]).astype(float)


def test_evaluate_model_perfect_predictions():
    y_test = [0, 0, 0, 1, 1]
    model = PerfectModel(y_test)

    metrics = evaluate_model(model, X_test=None, y_test=y_test)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"].tolist() == [[3, 0], [0, 2]]


def test_evaluate_model_returns_expected_keys():
    y_test = [0, 1, 0, 1]
    model = PerfectModel(y_test)

    metrics = evaluate_model(model, X_test=None, y_test=y_test)

    assert set(metrics.keys()) == {
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
    }
