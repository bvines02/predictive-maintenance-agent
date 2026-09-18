"""Tests for src/evaluate_rul.py."""

import numpy as np

from src.evaluate_rul import evaluate_rul_predictions


def test_evaluate_rul_predictions_perfect_predictions():
    y_true = [10, 20, 30]
    y_pred = [10, 20, 30]

    metrics = evaluate_rul_predictions(y_true, y_pred)

    assert metrics["mae"] == 0
    assert metrics["rmse"] == 0


def test_evaluate_rul_predictions_matches_worked_example():
    y_true = [10, 20, 30]
    y_pred = [15, 15, 35]  # errors: +5, -5, +5

    metrics = evaluate_rul_predictions(y_true, y_pred)

    assert metrics["mae"] == 5.0
    assert np.isclose(metrics["rmse"], 5.0)


def test_evaluate_rul_predictions_rmse_penalises_large_errors_more_than_mae():
    y_true = [0, 0, 0, 0]
    y_pred = [1, 1, 1, 10]  # one large outlier error

    metrics = evaluate_rul_predictions(y_true, y_pred)

    # RMSE should be pulled up more than MAE by the single large error.
    assert metrics["rmse"] > metrics["mae"]


def test_evaluate_rul_predictions_returns_expected_keys():
    metrics = evaluate_rul_predictions([1, 2, 3], [1, 2, 3])

    assert set(metrics.keys()) == {"mae", "rmse"}
