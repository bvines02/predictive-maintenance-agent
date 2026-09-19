"""Tests for src/error_analysis.py."""

import numpy as np
import pandas as pd
import pytest

from src.error_analysis import (
    DEFAULT_RUL_BANDS,
    add_error_analysis_columns,
    assign_rul_band,
    build_near_failure_comparison,
    compute_band_error_table,
    critical_region_analysis,
    threshold_diagnostics,
    top_over_predictions,
    top_under_predictions,
)


def _make_predictions_df():
    # error = predicted_rul - actual_rul, by convention.
    return pd.DataFrame(
        {
            "unit_number": [1, 1, 2, 2, 3, 3],
            "time_cycles": [10, 11, 20, 21, 30, 31],
            "actual_rul": [5, 20, 45, 70, 10, 100],
            "predicted_rul": [25, 15, 40, 75, 2, 95],
        }
    )


def _with_error(df):
    df = df.copy()
    df["error"] = df["predicted_rul"] - df["actual_rul"]
    return df


# --- Sign convention -------------------------------------------------------


def test_error_sign_convention_positive_means_over_prediction():
    # true RUL = 10, predicted RUL = 30 -> model thinks MORE life remains.
    df = _with_error(pd.DataFrame({"actual_rul": [10], "predicted_rul": [30]}))

    assert df["error"].iloc[0] == 20
    assert df["error"].iloc[0] > 0  # positive = optimistic / over-prediction


def test_error_sign_convention_negative_means_under_prediction():
    # true RUL = 30, predicted RUL = 10 -> model thinks LESS life remains.
    df = _with_error(pd.DataFrame({"actual_rul": [30], "predicted_rul": [10]}))

    assert df["error"].iloc[0] == -20
    assert df["error"].iloc[0] < 0  # negative = conservative / under-prediction


# --- RUL band assignment ---------------------------------------------------


def test_assign_rul_band_covers_each_default_band_boundary():
    values = pd.Series([0, 15, 16, 30, 31, 60, 61, 125])

    bands = assign_rul_band(values)

    assert bands.tolist() == [
        "Critical",
        "Critical",
        "High concern",
        "High concern",
        "Medium",
        "Medium",
        "Healthy / long horizon",
        "Healthy / long horizon",
    ]


def test_assign_rul_band_flags_out_of_range_as_unbanded():
    values = pd.Series([-5, 200])

    bands = assign_rul_band(values, bands=DEFAULT_RUL_BANDS)

    assert bands.tolist() == ["Unbanded", "Unbanded"]


def test_assign_rul_band_respects_custom_bands():
    custom_bands = [{"label": "Low", "min": 0, "max": 9}, {"label": "High", "min": 10, "max": 20}]
    values = pd.Series([5, 15])

    bands = assign_rul_band(values, bands=custom_bands)

    assert bands.tolist() == ["Low", "High"]


# --- Band error table -------------------------------------------------------


def test_compute_band_error_table_observations_sum_to_total_rows():
    df = _with_error(_make_predictions_df())
    df = add_error_analysis_columns(df)

    table = compute_band_error_table(df)

    assert table["observations"].sum() == len(df)


def test_compute_band_error_table_is_ordered_like_band_definitions():
    df = _with_error(_make_predictions_df())
    df = add_error_analysis_columns(df)

    table = compute_band_error_table(df)

    band_order = [b["label"] for b in DEFAULT_RUL_BANDS]
    present_bands = [b for b in band_order if b in table["rul_band"].tolist()]
    assert table["rul_band"].tolist() == present_bands


def test_compute_band_error_table_pct_over_and_under_are_consistent():
    df = _with_error(_make_predictions_df())
    df = add_error_analysis_columns(df)

    table = compute_band_error_table(df)

    # No row in the fixture has a zero error, so over + under should be 100%.
    assert np.allclose(table["pct_over"] + table["pct_under"], 100.0)


# --- Critical region analysis ----------------------------------------------


def test_critical_region_analysis_only_uses_rows_at_or_below_threshold():
    df = _with_error(_make_predictions_df())

    stats = critical_region_analysis(df, critical_max=15)

    # Only unit 1 (actual_rul=5) and unit 3 (actual_rul=10) qualify.
    assert stats["n_observations"] == 2


def test_critical_region_analysis_max_over_and_under_prediction():
    df = _with_error(_make_predictions_df())

    stats = critical_region_analysis(df, critical_max=15)

    # unit 1: error=20 (over); unit 3: error=-8 (under).
    assert stats["max_over_prediction"] == 20
    assert stats["max_under_prediction"] == -8


# --- Top over/under predictions ---------------------------------------------


def test_top_over_predictions_sorted_descending_by_error():
    df = _with_error(_make_predictions_df())

    top = top_over_predictions(df, n=3)

    assert top["error"].tolist() == sorted(top["error"].tolist(), reverse=True)
    assert top.iloc[0]["actual_rul"] == 5  # the biggest over-prediction (error=20)


def test_top_under_predictions_sorted_ascending_by_error():
    df = _with_error(_make_predictions_df())

    bottom = top_under_predictions(df, n=3)

    assert bottom["error"].tolist() == sorted(bottom["error"].tolist())
    assert bottom.iloc[0]["actual_rul"] == 10  # the biggest under-prediction (error=-8)


# --- Threshold diagnostics (confusion matrix) -------------------------------


def test_threshold_diagnostics_confusion_matrix_matches_hand_calculation():
    df = _with_error(_make_predictions_df())
    # actual_rul <= 15: units at rows (actual=5) and (actual=10) -> "near failure".
    # predicted_rul <= 15: rows with predicted 15, 2 -> "warned".
    # Row-by-row (actual, predicted): (5,25)FN (20,15)FP (45,40)TN (70,75)TN (10,2)TP (100,95)TN

    result = threshold_diagnostics(df, actual_threshold=15, predicted_threshold=15)

    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["false_negatives"] == 1
    assert result["true_negatives"] == 3


def test_threshold_diagnostics_precision_and_recall():
    df = _with_error(_make_predictions_df())

    result = threshold_diagnostics(df, actual_threshold=15, predicted_threshold=15)

    # precision = TP / (TP + FP) = 1 / (1 + 1) = 0.5
    # recall = TP / (TP + FN) = 1 / (1 + 1) = 0.5
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)


def test_threshold_diagnostics_handles_no_positive_predictions_without_error():
    df = pd.DataFrame({"actual_rul": [50, 60], "predicted_rul": [55, 65]})
    df["error"] = df["predicted_rul"] - df["actual_rul"]

    result = threshold_diagnostics(df, actual_threshold=15, predicted_threshold=15)

    assert result["true_positives"] == 0
    assert result["false_positives"] == 0
    assert result["precision"] == 0.0  # 0/0 handled safely, not a ZeroDivisionError
    assert result["recall"] == 0.0


def test_threshold_diagnostics_more_conservative_threshold_catches_more():
    df = _with_error(_make_predictions_df())

    strict = threshold_diagnostics(df, actual_threshold=15, predicted_threshold=15)
    lenient = threshold_diagnostics(df, actual_threshold=15, predicted_threshold=25)

    # Raising the predicted-RUL warning threshold should never reduce recall.
    assert lenient["recall"] >= strict["recall"]


# --- Near-failure model comparison ------------------------------------------


def test_build_near_failure_comparison_has_one_row_per_model():
    df_a = _with_error(_make_predictions_df())
    df_b = _with_error(_make_predictions_df())

    comparison = build_near_failure_comparison({"Model A": df_a, "Model B": df_b})

    assert comparison["model"].tolist() == ["Model A", "Model B"]
    assert "overall_mae" in comparison.columns
    assert "rul_le_30_mae" in comparison.columns
    assert "rul_le_15_mae" in comparison.columns
