"""Tests for src/evaluate_test_set.py.

Uses tiny hand-built DataFrames/files instead of the real NASA test set, so
these tests stay fast and independent of the downloaded dataset.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.evaluate_test_set import (
    build_final_predictions_df,
    build_test_features,
    compare_validation_vs_test,
    compute_summary_metrics,
    infer_kept_sensors_from_feature_list,
    load_and_validate_test_data,
    select_final_cycle_rows,
    verify_features_present,
)


def _write_cmapss_file(path, rows):
    lines = [" ".join(str(v) for v in row) + " " for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_row(unit_number, time_cycles):
    return [unit_number, time_cycles] + [0.0] * 24


# --- Alignment checks -------------------------------------------------------


def test_load_and_validate_test_data_accepts_clean_alignment(tmp_path):
    rows = [_make_row(1, 1), _make_row(1, 2), _make_row(2, 1)]
    test_path = _write_cmapss_file(tmp_path / "test_mock.txt", rows)
    rul_path = tmp_path / "RUL_mock.txt"
    rul_path.write_text("10 \n20 \n")

    test_df, rul = load_and_validate_test_data(test_path, rul_path)

    assert test_df["unit_number"].nunique() == 2
    assert list(rul) == [10, 20]


def test_load_and_validate_test_data_rejects_count_mismatch(tmp_path):
    rows = [_make_row(1, 1), _make_row(2, 1)]
    test_path = _write_cmapss_file(tmp_path / "test_mock.txt", rows)
    rul_path = tmp_path / "RUL_mock.txt"
    rul_path.write_text("10 \n20 \n30 \n")  # one too many

    with pytest.raises(ValueError, match="cannot align"):
        load_and_validate_test_data(test_path, rul_path)


def test_load_and_validate_test_data_rejects_non_dense_unit_numbers(tmp_path):
    rows = [_make_row(1, 1), _make_row(3, 1)]  # missing unit 2
    test_path = _write_cmapss_file(tmp_path / "test_mock.txt", rows)
    rul_path = tmp_path / "RUL_mock.txt"
    rul_path.write_text("10 \n20 \n")

    with pytest.raises(ValueError, match="dense sequence"):
        load_and_validate_test_data(test_path, rul_path)


# --- Sensor recovery, final-row selection -----------------------------------


def test_infer_kept_sensors_recovers_only_sensors_present_in_feature_list():
    feature_columns = ["time_cycles", "operational_setting_1", "sensor_2", "sensor_9", "sensor_2_roll_mean_5"]

    kept = infer_kept_sensors_from_feature_list(feature_columns, sensor_columns=[f"sensor_{i}" for i in range(1, 22)])

    assert kept == ["sensor_2", "sensor_9"]


def test_select_final_cycle_rows_picks_max_cycle_per_engine():
    df = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 2, 2],
            "time_cycles": [1, 2, 3, 1, 2],
            "sensor_2": [10.0, 11.0, 12.0, 20.0, 21.0],
        }
    )

    final_rows = select_final_cycle_rows(df)

    assert final_rows["unit_number"].tolist() == [1, 2]
    assert final_rows["last_observed_cycle"].tolist() == [3, 2]
    assert final_rows["sensor_2"].tolist() == [12.0, 21.0]


def test_verify_features_present_raises_on_missing_column():
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(ValueError, match="missing"):
        verify_features_present(df, ["a", "b"])


def test_verify_features_present_passes_when_all_columns_exist():
    df = pd.DataFrame({"a": [1], "b": [2]})

    verify_features_present(df, ["a", "b"])  # should not raise


# --- build_final_predictions_df: capping, sign convention -------------------


def _fitted_rf_on(feature_columns, n_rows=20):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({c: rng.normal(size=n_rows) for c in feature_columns})
    y = X[feature_columns[0]] * 5 + 50
    model = RandomForestRegressor(n_estimators=5, random_state=0)
    model.fit(X, y)
    return model


def test_build_final_predictions_df_caps_actual_rul_but_keeps_uncapped():
    feature_columns = ["x1"]
    final_rows_df = pd.DataFrame({"unit_number": [1, 2], "last_observed_cycle": [30, 40], "x1": [0.0, 0.0]})
    model = _fitted_rf_on(feature_columns, n_rows=20)
    actual_rul = pd.Series([200, 50])  # first exceeds the 125 cap

    result = build_final_predictions_df(final_rows_df, feature_columns, model, actual_rul, rul_cap=125)

    assert result["actual_rul_uncapped"].tolist() == [200, 50]
    assert result["actual_rul"].tolist() == [125, 50]  # capped


def test_build_final_predictions_df_error_sign_convention():
    feature_columns = ["x1"]
    final_rows_df = pd.DataFrame({"unit_number": [1], "last_observed_cycle": [10], "x1": [0.0]})
    model = _fitted_rf_on(feature_columns, n_rows=20)

    result = build_final_predictions_df(final_rows_df, feature_columns, model, pd.Series([50]))

    # error = predicted - actual_capped; sign convention check via direct computation
    expected_error = result["predicted_rul"].iloc[0] - result["actual_rul"].iloc[0]
    assert result["error"].iloc[0] == pytest.approx(expected_error)
    assert result["absolute_error"].iloc[0] == pytest.approx(abs(expected_error))


def test_build_final_predictions_df_row_count_matches_engines():
    feature_columns = ["x1"]
    final_rows_df = pd.DataFrame(
        {"unit_number": [1, 2, 3], "last_observed_cycle": [10, 20, 30], "x1": [0.0, 0.0, 0.0]}
    )
    model = _fitted_rf_on(feature_columns, n_rows=20)

    result = build_final_predictions_df(final_rows_df, feature_columns, model, pd.Series([50, 60, 70]))

    assert len(result) == 3


# --- Summary metrics ---------------------------------------------------------


def test_compute_summary_metrics_matches_hand_calculation():
    df = pd.DataFrame({"error": [10.0, -5.0, 0.0], "absolute_error": [10.0, 5.0, 0.0]})

    metrics = compute_summary_metrics(df, label="test")

    assert metrics["n"] == 3
    assert metrics["mae"] == pytest.approx(5.0)
    assert metrics["mean_signed_error"] == pytest.approx(5.0 / 3)
    assert metrics["median_absolute_error"] == pytest.approx(5.0)
    assert metrics["max_over_prediction"] == pytest.approx(10.0)
    assert metrics["max_under_prediction"] == pytest.approx(-5.0)
    assert metrics["rmse"] == pytest.approx(np.sqrt((100 + 25 + 0) / 3))


def test_compare_validation_vs_test_has_two_rows():
    val_metrics = compute_summary_metrics(pd.DataFrame({"error": [1.0], "absolute_error": [1.0]}), "Validation")
    test_metrics = compute_summary_metrics(pd.DataFrame({"error": [2.0], "absolute_error": [2.0]}), "Test")

    comparison = compare_validation_vs_test(val_metrics, test_metrics)

    assert comparison["dataset"].tolist() == ["Validation", "Test"]
    assert comparison["mae"].tolist() == [1.0, 2.0]


# --- build_test_features: reuses the causal pipeline, no leakage regression -


def test_build_test_features_computes_causally_per_engine():
    df = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 2, 2],
            "time_cycles": [1, 2, 3, 1, 2],
            "sensor_2": [10.0, 20.0, 30.0, 100.0, 90.0],
        }
    )

    featured = build_test_features(df, ["sensor_2"])

    assert "sensor_2_roll_mean_5" in featured.columns
    # engine 2's first row must not be influenced by engine 1's values
    engine_2_first = featured[(featured["unit_number"] == 2) & (featured["time_cycles"] == 1)]
    assert engine_2_first["sensor_2_roll_mean_5"].iloc[0] == 100.0
