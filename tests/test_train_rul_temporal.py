"""Tests for src/train_rul_temporal.py.

Uses a tiny synthetic multi-engine dataset instead of the real FD001 files,
so these tests stay fast and independent of the downloaded dataset.
"""

import json

import numpy as np
import pandas as pd

from src.rul import add_capped_rul, add_rul_target
from src.train_rul_baseline import build_feature_columns, get_kept_sensor_columns
from src.train_rul_temporal import (
    build_temporal_feature_columns,
    classify_feature,
    compare_models,
    get_feature_importance_by_category,
    improvement,
    save_temporal_model,
)
from src.cmapss_features import add_rolling_features
from src.engine_split import split_dataframe_by_unit, split_units
from src.train_rul_baseline import build_modeling_dataset, train_baseline_rf


def _make_train_df(n_units=6, rows_per_unit=20):
    rng = np.random.default_rng(42)
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, rows_per_unit + 1):
            rows.append(
                {
                    "unit_number": unit,
                    "time_cycles": cycle,
                    "operational_setting_1": 0.5,
                    "operational_setting_2": 0.2,
                    "operational_setting_3": 100.0,
                    "sensor_1": 518.67,  # constant -> screened out
                    "sensor_2": 600.0 + cycle + rng.normal(0, 0.1),  # varying -> kept
                }
            )
    df = pd.DataFrame(rows)
    df = add_rul_target(df)
    df = add_capped_rul(df)
    return df


def test_build_temporal_feature_columns_includes_baseline_and_rolling_features():
    df = _make_train_df()
    kept_sensors = get_kept_sensor_columns(df)
    baseline_columns = build_feature_columns(df)

    temporal_columns = build_temporal_feature_columns(df, kept_sensors)

    for column in baseline_columns:
        assert column in temporal_columns
    assert "sensor_2_roll_mean_5" in temporal_columns
    assert "sensor_2_roll_std_10" in temporal_columns
    assert "sensor_2_delta_1" in temporal_columns
    assert "sensor_2_trend_5" in temporal_columns
    # The constant sensor was screened out - no temporal features for it either.
    assert not any("sensor_1_" in column for column in temporal_columns)


def test_build_temporal_feature_columns_excludes_targets():
    df = _make_train_df()
    kept_sensors = get_kept_sensor_columns(df)

    temporal_columns = build_temporal_feature_columns(df, kept_sensors)

    assert "rul" not in temporal_columns
    assert "rul_capped" not in temporal_columns
    assert "unit_number" not in temporal_columns


def test_classify_feature_labels_each_category_correctly():
    assert classify_feature("sensor_2_roll_mean_5") == "rolling_mean"
    assert classify_feature("sensor_2_roll_std_10") == "rolling_std"
    assert classify_feature("sensor_2_delta_1") == "delta"
    assert classify_feature("sensor_2_trend_5") == "trend"
    assert classify_feature("sensor_2") == "raw"
    assert classify_feature("time_cycles") == "raw"


def test_get_feature_importance_by_category_labels_match_features():
    df = _make_train_df()
    kept_sensors = get_kept_sensor_columns(df)
    feature_columns = build_temporal_feature_columns(df, kept_sensors)
    featured_df = add_rolling_features(df, kept_sensors)
    X, y = build_modeling_dataset(featured_df, feature_columns)
    model = train_baseline_rf(X, y, random_state=42)

    importance_df = get_feature_importance_by_category(model, feature_columns, top_n=5)

    assert len(importance_df) == 5
    for _, row in importance_df.iterrows():
        assert row["category"] == classify_feature(row["feature"])
    assert list(importance_df["importance"]) == sorted(importance_df["importance"], reverse=True)


def test_compare_models_produces_one_row_per_model():
    metrics_a = {"train": {"mae": 4.0, "rmse": 6.0}, "val": {"mae": 10.0, "rmse": 15.0}}
    metrics_b = {"train": {"mae": 3.0, "rmse": 5.0}, "val": {"mae": 9.0, "rmse": 14.0}}

    comparison = compare_models("Baseline RF", metrics_a, "Temporal-feature RF", metrics_b)

    assert list(comparison["model"]) == ["Baseline RF", "Temporal-feature RF"]
    assert comparison.loc[0, "val_mae"] == 10.0
    assert comparison.loc[1, "val_mae"] == 9.0


def test_improvement_reports_negative_for_a_lower_error():
    absolute_change, percent_change = improvement(baseline_value=10.0, new_value=8.0)

    assert absolute_change == -2.0
    assert np.isclose(percent_change, -20.0)


def test_improvement_reports_positive_when_new_model_is_worse():
    absolute_change, percent_change = improvement(baseline_value=10.0, new_value=12.0)

    assert absolute_change == 2.0
    assert np.isclose(percent_change, 20.0)


def test_temporal_and_baseline_split_use_the_same_engines():
    """Re-running split_units with the same seed must reproduce Step 5's split."""
    df = _make_train_df()

    train_units_1, val_units_1 = split_units(df["unit_number"], test_size=0.2, random_state=42)
    train_units_2, val_units_2 = split_units(df["unit_number"], test_size=0.2, random_state=42)

    assert set(train_units_1) == set(train_units_2)
    assert set(val_units_1) == set(val_units_2)


def test_save_temporal_model_writes_both_files(tmp_path):
    df = _make_train_df()
    kept_sensors = get_kept_sensor_columns(df)
    feature_columns = build_temporal_feature_columns(df, kept_sensors)
    featured_df = add_rolling_features(df, kept_sensors)
    X, y = build_modeling_dataset(featured_df, feature_columns)
    model = train_baseline_rf(X, y, random_state=42)

    model_path = tmp_path / "temporal.joblib"
    features_path = tmp_path / "temporal_features.json"

    save_temporal_model(model, feature_columns, model_path, features_path)

    assert model_path.exists()
    assert features_path.exists()
    assert json.loads(features_path.read_text()) == feature_columns
