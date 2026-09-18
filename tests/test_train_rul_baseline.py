"""Tests for src/train_rul_baseline.py.

Uses a tiny synthetic multi-engine dataset instead of the real FD001 files,
so these tests stay fast and independent of the downloaded dataset.
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.rul import add_capped_rul, add_rul_target
from src.train_rul_baseline import (
    NON_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_feature_columns,
    build_modeling_dataset,
    build_predictions_df,
    get_feature_importance,
    save_model_and_features,
    train_baseline_rf,
)


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
                    "sensor_1": 518.67,  # constant -> should be screened out
                    "sensor_2": 600.0 + cycle + rng.normal(0, 0.1),  # varying -> kept
                }
            )
    df = pd.DataFrame(rows)
    df = add_rul_target(df)
    df = add_capped_rul(df)
    return df


def test_build_feature_columns_excludes_constant_sensor():
    df = _make_train_df()

    feature_columns = build_feature_columns(df)

    assert "sensor_1" not in feature_columns
    assert "sensor_2" in feature_columns


def test_build_feature_columns_excludes_targets_and_unit_number():
    df = _make_train_df()

    feature_columns = build_feature_columns(df)

    for excluded in NON_FEATURE_COLUMNS:
        assert excluded not in feature_columns


def test_build_feature_columns_includes_time_cycles_and_settings():
    df = _make_train_df()

    feature_columns = build_feature_columns(df)

    assert "time_cycles" in feature_columns
    assert "operational_setting_1" in feature_columns


def test_build_modeling_dataset_returns_matching_lengths():
    df = _make_train_df()
    feature_columns = build_feature_columns(df)

    X, y = build_modeling_dataset(df, feature_columns)

    assert len(X) == len(df)
    assert len(y) == len(df)
    assert list(X.columns) == feature_columns
    assert y.name == TARGET_COLUMN


def test_train_baseline_rf_returns_fitted_model():
    df = _make_train_df()
    feature_columns = build_feature_columns(df)
    X, y = build_modeling_dataset(df, feature_columns)

    model = train_baseline_rf(X, y, random_state=42)

    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 200
    preds = model.predict(X)
    assert len(preds) == len(X)


def test_build_predictions_df_has_expected_columns_and_error_sign():
    val_df = pd.DataFrame(
        {
            "unit_number": [1, 1],
            "time_cycles": [1, 2],
            "rul_capped": [10, 9],
        }
    )
    y_pred = np.array([12, 5])  # over-predicts then under-predicts

    predictions = build_predictions_df(val_df, y_pred)

    assert list(predictions.columns) == [
        "unit_number",
        "time_cycles",
        "actual_rul",
        "predicted_rul",
        "error",
    ]
    assert predictions["error"].tolist() == [2, -4]


def test_get_feature_importance_returns_top_n_sorted_descending():
    df = _make_train_df()
    feature_columns = build_feature_columns(df)
    X, y = build_modeling_dataset(df, feature_columns)
    model = train_baseline_rf(X, y, random_state=42)

    importance_df = get_feature_importance(model, feature_columns, top_n=3)

    assert len(importance_df) == min(3, len(feature_columns))
    assert list(importance_df["importance"]) == sorted(
        importance_df["importance"], reverse=True
    )


def test_save_model_and_features_writes_both_files(tmp_path):
    df = _make_train_df()
    feature_columns = build_feature_columns(df)
    X, y = build_modeling_dataset(df, feature_columns)
    model = train_baseline_rf(X, y, random_state=42)

    model_path = tmp_path / "model.joblib"
    features_path = tmp_path / "features.json"

    save_model_and_features(model, feature_columns, model_path, features_path)

    assert model_path.exists()
    assert features_path.exists()
    assert json.loads(features_path.read_text()) == feature_columns
