"""Tests for src/features.py."""

import pandas as pd

from src.features import ENGINEERED_FEATURE_COLUMNS, add_engineered_features


def test_add_engineered_features_computes_expected_values():
    df = pd.DataFrame(
        {
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500],
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [10],
        }
    )

    result = add_engineered_features(df)

    assert result["temperature_difference"].iloc[0] == 10.0  # 310 - 300
    assert result["power_proxy"].iloc[0] == 1500 * 40.0
    assert result["wear_torque_interaction"].iloc[0] == 10 * 40.0


def test_add_engineered_features_does_not_mutate_input():
    df = pd.DataFrame(
        {
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500],
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [10],
        }
    )
    original_columns = list(df.columns)

    add_engineered_features(df)

    assert list(df.columns) == original_columns


def test_add_engineered_features_adds_all_expected_columns():
    df = pd.DataFrame(
        {
            "Air temperature [K]": [300.0],
            "Process temperature [K]": [310.0],
            "Rotational speed [rpm]": [1500],
            "Torque [Nm]": [40.0],
            "Tool wear [min]": [10],
        }
    )

    result = add_engineered_features(df)

    for col in ENGINEERED_FEATURE_COLUMNS:
        assert col in result.columns
