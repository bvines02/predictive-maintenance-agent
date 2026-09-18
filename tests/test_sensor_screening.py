"""Tests for src/sensor_screening.py.

Uses tiny hand-built DataFrames instead of the real dataset, so the
screening rule's correctness is proven independently of FD001 itself.
"""

import pandas as pd

from src.sensor_screening import (
    STATUS_CONSTANT,
    STATUS_KEEP,
    STATUS_NEAR_CONSTANT,
    build_cleaned_features,
    recommended_removals,
    screen_sensors,
)


def _make_df(n_rows=20):
    return pd.DataFrame(
        {
            "unit_number": [1] * n_rows,
            "time_cycles": list(range(1, n_rows + 1)),
            "operational_setting_1": [0.5] * n_rows,
            "sensor_constant": [42.0] * n_rows,
            "sensor_varying": [float(i) for i in range(n_rows)],
            "rul": list(range(n_rows - 1, -1, -1)),
            "rul_capped": list(range(n_rows - 1, -1, -1)),
        }
    )


def test_screen_sensors_detects_constant_column():
    df = _make_df()

    table = screen_sensors(df, sensor_columns=["sensor_constant", "sensor_varying"])
    row = table.loc[table["sensor"] == "sensor_constant"].iloc[0]

    assert row["status"] == STATUS_CONSTANT
    assert row["std"] == 0.0
    assert row["unique_values"] == 1


def test_screen_sensors_retains_varying_column():
    df = _make_df()

    table = screen_sensors(df, sensor_columns=["sensor_constant", "sensor_varying"])
    row = table.loc[table["sensor"] == "sensor_varying"].iloc[0]

    assert row["status"] == STATUS_KEEP
    assert row["std"] > 0
    assert row["unique_values"] == len(df)


def test_screen_sensors_flags_near_constant_column():
    n_rows = 20
    df = _make_df(n_rows)
    # Only 2 distinct values across the whole column - near-constant, not
    # perfectly constant.
    df["sensor_near_constant"] = [1.0 if i % 10 == 0 else 1.001 for i in range(n_rows)]

    table = screen_sensors(df, sensor_columns=["sensor_near_constant"])
    row = table.loc[table["sensor"] == "sensor_near_constant"].iloc[0]

    assert row["unique_values"] == 2
    assert row["status"] == STATUS_NEAR_CONSTANT


def test_screen_sensors_stats_are_correct():
    df = pd.DataFrame({"sensor_varying": [1.0, 2.0, 3.0, 4.0, 5.0]})

    table = screen_sensors(df, sensor_columns=["sensor_varying"])
    row = table.loc[table["sensor"] == "sensor_varying"].iloc[0]

    assert row["count"] == 5
    assert row["mean"] == 3.0
    assert row["min"] == 1.0
    assert row["max"] == 5.0
    assert row["range"] == 4.0
    assert row["unique_values"] == 5
    assert row["pct_unique"] == 100.0


def test_recommended_removals_excludes_keep_status():
    df = _make_df()
    table = screen_sensors(df, sensor_columns=["sensor_constant", "sensor_varying"])

    removals = recommended_removals(table)

    assert removals == ["sensor_constant"]


def test_build_cleaned_features_preserves_row_count():
    df = _make_df()

    cleaned = build_cleaned_features(df, sensors_to_remove=["sensor_constant"])

    assert len(cleaned) == len(df)


def test_build_cleaned_features_drops_recommended_sensors():
    df = _make_df()

    cleaned = build_cleaned_features(df, sensors_to_remove=["sensor_constant"])

    assert "sensor_constant" not in cleaned.columns
    assert "sensor_varying" in cleaned.columns


def test_build_cleaned_features_excludes_targets_and_unit_number():
    df = _make_df()

    cleaned = build_cleaned_features(df, sensors_to_remove=[])

    assert "unit_number" not in cleaned.columns
    assert "rul" not in cleaned.columns
    assert "rul_capped" not in cleaned.columns


def test_build_cleaned_features_preserves_time_cycles_and_operational_settings():
    df = _make_df()

    cleaned = build_cleaned_features(df, sensors_to_remove=[])

    assert "time_cycles" in cleaned.columns
    assert "operational_setting_1" in cleaned.columns


def test_build_cleaned_features_does_not_mutate_input():
    df = _make_df()

    build_cleaned_features(df, sensors_to_remove=["sensor_constant"])

    assert "sensor_constant" in df.columns
