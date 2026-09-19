"""Tests for src/cmapss_features.py (causal rolling/trend features)."""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.cmapss_features import add_rolling_features, rolling_feature_names


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1, 2, 2],
            "time_cycles": [1, 2, 3, 4, 1, 2],
            "sensor_2": [10.0, 12.0, 14.0, 18.0, 100.0, 104.0],
        }
    )


def test_rolling_window_resets_between_engines():
    result = add_rolling_features(_history(), ["sensor_2"], windows=(3,))

    engine_2 = result[result["unit_number"] == 2]
    # If engine 1's history leaked in, engine 2's first-cycle mean would not
    # simply equal its own first (and only, so far) reading.
    assert engine_2["sensor_2_roll_mean_3"].tolist() == [100.0, 102.0]
    assert engine_2["sensor_2_delta_1"].tolist() == [0.0, 4.0]


def test_delta_does_not_compare_across_engine_boundary():
    result = add_rolling_features(_history(), ["sensor_2"], windows=(3,))

    # Engine 2's first row must not be diffed against engine 1's last row
    # (18.0), which is what a naive un-grouped .diff() would do.
    engine_2_first_delta = result.loc[result["unit_number"] == 2, "sensor_2_delta_1"].iloc[0]
    assert engine_2_first_delta == 0.0


def test_rolling_window_uses_only_trailing_cycles():
    result = add_rolling_features(_history(), ["sensor_2"], windows=(3,))
    engine_1 = result[result["unit_number"] == 1]

    assert engine_1["sensor_2_roll_mean_3"].tolist() == [10.0, 11.0, 12.0, 44 / 3]
    assert engine_1.iloc[0]["sensor_2_roll_std_3"] == 0.0


def test_no_future_observations_are_used():
    """The central leakage test: future rows cannot alter earlier features."""
    full_history = _history().query("unit_number == 1").reset_index(drop=True)
    prefix = full_history.iloc[:3].copy()

    features_from_full = add_rolling_features(full_history, ["sensor_2"], windows=(3,))
    features_from_prefix = add_rolling_features(prefix, ["sensor_2"], windows=(3,))

    pdt.assert_frame_equal(
        features_from_full.iloc[:3].reset_index(drop=True),
        features_from_prefix.reset_index(drop=True),
    )


def test_row_count_is_preserved():
    df = _history()

    result = add_rolling_features(df, ["sensor_2"], windows=(3, 5))

    assert len(result) == len(df)


def test_input_is_not_modified():
    df = _history()

    add_rolling_features(df, ["sensor_2"])

    assert list(df.columns) == ["unit_number", "time_cycles", "sensor_2"]


def test_target_columns_are_never_produced_as_features():
    df = _history()
    df["rul"] = [3, 2, 1, 0, 1, 0]
    df["rul_capped"] = df["rul"]

    result = add_rolling_features(df, ["sensor_2"], windows=(3,))

    added_columns = [c for c in result.columns if c not in df.columns]
    assert not any("rul" in column for column in added_columns)


def test_adds_both_windows_mean_and_std():
    result = add_rolling_features(_history(), ["sensor_2"], windows=(2, 3))

    for expected in [
        "sensor_2_roll_mean_2",
        "sensor_2_roll_std_2",
        "sensor_2_roll_mean_3",
        "sensor_2_roll_std_3",
    ]:
        assert expected in result.columns


def test_trend_is_zero_on_first_cycle_of_each_engine():
    result = add_rolling_features(_history(), ["sensor_2"], windows=(3,), trend_window=3)

    first_cycle_trends = result.groupby("unit_number").first()["sensor_2_trend_3"]
    assert (first_cycle_trends == 0.0).all()


def test_trend_is_positive_for_a_steadily_increasing_sensor():
    df = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1, 1],
            "time_cycles": [1, 2, 3, 4, 5],
            "sensor_2": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    result = add_rolling_features(df, ["sensor_2"], windows=(5,), trend_window=5)

    assert np.isclose(result["sensor_2_trend_5"].iloc[-1], 10.0)


def test_trend_is_negative_for_a_steadily_decreasing_sensor():
    df = pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1, 1],
            "time_cycles": [1, 2, 3, 4, 5],
            "sensor_2": [50.0, 40.0, 30.0, 20.0, 10.0],
        }
    )

    result = add_rolling_features(df, ["sensor_2"], windows=(5,), trend_window=5)

    assert np.isclose(result["sensor_2_trend_5"].iloc[-1], -10.0)


def test_unsorted_cycles_are_rejected():
    df = pd.DataFrame(
        {"unit_number": [1, 1], "time_cycles": [2, 1], "sensor_2": [12.0, 10.0]}
    )

    with pytest.raises(ValueError, match="in increasing order"):
        add_rolling_features(df, ["sensor_2"])


def test_invalid_window_is_rejected():
    with pytest.raises(ValueError, match="at least 1"):
        add_rolling_features(_history(), ["sensor_2"], windows=(0,))


def test_missing_sensor_is_rejected():
    with pytest.raises(ValueError, match="sensor_99"):
        add_rolling_features(_history(), ["sensor_99"])


def test_rolling_feature_names_matches_columns_actually_added():
    df = _history()
    sensor_columns = ["sensor_2"]

    result = add_rolling_features(df, sensor_columns, windows=(3, 5), trend_window=5)
    added_columns = [c for c in result.columns if c not in df.columns]

    assert added_columns == rolling_feature_names(sensor_columns, windows=(3, 5), trend_window=5)
