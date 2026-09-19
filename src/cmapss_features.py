"""Stage 14g - causal time-series (rolling) features for C-MAPSS FD001.

Responsible for:
- Turning each engine's raw sensor trajectory into "recent history" features:
  rolling mean, rolling standard deviation, one-cycle delta, and a short-term
  trend (slope)
- Guaranteeing every feature is calculated PER ENGINE and PER CYCLE using
  only that engine's own current and earlier cycles

Not responsible for:
- Deciding which sensors are worth featuring (that's Step 4 / sensor_screening.py -
  this module only transforms whatever sensor_columns it's given)
- Model training

Why per-engine, causal-only calculation matters here specifically:
- Rolling windows are computed after grouping by unit_number, so a window
  never mixes cycles from two different engines - engine 2's history has
  nothing to do with engine 1's.
- pandas' `rolling()` and `diff()` are both trailing by construction: the
  value at row t is computed from rows t, t-1, ..., never t+1. As long as
  each engine's rows are sorted by time_cycles first (enforced below), this
  guarantees no cycle ever sees its own future. This is checked directly in
  tests/test_cmapss_features.py by proving a feature's value does not
  change when later rows are removed from the input.
"""

import numpy as np
import pandas as pd

DEFAULT_WINDOWS = (5, 10)
DEFAULT_TREND_WINDOW = 5


def _validate_inputs(df: pd.DataFrame, sensor_columns: list[str], windows, trend_window: int) -> None:
    required = {"unit_number", "time_cycles", *sensor_columns}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if any(window < 1 for window in windows):
        raise ValueError("All windows must be at least 1.")

    if trend_window < 1:
        raise ValueError("trend_window must be at least 1.")

    for unit_number, engine_rows in df.groupby("unit_number", sort=False):
        if not engine_rows["time_cycles"].is_monotonic_increasing:
            raise ValueError(
                f"Unit {unit_number}: time_cycles must be in increasing order "
                "before time-series features are calculated."
            )


def _slope(values: np.ndarray) -> float:
    """Least-squares slope of `values` against their position (0, 1, 2, ...).

    A single point has no slope, so that case returns 0.0 (flat/unknown)
    rather than NaN - see the trend_window docstring for why this trade-off
    was chosen over dropping early rows.
    """
    n = len(values)
    if n < 2:
        return 0.0
    return float(np.polyfit(np.arange(n), values, 1)[0])


def rolling_feature_names(
    sensor_columns: list[str],
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    trend_window: int = DEFAULT_TREND_WINDOW,
) -> list[str]:
    """The exact, ordered list of column names `add_rolling_features` adds.

    Kept as its own function (rather than re-deriving it from a fitted
    model's columns) so callers can build the full feature list - and save
    it - before or without ever calling add_rolling_features.
    """
    names = []
    for sensor in sensor_columns:
        for window in windows:
            names.append(f"{sensor}_roll_mean_{window}")
            names.append(f"{sensor}_roll_std_{window}")
        names.append(f"{sensor}_delta_1")
        names.append(f"{sensor}_trend_{trend_window}")
    return names


def add_rolling_features(
    df: pd.DataFrame,
    sensor_columns: list[str],
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    trend_window: int = DEFAULT_TREND_WINDOW,
) -> pd.DataFrame:
    """Add rolling mean/std (per window), a 1-cycle delta, and a trend slope.

    For each sensor in `sensor_columns`, adds:
        {sensor}_roll_mean_{w}  - mean of the trailing `w` cycles (this one included)
        {sensor}_roll_std_{w}   - std of the same trailing window
        {sensor}_delta_1        - value at cycle t minus value at cycle t-1
        {sensor}_trend_{trend_window} - slope of the trailing `trend_window` cycles

    Early-life handling (fewer than `w` prior cycles available):
    Uses `min_periods=1`, so early rows compute the mean/std/slope over
    however many cycles actually exist so far, rather than producing NaN.
    This is the simplest option that preserves every row - the alternative
    (drop rows until the window is full) would throw away a meaningful
    fraction of every engine's most healthy, early-life cycles, which we
    don't want given how few engines FD001 has. The trade-off: an early
    "rolling mean over 5" for cycle 2 is really a mean of 2 points, not 5 -
    it's a lower-confidence estimate, not a wrong one.
    delta_1 and the trend for the very first cycle of an engine have no
    prior cycle at all, so they default to 0.0 (no known change yet),
    rather than NaN, for the same row-preservation reason.

    Trend implementation: an ordinary least-squares slope over the trailing
    `trend_window` cycles, via `numpy.polyfit`. The simpler alternative -
    (value_now - value_n_ago) / n - was considered and rejected because it
    only looks at the window's two endpoints, so a single noisy reading at
    either end swings the whole trend estimate. The regression slope uses
    every point in the window, so a lone noisy cycle influences it far
    less. The cost is that a slope is more computation than a subtraction -
    negligible at FD001's scale, but worth knowing if this were applied to
    a much larger fleet.

    Calculated independently per unit_number (see module docstring for why
    this can't leak across engines or across cycles). Returns a new
    DataFrame, sorted by unit_number then time_cycles - the input df is not
    modified.
    """
    windows = tuple(windows)
    _validate_inputs(df, sensor_columns, windows, trend_window)

    result = df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True).copy()
    grouped = result.groupby("unit_number", sort=False)

    for sensor in sensor_columns:
        for window in windows:
            result[f"{sensor}_roll_mean_{window}"] = grouped[sensor].transform(
                lambda values, w=window: values.rolling(window=w, min_periods=1).mean()
            )
            result[f"{sensor}_roll_std_{window}"] = grouped[sensor].transform(
                lambda values, w=window: values.rolling(window=w, min_periods=1).std(ddof=0)
            )

        result[f"{sensor}_delta_1"] = grouped[sensor].diff().fillna(0.0)

        result[f"{sensor}_trend_{trend_window}"] = grouped[sensor].transform(
            lambda values: values.rolling(window=trend_window, min_periods=1).apply(
                _slope, raw=True
            )
        )

    return result


if __name__ == "__main__":
    from src.cmapss_loader import load_train_fd001, validate_cmapss_dataframe
    from src.sensor_screening import STATUS_KEEP, screen_sensors, sensors_with_status

    train_df = load_train_fd001()
    validate_cmapss_dataframe(train_df, "train_FD001")

    screening = screen_sensors(train_df)
    kept_sensors = sensors_with_status(screening, STATUS_KEEP)

    featured = add_rolling_features(train_df, kept_sensors)
    added_columns = [column for column in featured.columns if column not in train_df.columns]

    print(f"Input rows: {len(train_df)}")
    print(f"Sensors used: {len(kept_sensors)}")
    print(f"Time-series features added: {len(added_columns)}")
    print(f"Output shape: {featured.shape}")

    print("\nWorked example - engine 1, sensor_11, cycles 1-10:")
    example_columns = [
        "unit_number",
        "time_cycles",
        "sensor_11",
        "sensor_11_roll_mean_5",
        "sensor_11_roll_std_5",
        "sensor_11_delta_1",
        "sensor_11_trend_5",
    ]
    engine_1 = featured[featured["unit_number"] == 1]
    print(engine_1[example_columns].head(10).to_string(index=False))
