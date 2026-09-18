"""Stage 14d (screening) - Sensor screening for FD001.

Responsible for:
- Computing descriptive statistics for each raw sensor column
- Classifying each sensor as "keep", "constant" or "near_constant" using a
  simple, explicit, explainable rule (see NEAR_CONSTANT_MAX_UNIQUE below)
- Producing a configurable list of sensors recommended for removal
- Building a cleaned *feature* DataFrame that drops the recommended sensors,
  without touching the raw DataFrame

Not responsible for (see V2 principles in CLAUDE.md - these come later):
- Training a model
- Normalising features
- Rolling/trend feature engineering
- Feature importance analysis

This screening must only ever be run on the TRAINING set. See the "why"
notes in the module docstring for `run_screening_report` and the project
explanation for why the test set must not influence this decision.
"""

import pandas as pd

from src.cmapss_loader import SENSOR_COLUMNS

# --- The near-constant rule --------------------------------------------------
#
# A sensor is "constant" if it takes exactly one value across every row in
# training data. That is unambiguous - std is exactly 0, and no model can
# extract information from a column that never changes.
#
# A sensor is "near_constant" if it takes very few distinct values, even
# though it is not perfectly constant. The threshold below is deliberately
# based on COUNT OF UNIQUE VALUES, not on a ratio like std/range or
# std/mean. Those ratio-based measures were tried against the real FD001
# training data first and rejected: std/range clusters around 0.10-0.16 for
# EVERY sensor in this dataset, constant or not, because it mostly reflects
# each sensor's own distribution shape rather than how much real information
# it carries. A sensor with only 2 distinct values (sensor_6, in FD001) and
# a sensor with 6000+ distinct values (sensor_9) end up with similar
# std/range ratios, so that measure cannot tell them apart here.
# Unique-value count does not have this problem: a sensor that only ever
# reports 2-5 discrete readings is not tracking a continuously changing
# physical quantity, regardless of its scale.
NEAR_CONSTANT_MAX_UNIQUE = 5

STATUS_KEEP = "keep"
STATUS_CONSTANT = "constant"
STATUS_NEAR_CONSTANT = "near_constant"


def _classify(nunique: int) -> str:
    if nunique <= 1:
        return STATUS_CONSTANT
    if nunique <= NEAR_CONSTANT_MAX_UNIQUE:
        return STATUS_NEAR_CONSTANT
    return STATUS_KEEP


def screen_sensors(
    df: pd.DataFrame, sensor_columns: list[str] = SENSOR_COLUMNS
) -> pd.DataFrame:
    """Compute per-sensor descriptive statistics and a keep/constant/near_constant status.

    Returns one row per sensor column, with:
        sensor, count, mean, std, min, max, range, unique_values, pct_unique, status
    """
    rows = []
    for sensor in sensor_columns:
        series = df[sensor]
        count = int(series.count())
        minimum = float(series.min())
        maximum = float(series.max())
        unique_values = int(series.nunique())

        rows.append(
            {
                "sensor": sensor,
                "count": count,
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": minimum,
                "max": maximum,
                "range": maximum - minimum,
                "unique_values": unique_values,
                "pct_unique": 100 * unique_values / count if count else 0.0,
                "status": _classify(unique_values),
            }
        )

    return pd.DataFrame(rows)


def sensors_with_status(screening_table: pd.DataFrame, status: str) -> list[str]:
    """List sensor names in `screening_table` matching a given status."""
    return screening_table.loc[screening_table["status"] == status, "sensor"].tolist()


def recommended_removals(screening_table: pd.DataFrame) -> list[str]:
    """Sensors recommended for removal: anything not classified "keep".

    Returned as a plain list, deliberately separate from the constant /
    near_constant data behind it, so a caller can override this list (e.g.
    keep a near_constant sensor anyway) without re-deriving the whole table -
    see CLAUDE.md's warning against blindly deleting features.
    """
    return screening_table.loc[screening_table["status"] != STATUS_KEEP, "sensor"].tolist()


def build_cleaned_features(df: pd.DataFrame, sensors_to_remove: list[str]) -> pd.DataFrame:
    """Build a cleaned FEATURE DataFrame - not a replacement for the raw dataset.

    - Drops `sensors_to_remove` from the sensor columns.
    - Deliberately excludes unit_number, rul and rul_capped, even if present
      in `df` - those are identifiers/targets, never model inputs.
    - Keeps time_cycles and the operational settings, since those are still
      useful features at this stage (V2 principles say to preserve them for
      now).
    - Does not modify `df` - returns a new DataFrame, same no-mutation
      reasoning as src/rul.py.
    """
    excluded = {"unit_number", "rul", "rul_capped"} | set(sensors_to_remove)
    keep_columns = [col for col in df.columns if col not in excluded]
    return df[keep_columns].copy()


def print_screening_report(screening_table: pd.DataFrame) -> None:
    """Print the screening table and a summary of what was flagged."""
    print(f"\n{'=' * 90}")
    print("Sensor screening report (train_FD001)")
    print(f"{'=' * 90}")
    print(
        screening_table.to_string(
            index=False,
            columns=["sensor", "std", "min", "max", "range", "unique_values", "status"],
            float_format=lambda v: f"{v:.6f}",
        )
    )

    constant = sensors_with_status(screening_table, STATUS_CONSTANT)
    near_constant = sensors_with_status(screening_table, STATUS_NEAR_CONSTANT)
    keep = sensors_with_status(screening_table, STATUS_KEEP)

    print(f"\nConstant sensors ({len(constant)}): {constant}")
    print(f"Near-constant sensors ({len(near_constant)}): {near_constant}")
    print(f"Sensors to keep ({len(keep)}): {keep}")

    n_original = len(screening_table)
    n_remaining = len(keep)
    print(f"\nOriginal sensor features: {n_original}")
    print(f"Remaining after screening: {n_remaining}")


if __name__ == "__main__":
    from src.cmapss_loader import load_train_fd001, validate_cmapss_dataframe

    train_df = load_train_fd001()
    validate_cmapss_dataframe(train_df, "train_FD001")

    screening_table = screen_sensors(train_df)
    print_screening_report(screening_table)

    to_remove = recommended_removals(screening_table)
    cleaned = build_cleaned_features(train_df, to_remove)

    print(f"\n{'=' * 90}")
    print("Cleaned feature DataFrame")
    print(f"{'=' * 90}")
    print(f"Columns: {list(cleaned.columns)}")
    print(f"Row count unchanged: {len(cleaned) == len(train_df)}")
