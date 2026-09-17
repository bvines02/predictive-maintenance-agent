"""Stage 14c - Remaining Useful Life (RUL) label construction.

Responsible for:
- Turning a run-to-failure DataFrame (like train_FD001) into a supervised
  learning dataset by adding a "rul" column
- Capping RUL at a configurable ceiling for the baseline model
- Validating that the resulting labels make physical sense

Not responsible for (see V2 principles in CLAUDE.md - these come later):
- Feature engineering
- Model training

RUL can only be calculated this way for data that runs all the way to
failure. train_FD001 does; test_FD001 does not (each test engine's history
is deliberately cut off before failure - that's the whole point of the
prediction task). Do not call add_rul_target on the test set.
"""

import pandas as pd

# The baseline cap, in cycles. Deliberately a named constant, not a literal
# scattered through the code - see README/CLAUDE.md for why 125 is a
# modelling choice to revisit later, not a fixed physical fact.
DEFAULT_RUL_CAP = 125


def add_rul_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add a "rul" column: cycles remaining until failure, per engine.

    For each unit_number, RUL at a given row is that engine's maximum
    observed time_cycles minus the row's own time_cycles. The last row of
    every engine therefore has RUL 0 (that's the failure point).

    Returns a new DataFrame - the input df is not modified. This matters
    because a function that mutates its input silently is easy to call
    twice by accident (e.g. re-running a notebook cell), which either
    raises a confusing error or silently reprocesses already-processed data.
    """
    result = df.copy()
    max_cycle_per_unit = result.groupby("unit_number")["time_cycles"].transform("max")
    result["rul"] = max_cycle_per_unit - result["time_cycles"]
    return result


def validate_rul(df: pd.DataFrame) -> None:
    """Check that a DataFrame with an "rul" column has physically sensible labels.

    Raises ValueError with a specific message on the first check that fails.
    """
    if "rul" not in df.columns:
        raise ValueError("Expected an 'rul' column - did you call add_rul_target?")

    if df["rul"].isna().any():
        raise ValueError("Found missing rul values.")

    if (df["rul"] < 0).any():
        raise ValueError("Found negative rul values - RUL cannot be negative.")

    for unit_number, engine_rows in df.groupby("unit_number"):
        engine_rows = engine_rows.sort_values("time_cycles")
        rul_values = engine_rows["rul"].to_numpy()

        if rul_values[-1] != 0:
            raise ValueError(
                f"Unit {unit_number}: final cycle has rul={rul_values[-1]}, expected 0."
            )

        # RUL must count down by exactly 1 each cycle - anything else means
        # either a gap in the source cycles or a mistake in the RUL formula.
        rul_steps = rul_values[:-1] - rul_values[1:]
        if not (rul_steps == 1).all():
            raise ValueError(
                f"Unit {unit_number}: rul does not decrease by exactly 1 between "
                "consecutive cycles."
            )


def add_capped_rul(df: pd.DataFrame, cap: int = DEFAULT_RUL_CAP) -> pd.DataFrame:
    """Add a "rul_capped" column: rul, ceilinged at `cap`.

    capped_rul = min(rul, cap)

    Leaves the original "rul" column untouched, so both the true and capped
    targets remain available side by side. Returns a new DataFrame, same
    no-mutation reasoning as add_rul_target.
    """
    if "rul" not in df.columns:
        raise ValueError("Expected an 'rul' column - did you call add_rul_target?")

    result = df.copy()
    result["rul_capped"] = result["rul"].clip(upper=cap)
    return result


def validate_capped_rul(df: pd.DataFrame, cap: int = DEFAULT_RUL_CAP) -> None:
    """Check that "rul_capped" was derived correctly from "rul".

    Raises ValueError with a specific message on the first check that fails.
    """
    if "rul_capped" not in df.columns:
        raise ValueError("Expected a 'rul_capped' column - did you call add_capped_rul?")

    if df["rul_capped"].isna().any():
        raise ValueError("Found missing rul_capped values.")

    if (df["rul_capped"] < 0).any():
        raise ValueError("Found negative rul_capped values - RUL cannot be negative.")

    if (df["rul_capped"] > cap).any():
        raise ValueError(f"Found rul_capped values above the configured cap ({cap}).")

    below_or_at_cap = df["rul"] <= cap
    if not (df.loc[below_or_at_cap, "rul_capped"] == df.loc[below_or_at_cap, "rul"]).all():
        raise ValueError("Found rows with rul <= cap where rul_capped != rul.")

    above_cap = df["rul"] > cap
    if not (df.loc[above_cap, "rul_capped"] == cap).all():
        raise ValueError(f"Found rows with rul > cap where rul_capped != {cap}.")


if __name__ == "__main__":
    from src.cmapss_loader import load_train_fd001, validate_cmapss_dataframe

    train_df = load_train_fd001()
    validate_cmapss_dataframe(train_df, "train_FD001")

    train_with_rul = add_rul_target(train_df)
    validate_rul(train_with_rul)

    if len(train_with_rul) != len(train_df):
        raise AssertionError("Row count changed while adding the rul column.")

    engine_1 = train_with_rul[train_with_rul["unit_number"] == 1]

    print("First rows for engine 1:")
    print(engine_1[["unit_number", "time_cycles", "rul"]].head())

    print("\nLast 10 rows for engine 1:")
    print(engine_1[["unit_number", "time_cycles", "rul"]].tail(10))

    print(f"\nMinimum RUL: {train_with_rul['rul'].min()}")
    print(f"Maximum RUL: {train_with_rul['rul'].max()}")

    print("\nDescriptive statistics for rul:")
    print(train_with_rul["rul"].describe())

    lifecycle_lengths = train_with_rul.groupby("unit_number")["time_cycles"].max()
    print("\nLifecycle length (max cycle) per engine - summary:")
    print(lifecycle_lengths.describe())

    train_with_capped_rul = add_capped_rul(train_with_rul)
    validate_capped_rul(train_with_capped_rul)

    if len(train_with_capped_rul) != len(train_df):
        raise AssertionError("Row count changed while adding the rul_capped column.")

    engine_1 = train_with_capped_rul[train_with_capped_rul["unit_number"] == 1]

    print(f"\n{'=' * 60}")
    print(f"Capped RUL (cap={DEFAULT_RUL_CAP})")
    print(f"{'=' * 60}")

    print("\nRaw vs. capped RUL for engine 1 (every 20th cycle):")
    print(engine_1[["unit_number", "time_cycles", "rul", "rul_capped"]].iloc[::20])

    print("\nFirst 10 rows (whole dataset):")
    print(train_with_capped_rul[["unit_number", "time_cycles", "rul", "rul_capped"]].head(10))

    print("\nLast 10 rows (whole dataset):")
    print(train_with_capped_rul[["unit_number", "time_cycles", "rul", "rul_capped"]].tail(10))

    print(f"\nMaximum raw rul: {train_with_capped_rul['rul'].max()}")
    print(f"Maximum capped rul: {train_with_capped_rul['rul_capped'].max()}")

    n_affected = (train_with_capped_rul["rul"] > DEFAULT_RUL_CAP).sum()
    pct_affected = 100 * n_affected / len(train_with_capped_rul)
    print(
        f"\nRows affected by the cap: {n_affected} of {len(train_with_capped_rul)} "
        f"({pct_affected:.1f}%)"
    )

    print("\nRaw rul descriptive statistics:")
    print(train_with_capped_rul["rul"].describe())

    print("\nCapped rul descriptive statistics:")
    print(train_with_capped_rul["rul_capped"].describe())
