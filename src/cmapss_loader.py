"""Stage 14b - C-MAPSS data loading (FD001 only).

Responsible for:
- Reading the raw, whitespace-separated, header-less NASA text files
- Assigning the correct column names
- Validating that the loaded data looks structurally sound
- Printing an exploratory summary

Not responsible for (see V2 principles in CLAUDE.md - these come later):
- Calculating RUL
- Feature engineering
- Removing/selecting sensors
- Training a model
- Normalising data

Each row in these files is one operating cycle for one engine unit. Rows are
ordered within a unit and dependent on each other - this is time-series
degradation data, not independent samples.
"""

from pathlib import Path

import pandas as pd

from src.config import (
    CMAPSS_FD001_RUL_FILE,
    CMAPSS_FD001_TEST_FILE,
    CMAPSS_FD001_TRAIN_FILE,
)

# Column order as documented by NASA's C-MAPSS readme: unit id, cycle count,
# 3 operational settings, then 21 sensor readings. The raw files have no
# header row, so this list is the only place this mapping is written down.
OPERATIONAL_SETTING_COLUMNS = [
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
COLUMN_NAMES = ["unit_number", "time_cycles"] + OPERATIONAL_SETTING_COLUMNS + SENSOR_COLUMNS
EXPECTED_COLUMN_COUNT = len(COLUMN_NAMES)  # 26


def _read_raw_cmapss_file(file_path: str | Path) -> pd.DataFrame:
    """Read one C-MAPSS text file and assign the standard column names.

    Handles the raw format's quirks:
    - whitespace-separated (not comma-separated), with variable-width gaps
    - a trailing space at the end of every line

    `sep=r"\\s+"` treats any run of whitespace as a single separator, which
    avoids the classic C-MAPSS bug of phantom empty columns from that
    trailing whitespace. As a safety net, any column that ends up entirely
    empty is dropped before the column names are assigned - so a slightly
    different whitespace format fails loudly (wrong column count) instead
    of silently mislabelling sensors.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"C-MAPSS file not found: {file_path}\n"
            "Download the NASA C-MAPSS dataset and place FD001 files under "
            "data/raw/cmapss/FD001/ - see the 'Dataset' section in README.md."
        )

    df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis="columns", how="all")

    if df.shape[1] != EXPECTED_COLUMN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COLUMN_COUNT} columns after parsing {file_path}, "
            f"got {df.shape[1]}. The raw file's format may not match what this "
            "loader expects."
        )

    df.columns = COLUMN_NAMES
    return df


def load_train_fd001(file_path: str | Path = CMAPSS_FD001_TRAIN_FILE) -> pd.DataFrame:
    """Load the FD001 training set: full run-to-failure histories for each engine."""
    return _read_raw_cmapss_file(file_path)


def load_test_fd001(file_path: str | Path = CMAPSS_FD001_TEST_FILE) -> pd.DataFrame:
    """Load the FD001 test set: truncated histories (each engine stops before failure)."""
    return _read_raw_cmapss_file(file_path)


def load_rul_fd001(file_path: str | Path = CMAPSS_FD001_RUL_FILE) -> pd.Series:
    """Load the true remaining-useful-life values for the test set's engines.

    One value per engine, in the same order as the engines appear in the test
    set (engine 1's true RUL first, engine 2's second, and so on) - this file
    has no unit_number column, so order is the only thing that ties a value
    back to an engine.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"C-MAPSS RUL file not found: {file_path}")

    rul = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
    rul = rul.dropna(axis="columns", how="all")

    if rul.shape[1] != 1:
        raise ValueError(
            f"Expected exactly 1 column in {file_path}, got {rul.shape[1]}."
        )

    return rul.iloc[:, 0].rename("true_rul")


def validate_cmapss_dataframe(df: pd.DataFrame, name: str) -> None:
    """Run basic structural checks on a loaded train/test DataFrame.

    Raises ValueError with a specific message on the first check that fails,
    so a bad file is easy to diagnose rather than producing a confusing
    downstream error much later.
    """
    if df.shape[1] != EXPECTED_COLUMN_COUNT:
        raise ValueError(
            f"{name}: expected {EXPECTED_COLUMN_COUNT} columns, got {df.shape[1]}."
        )

    if df["unit_number"].isna().any():
        raise ValueError(f"{name}: found missing unit_number values.")

    if df["time_cycles"].isna().any():
        raise ValueError(f"{name}: found missing time_cycles values.")

    if (df["unit_number"] <= 0).any():
        raise ValueError(f"{name}: found unit_number values that are not positive.")

    if (df["time_cycles"] <= 0).any():
        raise ValueError(f"{name}: found time_cycles values that are not positive.")

    # Cycles reset to 1 for every new engine, so this must be checked per
    # engine, not across the whole file - a global check would be meaningless.
    for unit_number, engine_rows in df.groupby("unit_number"):
        cycles = engine_rows["time_cycles"].to_numpy()
        expected_cycles = range(1, len(cycles) + 1)
        if list(cycles) != list(expected_cycles):
            raise ValueError(
                f"{name}: unit {unit_number}'s time_cycles are not a clean "
                f"1..{len(cycles)} sequence in order."
            )


def summarize_cmapss_dataframe(df: pd.DataFrame, name: str) -> None:
    """Print a concise exploratory summary of a loaded C-MAPSS DataFrame."""
    n_engines = df["unit_number"].nunique()

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Number of engines: {n_engines}")
    print(f"Cycle range: {df['time_cycles'].min()} to {df['time_cycles'].max()}")

    print("\nFirst five rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    missing_counts = df.isna().sum()
    total_missing = missing_counts.sum()
    print(f"\nMissing values: {total_missing} total")
    if total_missing:
        print(missing_counts[missing_counts > 0])

    print("\nDescriptive statistics:")
    print(df.describe())


if __name__ == "__main__":
    train_df = load_train_fd001()
    validate_cmapss_dataframe(train_df, "train_FD001")
    summarize_cmapss_dataframe(train_df, "train_FD001")

    test_df = load_test_fd001()
    validate_cmapss_dataframe(test_df, "test_FD001")
    summarize_cmapss_dataframe(test_df, "test_FD001")

    rul = load_rul_fd001()
    print(f"\n{'=' * 60}")
    print("RUL_FD001")
    print(f"{'=' * 60}")
    print(f"Shape: {rul.shape[0]} values")
    print(f"Matches number of test engines: {rul.shape[0] == test_df['unit_number'].nunique()}")
    print(rul.describe())
