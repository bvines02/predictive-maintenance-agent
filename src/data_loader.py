"""Stage 2 - Data loading.

Responsible for:
- Reading raw telemetry from data/raw/
- Checking the columns we expect are present
- Returning a pandas DataFrame

Not responsible for: cleaning or feature engineering (see features.py).
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_FILE

# The AI4I 2020 dataset's columns. Used to fail fast if the file we're given
# doesn't look like the dataset this project expects (wrong file, corrupted
# download, or a schema change upstream).
EXPECTED_COLUMNS = [
    "UDI",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
]


def load_telemetry(file_path: str | Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw telemetry CSV into a DataFrame.

    Args:
        file_path: Path to the CSV file. Defaults to data/raw/ai4i2020.csv.

    Returns:
        The telemetry data, unmodified.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If the file is missing expected columns.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Telemetry file not found: {file_path}\n"
            "Download the AI4I 2020 dataset and place it there - see the "
            "'Dataset' section in README.md."
        )

    df = pd.read_csv(file_path)

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Telemetry file is missing expected columns: {sorted(missing_columns)}\n"
            f"Found columns: {list(df.columns)}"
        )

    return df


if __name__ == "__main__":
    data = load_telemetry()
    print(f"Loaded {len(data)} rows from {RAW_DATA_FILE}")
    print(data.head())
