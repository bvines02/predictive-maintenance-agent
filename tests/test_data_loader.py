"""Tests for src/data_loader.py.

Uses a tiny in-memory CSV instead of the real dataset, so tests run without
needing the (large, git-ignored) download and stay fast and deterministic.
"""

import pandas as pd
import pytest

from src.data_loader import EXPECTED_COLUMNS, load_telemetry


def _write_mock_csv(path, columns=EXPECTED_COLUMNS):
    """Write a 2-row CSV with the given columns to path."""
    mock_df = pd.DataFrame(
        [
            {col: 0 for col in columns},
            {col: 1 for col in columns},
        ]
    )
    mock_df.to_csv(path, index=False)
    return path


def test_load_telemetry_returns_dataframe(tmp_path):
    csv_path = _write_mock_csv(tmp_path / "mock.csv")

    df = load_telemetry(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_telemetry_missing_file_raises_clear_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError, match="not found"):
        load_telemetry(missing_path)


def test_load_telemetry_missing_columns_raises_value_error(tmp_path):
    # A CSV that's missing "Machine failure" and other expected columns.
    incomplete_columns = ["UDI", "Product ID", "Type"]
    csv_path = _write_mock_csv(tmp_path / "incomplete.csv", columns=incomplete_columns)

    with pytest.raises(ValueError, match="missing expected columns"):
        load_telemetry(csv_path)
