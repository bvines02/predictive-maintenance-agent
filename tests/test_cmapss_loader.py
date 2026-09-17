"""Tests for src/cmapss_loader.py.

Uses tiny hand-built text files instead of the real (large) FD001 files, so
tests run fast and don't depend on the dataset being downloaded.
"""

import pandas as pd
import pytest

from src.cmapss_loader import (
    EXPECTED_COLUMN_COUNT,
    load_rul_fd001,
    load_train_fd001,
    validate_cmapss_dataframe,
)


def _write_cmapss_file(path, rows):
    """Write whitespace-separated rows (each a list of numbers) to path,
    including the trailing space each real C-MAPSS line has."""
    lines = [" ".join(str(v) for v in row) + " " for row in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_row(unit_number, time_cycles):
    # 3 operational settings + 21 sensors = 24 filler values.
    return [unit_number, time_cycles] + [0.0] * 24


def test_load_train_fd001_parses_whitespace_and_assigns_columns(tmp_path):
    rows = [_make_row(1, 1), _make_row(1, 2), _make_row(2, 1)]
    file_path = _write_cmapss_file(tmp_path / "train_mock.txt", rows)

    df = load_train_fd001(file_path)

    assert df.shape == (3, EXPECTED_COLUMN_COUNT)
    assert list(df.columns[:2]) == ["unit_number", "time_cycles"]
    assert df["unit_number"].tolist() == [1, 1, 2]


def test_load_train_fd001_missing_file_raises_clear_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.txt"

    with pytest.raises(FileNotFoundError, match="not found"):
        load_train_fd001(missing_path)


def test_load_rul_fd001_returns_one_value_per_engine(tmp_path):
    file_path = tmp_path / "RUL_mock.txt"
    file_path.write_text("112 \n98 \n69 \n")

    rul = load_rul_fd001(file_path)

    assert list(rul) == [112, 98, 69]


def test_validate_passes_on_clean_data(tmp_path):
    rows = [_make_row(1, 1), _make_row(1, 2), _make_row(2, 1)]
    file_path = _write_cmapss_file(tmp_path / "train_mock.txt", rows)
    df = load_train_fd001(file_path)

    validate_cmapss_dataframe(df, "mock")  # should not raise


def test_validate_rejects_missing_unit_number():
    df = pd.DataFrame(
        {
            "unit_number": [1, None, 2],
            "time_cycles": [1, 2, 1],
            **{f"sensor_{i}": [0.0, 0.0, 0.0] for i in range(1, 22)},
            "operational_setting_1": [0.0, 0.0, 0.0],
            "operational_setting_2": [0.0, 0.0, 0.0],
            "operational_setting_3": [0.0, 0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="missing unit_number"):
        validate_cmapss_dataframe(df, "mock")


def test_validate_rejects_non_sequential_cycles(tmp_path):
    # Unit 1 skips from cycle 1 to cycle 3 - not a clean 1..N sequence.
    rows = [_make_row(1, 1), _make_row(1, 3)]
    file_path = _write_cmapss_file(tmp_path / "train_mock.txt", rows)
    df = load_train_fd001(file_path)

    with pytest.raises(ValueError, match="not a clean"):
        validate_cmapss_dataframe(df, "mock")
