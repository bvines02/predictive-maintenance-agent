"""Tests for src/engine_split.py."""

import numpy as np
import pandas as pd
import pytest

from src.engine_split import split_dataframe_by_unit, split_units, validate_engine_split


def _make_df(n_units=10, rows_per_unit=5):
    rows = []
    for unit in range(1, n_units + 1):
        for cycle in range(1, rows_per_unit + 1):
            rows.append({"unit_number": unit, "time_cycles": cycle, "value": unit * 100 + cycle})
    return pd.DataFrame(rows)


def test_split_units_has_no_overlap():
    units = np.arange(1, 21)

    train_units, val_units = split_units(units, test_size=0.2, random_state=42)

    assert set(train_units).isdisjoint(set(val_units))


def test_split_units_covers_all_units():
    units = np.arange(1, 21)

    train_units, val_units = split_units(units, test_size=0.2, random_state=42)

    assert set(train_units) | set(val_units) == set(units)


def test_split_units_is_reproducible_with_same_seed():
    units = np.arange(1, 21)

    train_1, val_1 = split_units(units, test_size=0.2, random_state=42)
    train_2, val_2 = split_units(units, test_size=0.2, random_state=42)

    assert set(train_1) == set(train_2)
    assert set(val_1) == set(val_2)


def test_split_units_respects_test_size_fraction():
    units = np.arange(1, 101)

    train_units, val_units = split_units(units, test_size=0.2, random_state=42)

    assert len(val_units) == 20
    assert len(train_units) == 80


def test_split_dataframe_by_unit_keeps_each_engine_whole():
    df = _make_df(n_units=10, rows_per_unit=5)
    train_units, val_units = split_units(df["unit_number"].unique(), test_size=0.2, random_state=42)

    train_df, val_df = split_dataframe_by_unit(df, train_units, val_units)

    # No engine split across both frames.
    assert set(train_df["unit_number"]).isdisjoint(set(val_df["unit_number"]))
    # Every row for a given unit lands in exactly one frame.
    for unit in df["unit_number"].unique():
        in_train = unit in train_units
        rows_in_train = (train_df["unit_number"] == unit).sum()
        rows_in_val = (val_df["unit_number"] == unit).sum()
        if in_train:
            assert rows_in_train == 5 and rows_in_val == 0
        else:
            assert rows_in_val == 5 and rows_in_train == 0


def test_validate_engine_split_passes_on_clean_split():
    df = _make_df(n_units=10, rows_per_unit=5)
    train_units, val_units = split_units(df["unit_number"].unique(), test_size=0.2, random_state=42)
    train_df, val_df = split_dataframe_by_unit(df, train_units, val_units)

    validate_engine_split(df, train_df, val_df, train_units, val_units)  # should not raise


def test_validate_engine_split_rejects_overlapping_units():
    df = _make_df(n_units=10, rows_per_unit=5)
    train_units = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    val_units = np.array([8, 9, 10])  # unit 8 overlaps
    train_df, val_df = split_dataframe_by_unit(df, train_units, val_units)

    with pytest.raises(AssertionError, match="both train and validation"):
        validate_engine_split(df, train_df, val_df, train_units, val_units)


def test_validate_engine_split_rejects_row_count_mismatch():
    df = _make_df(n_units=10, rows_per_unit=5)
    train_units, val_units = split_units(df["unit_number"].unique(), test_size=0.2, random_state=42)
    train_df, val_df = split_dataframe_by_unit(df, train_units, val_units)
    val_df = val_df.iloc[:-1]  # drop a row so counts no longer add up

    with pytest.raises(AssertionError, match="Row counts do not add up"):
        validate_engine_split(df, train_df, val_df, train_units, val_units)


def test_validate_engine_split_rejects_single_engine_split():
    df = _make_df(n_units=10, rows_per_unit=5)
    train_units = np.arange(1, 10)
    val_units = np.array([10])  # only one engine in validation
    train_df, val_df = split_dataframe_by_unit(df, train_units, val_units)

    with pytest.raises(AssertionError, match="multiple engines"):
        validate_engine_split(df, train_df, val_df, train_units, val_units)
