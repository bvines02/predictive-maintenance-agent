"""Tests for src/rul.py.

Uses tiny hand-built DataFrames instead of the real dataset, so the RUL
formula's correctness is proven independently of the NASA files.
"""

import pandas as pd
import pytest

from src.rul import add_capped_rul, add_rul_target, validate_capped_rul, validate_rul


def _make_df(rows):
    """rows: list of (unit_number, time_cycles) tuples."""
    return pd.DataFrame(rows, columns=["unit_number", "time_cycles"])


def test_add_rul_target_matches_worked_example():
    # Unit 1 fails at cycle 192, per the example in the task description.
    df = _make_df([(1, 190), (1, 191), (1, 192)])

    result = add_rul_target(df)

    assert result["rul"].tolist() == [2, 1, 0]


def test_add_rul_target_handles_multiple_engines_independently():
    df = _make_df([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)])

    result = add_rul_target(df)

    # Unit 1 has 3 cycles (max 3), unit 2 has 2 cycles (max 2) - each
    # engine's RUL must be computed against its OWN max, not a global one.
    assert result[result["unit_number"] == 1]["rul"].tolist() == [2, 1, 0]
    assert result[result["unit_number"] == 2]["rul"].tolist() == [1, 0]


def test_add_rul_target_does_not_mutate_input():
    df = _make_df([(1, 1), (1, 2)])

    add_rul_target(df)

    assert "rul" not in df.columns


def test_add_rul_target_preserves_row_count():
    df = _make_df([(1, 1), (1, 2), (2, 1), (2, 2), (2, 3)])

    result = add_rul_target(df)

    assert len(result) == len(df)


def test_validate_rul_passes_on_correct_labels():
    df = _make_df([(1, 1), (1, 2), (1, 3)])
    result = add_rul_target(df)

    validate_rul(result)  # should not raise


def test_validate_rul_rejects_missing_rul_column():
    df = _make_df([(1, 1), (1, 2)])

    with pytest.raises(ValueError, match="rul"):
        validate_rul(df)


def test_validate_rul_rejects_negative_rul():
    df = _make_df([(1, 1), (1, 2)])
    result = add_rul_target(df)
    result.loc[0, "rul"] = -5

    with pytest.raises(ValueError, match="negative"):
        validate_rul(result)


def test_validate_rul_rejects_nonzero_final_rul():
    df = _make_df([(1, 1), (1, 2)])
    result = add_rul_target(df)
    result.loc[result["time_cycles"] == 2, "rul"] = 3  # final cycle should be 0

    with pytest.raises(ValueError, match="expected 0"):
        validate_rul(result)


def test_validate_rul_rejects_non_unit_decrement():
    df = _make_df([(1, 1), (1, 2), (1, 3)])
    result = add_rul_target(df)
    result.loc[result["time_cycles"] == 2, "rul"] = 10  # breaks the -1 step

    with pytest.raises(ValueError, match="decrease by exactly 1"):
        validate_rul(result)


def test_add_capped_rul_matches_worked_example():
    # From the task description: raw [200, 150, 125, 100, 20, 0] with cap=125.
    df = pd.DataFrame({"rul": [200, 150, 125, 100, 20, 0]})

    result = add_capped_rul(df, cap=125)

    assert result["rul_capped"].tolist() == [125, 125, 125, 100, 20, 0]


def test_add_capped_rul_leaves_original_rul_column_untouched():
    df = pd.DataFrame({"rul": [200, 100, 0]})

    result = add_capped_rul(df, cap=125)

    assert result["rul"].tolist() == [200, 100, 0]


def test_add_capped_rul_does_not_mutate_input():
    df = pd.DataFrame({"rul": [200, 100, 0]})

    add_capped_rul(df, cap=125)

    assert "rul_capped" not in df.columns


def test_add_capped_rul_preserves_row_count():
    df = pd.DataFrame({"rul": [200, 150, 125, 100, 20, 0]})

    result = add_capped_rul(df, cap=125)

    assert len(result) == len(df)


def test_add_capped_rul_respects_configurable_cap():
    df = pd.DataFrame({"rul": [50, 30, 10, 0]})

    result = add_capped_rul(df, cap=20)

    assert result["rul_capped"].tolist() == [20, 20, 10, 0]


def test_add_capped_rul_requires_rul_column():
    df = pd.DataFrame({"time_cycles": [1, 2, 3]})

    with pytest.raises(ValueError, match="rul"):
        add_capped_rul(df, cap=125)


def test_validate_capped_rul_passes_on_correct_labels():
    df = pd.DataFrame({"rul": [200, 150, 125, 100, 20, 0]})
    result = add_capped_rul(df, cap=125)

    validate_capped_rul(result, cap=125)  # should not raise


def test_validate_capped_rul_rejects_missing_column():
    df = pd.DataFrame({"rul": [200, 100, 0]})

    with pytest.raises(ValueError, match="rul_capped"):
        validate_capped_rul(df, cap=125)


def test_validate_capped_rul_rejects_value_above_cap():
    df = pd.DataFrame({"rul": [200, 100, 0], "rul_capped": [125, 100, 0]})
    df.loc[0, "rul_capped"] = 150  # above the cap

    with pytest.raises(ValueError, match="above the configured cap"):
        validate_capped_rul(df, cap=125)


def test_validate_capped_rul_rejects_negative_value():
    df = pd.DataFrame({"rul": [200, 100, 0], "rul_capped": [125, 100, 0]})
    df.loc[2, "rul_capped"] = -1

    with pytest.raises(ValueError, match="negative"):
        validate_capped_rul(df, cap=125)


def test_validate_capped_rul_rejects_mismatch_below_cap():
    # rul=100 is <= cap (125), so rul_capped must equal rul exactly.
    df = pd.DataFrame({"rul": [200, 100, 0], "rul_capped": [125, 99, 0]})

    with pytest.raises(ValueError, match="rul <= cap"):
        validate_capped_rul(df, cap=125)


def test_validate_capped_rul_rejects_mismatch_above_cap():
    # rul=200 is > cap (125), so rul_capped must equal the cap exactly.
    df = pd.DataFrame({"rul": [200, 100, 0], "rul_capped": [124, 100, 0]})

    with pytest.raises(ValueError, match="rul > cap"):
        validate_capped_rul(df, cap=125)
