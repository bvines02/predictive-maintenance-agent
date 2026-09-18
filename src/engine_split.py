"""Stage 14e (split) - Engine-level train/validation split.

Responsible for:
- Splitting a C-MAPSS DataFrame into training and validation sets by
  unit_number (engine), never by individual row
- Validating that the split has no leakage

Not responsible for:
- Feature engineering
- Model training

Why split by engine and not by row: rows from the same engine are cycles of
one continuous degradation history - they are highly correlated with each
other (cycle 150 of engine 3 looks a lot like cycle 149 of engine 3). If a
random row-level split put cycle 149 in training and cycle 150 in
validation, the model would effectively be tested on data it has already
almost seen, making validation performance look far better than the model
would actually achieve on a genuinely new engine. Splitting by engine means
an engine's entire run-to-failure history is either fully "known" to the
model or fully unseen - the only split that reflects how this model would
really be used (predicting RUL for an engine it has never observed before).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_units(
    unit_numbers: pd.Series | np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split unique engine (unit_number) values into train/validation groups.

    Splits the UNIQUE unit numbers, not the rows - this is what makes the
    resulting split engine-level rather than row-level.
    """
    unique_units = np.unique(unit_numbers)
    train_units, val_units = train_test_split(
        unique_units, test_size=test_size, random_state=random_state
    )
    return train_units, val_units


def split_dataframe_by_unit(
    df: pd.DataFrame, train_units: np.ndarray, val_units: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame's rows according to which engine group they belong to."""
    train_df = df[df["unit_number"].isin(train_units)].copy()
    val_df = df[df["unit_number"].isin(val_units)].copy()
    return train_df, val_df


def validate_engine_split(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_units: np.ndarray,
    val_units: np.ndarray,
) -> None:
    """Prove the split has no engine-level leakage.

    Raises AssertionError with a specific message on the first check that
    fails, so a broken split is easy to diagnose rather than silently
    producing an over-optimistic model.
    """
    train_unit_set = set(train_units)
    val_unit_set = set(val_units)

    if train_unit_set & val_unit_set:
        raise AssertionError(
            f"Units appear in both train and validation: {train_unit_set & val_unit_set}"
        )

    if not set(train_df["unit_number"]).issubset(train_unit_set):
        raise AssertionError("train_df contains a unit_number outside train_units.")

    if not set(val_df["unit_number"]).issubset(val_unit_set):
        raise AssertionError("val_df contains a unit_number outside val_units.")

    if set(train_df["unit_number"]) & set(val_df["unit_number"]):
        raise AssertionError("The same unit_number appears in both train_df and val_df.")

    if train_df["unit_number"].nunique() < 2:
        raise AssertionError("train_df must contain multiple engines.")

    if val_df["unit_number"].nunique() < 2:
        raise AssertionError("val_df must contain multiple engines.")

    if len(train_df) + len(val_df) != len(df):
        raise AssertionError(
            f"Row counts do not add up: train ({len(train_df)}) + val ({len(val_df)}) "
            f"!= original ({len(df)})."
        )
