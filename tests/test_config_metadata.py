"""Tests for the C-MAPSS column metadata in src/config.py."""

from src.cmapss_loader import OPERATIONAL_SETTING_COLUMNS, SENSOR_COLUMNS
from src.config import OPERATIONAL_SETTING_METADATA, SENSOR_METADATA
from src.train_rul_baseline import get_kept_sensor_columns
from src.cmapss_loader import load_train_fd001

VALID_TRENDS = {"rises", "falls", "constant"}


def test_sensor_metadata_covers_every_sensor_column_exactly():
    assert list(SENSOR_METADATA) == SENSOR_COLUMNS


def test_operational_setting_metadata_covers_every_setting_column():
    assert list(OPERATIONAL_SETTING_METADATA) == OPERATIONAL_SETTING_COLUMNS


def test_every_sensor_has_required_fields_and_valid_trend():
    for sensor, meta in SENSOR_METADATA.items():
        assert {"symbol", "name", "unit", "expected_trend"} <= set(meta), sensor
        assert meta["expected_trend"] in VALID_TRENDS, sensor


def test_metadata_constants_agree_with_step_4_screening():
    train_df = load_train_fd001()
    kept = set(get_kept_sensor_columns(train_df))

    for sensor, meta in SENSOR_METADATA.items():
        assert (meta["expected_trend"] == "constant") == (sensor not in kept), sensor
