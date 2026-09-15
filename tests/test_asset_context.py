"""Tests for src/asset_context.py."""

import json

import pytest

from src.asset_context import has_redundancy, load_asset_context

SAMPLE_ASSETS = [
    {
        "asset_id": "pump-01",
        "criticality": "High",
        "single_point_of_failure": True,
        "redundancy": "None",
    },
    {
        "asset_id": "pump-02",
        "criticality": "Medium",
        "single_point_of_failure": False,
        "redundancy": "1 standby pump (pump-02b)",
    },
]


@pytest.fixture()
def sample_context_file(tmp_path):
    path = tmp_path / "asset_context.json"
    path.write_text(json.dumps(SAMPLE_ASSETS))
    return path


def test_load_asset_context_finds_known_asset(sample_context_file):
    asset = load_asset_context("pump-02", path=sample_context_file)

    assert asset["asset_id"] == "pump-02"
    assert asset["criticality"] == "Medium"


def test_load_asset_context_unknown_asset_raises_key_error(sample_context_file):
    with pytest.raises(KeyError, match="Unknown asset_id"):
        load_asset_context("does-not-exist", path=sample_context_file)


def test_load_asset_context_missing_file_raises_file_not_found(tmp_path):
    missing_path = tmp_path / "does_not_exist.json"

    with pytest.raises(FileNotFoundError):
        load_asset_context("pump-01", path=missing_path)


def test_has_redundancy_true_when_described():
    asset = {"redundancy": "1 standby pump (pump-02b)"}
    assert has_redundancy(asset) is True


def test_has_redundancy_false_when_none():
    asset = {"redundancy": "None"}
    assert has_redundancy(asset) is False


def test_has_redundancy_is_case_insensitive():
    asset = {"redundancy": "none"}
    assert has_redundancy(asset) is False


def test_real_asset_context_file_loads_and_has_required_fields():
    """Integration check against the real data/assets/asset_context.json -
    catches typos or missing fields in the actual data file, not just the
    loader logic.
    """
    required_fields = {
        "asset_id",
        "asset_type",
        "system",
        "criticality",
        "duty",
        "redundancy",
        "single_point_of_failure",
        "known_failure_modes",
        "operational_workaround",
        "maintenance_strategy",
    }

    asset = load_asset_context("pump-01")  # uses the real file, default path

    assert required_fields.issubset(asset.keys())
