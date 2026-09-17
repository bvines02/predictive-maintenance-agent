"""Unit and integration tests, all running against MockCogniteClient.

The integration test is the valuable one: it runs the real ``main()`` with only
the auth boundary patched, so every stage executes. The unit tests then pin the
specific behaviours that are easy to regress silently — the UserList trap, the
UTC contract, frozen-sensor detection, and the event classifier's refusal to
over-claim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.discover_assets import classify_subsystems, discover  # noqa: E402
from src.download_assets import assets_to_dataframe  # noqa: E402
from src.download_events import (  # noqa: E402
    classify_event_nature,
    events_to_dataframe,
    profile_events,
)
from src.download_relationships import build_implicit_edges  # noqa: E402
from src.download_timeseries import (  # noqa: E402
    _time_chunks,
    discover_timeseries,
    resolve_window,
    timeseries_to_dataframe,
)
from src.quality_report import analyse_datapoint_files  # noqa: E402
from src.storage import json_or_none, ms_to_utc, safe_filename  # noqa: E402
from tests.mock_cdf import DUPLICATE_TAG, FROZEN_TAG, MockCogniteClient  # noqa: E402


@pytest.fixture
def client() -> MockCogniteClient:
    return MockCogniteClient()


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------
def test_finds_target_by_external_id(client):
    result = discover(client, "23-KA-9101")
    assert result.target.external_id == "23-KA-9101"
    assert result.target.id == 3


def test_discovery_collects_ancestors_descendants_and_siblings(client):
    result = discover(client, "23-KA-9101")
    assert [a.external_id for a in result.ancestors] == ["VAL", "VAL-23"]
    assert len(result.descendants) == 5
    assert {a.external_id for a in result.siblings} == {"23-VG-9101", "23-HA-9114"}


def test_selected_is_deduplicated_and_deterministic(client):
    result = discover(client, "23-KA-9101")
    ids = result.selected_ids
    assert len(ids) == len(set(ids)), "selected must not contain duplicates"
    assert result.selected_ids == discover(client, "23-KA-9101").selected_ids


def test_roles_are_assigned_with_target_winning(client):
    result = discover(client, "23-KA-9101")
    roles = result.role_of
    assert roles[3] == "target"
    assert roles[1] == "ancestor"
    assert roles[5] == "descendant"
    assert roles[9] == "sibling"


def test_unknown_asset_raises_lookup_error(client):
    with pytest.raises(LookupError):
        discover(client, "DOES-NOT-EXIST-9999")


def test_keyword_classification_reports_absences(client):
    result = discover(client, "23-KA-9101")
    matches = classify_subsystems(result.selected)
    # Every configured subsystem must appear as a key, present or not: a
    # missing key would hide the fact that we looked and found nothing.
    assert set(matches) == set(config.SUBSYSTEM_KEYWORDS)
    assert matches["bearing"], "bearing assembly should be matched"
    assert matches["gearbox"] == [], "this synthetic train has no gearbox"


# --------------------------------------------------------------------------
# Asset persistence
# --------------------------------------------------------------------------
def test_assets_dataframe_has_both_identifiers_and_a_materialised_path(client):
    result = discover(client, "23-KA-9101")
    df = assets_to_dataframe(result)
    assert {"asset_id", "external_id", "parent_external_id", "path_external_ids"} <= set(df.columns)
    assert df["asset_id"].notna().all()
    row = df[df["external_id"] == "23-KA-9101-B01"].iloc[0]
    assert row["path_external_ids"] == "VAL / VAL-23 / 23-KA-9101 / 23-KA-9101-B01"
    assert row["depth"] == 3


def test_root_asset_has_no_parent(client):
    df = assets_to_dataframe(discover(client, "23-KA-9101"))
    root = df[df["external_id"] == "VAL"].iloc[0]
    assert pd.isna(root["parent_id"]) or root["parent_id"] is None
    assert root["depth"] == 0


# --------------------------------------------------------------------------
# Time series
# --------------------------------------------------------------------------
def test_timeseries_discovery_links_to_selected_assets(client):
    result = discover(client, "23-KA-9101")
    series = discover_timeseries(client, result.selected_ids)
    assert len(series) == 11
    df = timeseries_to_dataframe(series, {a.id: a.external_id for a in result.selected})
    assert df["asset_external_id"].notna().all()
    assert "is_step" in df.columns, "is_step drives interpolation policy downstream"


def test_datapoints_list_is_not_a_builtin_list(client):
    """Guards the exact trap that broke the first implementation."""
    result = client.time_series.data.retrieve(
        external_id=["pi:163657", "pi:160184"],
        start=pd.Timestamp("2024-05-01", tz="UTC").to_pydatetime(),
        end=pd.Timestamp("2024-05-02", tz="UTC").to_pydatetime(),
    )
    assert not isinstance(result, list), (
        "DatapointsList is a UserList; code must not rely on isinstance(x, list)"
    )
    assert len(result) == 2


def test_time_chunks_cover_the_window_exactly():
    start = pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime()
    end = pd.Timestamp("2024-01-31", tz="UTC").to_pydatetime()
    chunks = _time_chunks(start, end, 7)
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    # Contiguous, non-overlapping.
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert prev_end == next_start


def test_resolve_window_defaults_to_lookback_before_latest_known():
    latest = pd.Timestamp("2024-06-01", tz="UTC")
    start, end = resolve_window(None, None, 30, latest_known=latest)
    assert end == latest.to_pydatetime()
    assert (end - start).days == 30


def test_resolve_window_rejects_inverted_range():
    with pytest.raises(ValueError):
        resolve_window("2024-06-01", "2024-01-01", 30)


# --------------------------------------------------------------------------
# Storage helpers
# --------------------------------------------------------------------------
def test_ms_to_utc_is_timezone_aware():
    ts = ms_to_utc(1717200000000)
    assert ts is not None and ts.tzinfo is not None
    assert str(ts.tz) == "UTC"


def test_safe_filename_neutralises_colons_without_collision():
    a = safe_filename("VAL_23-KA-9101_ASP:VALUE")
    b = safe_filename("VAL_23-KA-9101_ADP:VALUE")
    assert ":" not in a and ":" not in b
    assert a != b


def test_safe_filename_truncates_with_a_hash_to_avoid_collision():
    long_a = safe_filename("x" * 400 + "AAA")
    long_b = safe_filename("x" * 400 + "BBB")
    assert long_a != long_b, "truncation must not collapse distinct tags"


def test_json_or_none_round_trips():
    import json

    assert json_or_none(None) is None
    assert json.loads(json_or_none({"b": 1, "a": 2})) == {"a": 2, "b": 1}


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------
def test_event_classifier_does_not_claim_work_orders_for_alarm_records(client):
    result = discover(client, "23-KA-9101")
    events = client.events.list(asset_ids=result.selected_ids, limit=None)
    df = events_to_dataframe(list(events), {})
    profile = profile_events(df)
    assessment = classify_event_nature(df, profile)
    assert assessment["verdict"] == "operational event"
    assert assessment["verdict"] != "maintenance work"
    # And it must say why it is not treating them as completed work.
    assert "end_time" in assessment["caveat"]


def test_event_classifier_admits_ignorance_on_opaque_records():
    df = pd.DataFrame([
        {"event_id": 1, "type": "XYZ", "subtype": "QQQ", "description": "zzz",
         "start_time": pd.Timestamp("2024-01-01", tz="UTC"), "end_time": None,
         "duration_hours": None, "n_assets": 1, "metadata_keys": "foo", "source": "s"}
    ])
    assessment = classify_event_nature(df, profile_events(df))
    assert assessment["verdict"] == "unknown / project-specific"
    assert assessment["confidence"] == "low"


def test_event_classifier_handles_no_events():
    assessment = classify_event_nature(pd.DataFrame(), {"n_events": 0})
    assert assessment["verdict"] == "no events found"


# --------------------------------------------------------------------------
# Relationships
# --------------------------------------------------------------------------
def test_implicit_edges_use_string_node_keys_only(client):
    result = discover(client, "23-KA-9101")
    assets_df = assets_to_dataframe(result)
    series = discover_timeseries(client, result.selected_ids)
    ts_df = timeseries_to_dataframe(series, {a.id: a.external_id for a in result.selected})
    events_df = events_to_dataframe(
        list(client.events.list(asset_ids=result.selected_ids, limit=None)),
        {a.id: a.external_id for a in result.selected},
    )
    edges = build_implicit_edges(assets_df, ts_df, events_df, pd.DataFrame())

    assert not edges.empty
    assert set(edges["relationship"]) == {"PART_OF", "MEASURES", "AFFECTS"}
    # Mixed int/str node keys break the Parquet write and the graph build.
    for col in ("source_external_id", "target_external_id"):
        assert all(isinstance(v, str) for v in edges[col].dropna())
    # PART_OF must not include the root (it has no parent).
    part_of = edges[edges["relationship"] == "PART_OF"]
    assert "VAL" not in set(part_of["source_external_id"])


def test_implicit_edges_empty_input_keeps_schema():
    edges = build_implicit_edges(
        pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    )
    assert edges.empty
    assert "relationship" in edges.columns


# --------------------------------------------------------------------------
# Quality analysis
# --------------------------------------------------------------------------
def test_quality_analysis_detects_injected_defects(tmp_path, monkeypatch):
    """Run the download, then confirm the report finds the planted faults."""
    from src import download_timeseries

    monkeypatch.setattr(config, "DATAPOINTS_DIR", tmp_path)
    monkeypatch.setattr(download_timeseries.config, "DATAPOINTS_DIR", tmp_path)
    monkeypatch.setattr(config, "TIMESERIES_DIR", tmp_path)
    monkeypatch.setattr(download_timeseries.config, "TIMESERIES_DIR", tmp_path)

    client = MockCogniteClient()
    result = discover(client, "23-KA-9101")
    series = discover_timeseries(client, result.selected_ids)
    start = pd.Timestamp("2024-05-25", tz="UTC").to_pydatetime()
    end = pd.Timestamp("2024-06-01", tz="UTC").to_pydatetime()
    download_timeseries.download_datapoints(client, series, start, end, aggregate=None)

    stats = analyse_datapoint_files(tmp_path)
    assert not stats.empty
    with_data = stats[stats["n_points"] > 0]

    frozen = with_data[with_data["external_id"] == FROZEN_TAG]
    assert len(frozen) == 1 and bool(frozen.iloc[0]["is_frozen"]), "frozen sensor missed"

    dups = with_data[with_data["external_id"] == DUPLICATE_TAG]
    assert int(dups.iloc[0]["n_duplicate_timestamps"]) > 0, "duplicate timestamps missed"

    # Healthy channels must NOT be flagged -- a check that fires on everything
    # is as useless as one that never fires.
    healthy = with_data[with_data["external_id"] == "pi:163657"]
    assert not bool(healthy.iloc[0]["is_frozen"])
    assert int(healthy.iloc[0]["n_duplicate_timestamps"]) == 0


def test_datapoint_files_are_utc_and_carry_identifiers(tmp_path, monkeypatch):
    from src import download_timeseries

    monkeypatch.setattr(download_timeseries.config, "DATAPOINTS_DIR", tmp_path)
    monkeypatch.setattr(download_timeseries.config, "TIMESERIES_DIR", tmp_path)

    client = MockCogniteClient()
    series = [s for s in client.time_series.list(limit=None) if s.external_id == "pi:163657"]
    start = pd.Timestamp("2024-05-30", tz="UTC").to_pydatetime()
    end = pd.Timestamp("2024-06-01", tz="UTC").to_pydatetime()
    download_timeseries.download_datapoints(client, series, start, end, aggregate=None)

    files = list(tmp_path.glob("*.parquet"))
    dp_files = [f for f in files if "summary" not in f.name]
    assert dp_files
    df = pd.read_parquet(dp_files[0])
    assert str(df["timestamp"].dt.tz) == "UTC"
    assert set(df["external_id"]) == {"pi:163657"}
    assert df["timeseries_id"].notna().all()
    assert df["timestamp"].is_monotonic_increasing


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def test_full_pipeline_runs_and_writes_every_artefact(tmp_path, monkeypatch):
    from src import cognite_client, main as main_mod

    # Redirect all output into the test's temp directory.
    for name, sub in [
        ("DATA_DIR", ""), ("ASSETS_DIR", "assets"), ("TIMESERIES_DIR", "timeseries"),
        ("DATAPOINTS_DIR", "timeseries/datapoints"), ("EVENTS_DIR", "events"),
        ("FILES_DIR", "files"), ("DOCUMENTS_DIR", "files/documents"),
        ("RELATIONSHIPS_DIR", "relationships"),
    ]:
        target = tmp_path / sub if sub else tmp_path
        target.mkdir(parents=True, exist_ok=True)
        for module in (config, main_mod.config):
            monkeypatch.setattr(module, name, target)
    monkeypatch.setattr(config, "ALL_DIRS", [])

    client = MockCogniteClient(with_relationships=True)
    monkeypatch.setattr(cognite_client, "get_verified_client", lambda: client)
    monkeypatch.setattr(main_mod, "get_verified_client", lambda: client)

    exit_code = main_mod.main(["--lookback-days", "3", "--granularity", "1h"])
    assert exit_code == 0

    expected = [
        "assets/assets.parquet", "assets/assets.csv",
        "timeseries/timeseries_metadata.parquet",
        "events/events.parquet", "events/events.csv",
        "files/files_metadata.parquet",
        "relationships/relationships.parquet",
        "relationships/implicit_edges.parquet",
        "data_quality_report.md",
    ]
    for rel in expected:
        assert (tmp_path / rel).exists(), f"missing artefact: {rel}"

    report = (tmp_path / "data_quality_report.md").read_text()
    assert "23-KA-9101" in report
    assert "Fitness for modelling" in report
    # The report must never render a NaN unit as the literal string "nan".
    assert "| nan |" not in report


def test_pipeline_returns_error_code_for_unknown_asset(tmp_path, monkeypatch):
    from src import cognite_client, main as main_mod

    monkeypatch.setattr(config, "ALL_DIRS", [])
    client = MockCogniteClient()
    monkeypatch.setattr(cognite_client, "get_verified_client", lambda: client)
    monkeypatch.setattr(main_mod, "get_verified_client", lambda: client)
    assert main_mod.main(["--asset", "NO-SUCH-TAG", "--skip-datapoints", "--skip-files"]) == 3
