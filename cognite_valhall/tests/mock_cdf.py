"""An in-memory fake of the CDF APIs this pipeline uses.

WHY THIS EXISTS
---------------
Two reasons, and the second matters more than the first:

1. **CI / offline development.** The pipeline can be exercised end to end with
   no credentials and no network. That is how you iterate on the reporting and
   storage logic without hammering a live API.

2. **Contract verification.** Every object this fake returns is a *real* SDK
   data class (``Asset``, ``TimeSeries``, ``Event``, ``FileMetadata``,
   ``Datapoints``, ``LatestDatapoint``), not a dict or a MagicMock. So if the
   pipeline reads a field that does not exist on the real class, or assumes the
   wrong shape (``timestamp`` as a list vs a scalar — a genuine difference
   between SDK majors), the test fails here rather than in production. A
   MagicMock-based fake would happily return a Mock for any attribute and
   verify nothing.

The synthetic hierarchy deliberately mirrors the *shape* of a compressor train
without claiming to be the real Valhall data, and the synthetic telemetry
deliberately embeds the defects the quality report is supposed to catch: a
frozen transmitter, duplicate timestamps, a data gap, and an empty tag.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from cognite.client.data_classes import (
    Asset,
    AssetList,
    Datapoints,
    DatapointsList,
    Event,
    EventList,
    FileMetadata,
    FileMetadataList,
    Relationship,
    RelationshipList,
    TimeSeries,
    TimeSeriesList,
)
from cognite.client.data_classes.datapoints import LatestDatapoint, LatestDatapointList
from cognite.client.data_classes.iam import ProjectSpec, TokenInspection

NOW = datetime(2024, 6, 1, tzinfo=timezone.utc)
NOW_MS = int(NOW.timestamp() * 1000)


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


# --------------------------------------------------------------------------
# Synthetic hierarchy (shape only -- not a claim about the real dataset)
# --------------------------------------------------------------------------
_ASSET_SPECS = [
    # (id, external_id, name, description, parent_id)
    (1, "VAL", "Valhall", "Valhall platform", None),
    (2, "VAL-23", "23", "First stage recompression train", 1),
    (3, "23-KA-9101", "23-KA-9101", "1st stage compressor", 2),
    (4, "23-KA-9101-M01", "23-KA-9101-M01", "Compressor drive motor", 3),
    (5, "23-KA-9101-B01", "23-KA-9101-B01", "Drive end bearing assembly", 3),
    (6, "23-KA-9101-LO", "23-KA-9101-LO", "Lube oil system", 3),
    (7, "23-KA-9101-DGS", "23-KA-9101-DGS", "Dry gas seal system", 3),
    (8, "23-KA-9101-AS", "23-KA-9101-AS", "Anti-surge recycle valve", 3),
    (9, "23-VG-9101", "23-VG-9101", "Suction scrubber", 2),
    (10, "23-HA-9114", "23-HA-9114", "Discharge aftercooler", 2),
    (11, "23-PT-92531", "23-PT-92531", "Suction pressure transmitter", 9),
]

_TS_SPECS = [
    # (id, external_id, name, description, unit, asset_id, is_string, is_step)
    (101, "pi:163657", "VAL_23-KA-9101_ASP:VALUE", "Suction pressure", "barg", 3, False, False),
    (102, "pi:160184", "VAL_23-KA-9101_ADP:VALUE", "Discharge pressure", "barg", 3, False, False),
    (103, "pi:163661", "VAL_23-KA-9101_ADT:VALUE", "Discharge temperature", "degC", 3, False, False),
    (104, "pi:163708", "VAL_23-KA-9101_SPD:VALUE", "Shaft speed", "rpm", 3, False, False),
    (105, "pi:200001", "VAL_23-KA-9101-B01_VIB:VALUE", "Bearing vibration", "mm/s", 5, False, False),
    (106, "pi:200002", "VAL_23-KA-9101-LO_PT:VALUE", "Lube oil pressure", "barg", 6, False, False),
    (107, "pi:200003", "VAL_23-KA-9101-DGS_LEAK:VALUE", "Seal gas leakage", None, 7, False, False),
    (108, "pi:200004", "VAL_23-KA-9101-AS_POS:VALUE", "Anti-surge valve position", "%", 8, False, True),
    (109, "pi:200005", "VAL_23-HA-9114_TEMP:VALUE", "Cooler outlet temperature", "degC", 10, False, False),
    (110, "pi:200006", "VAL_23-KA-9101_STATE:VALUE", "Running state", None, 3, True, True),
    (111, "pi:200007", "VAL_23-VG-9101_LVL:VALUE", "Scrubber level (decommissioned)", "%", 9, False, False),
]

# Tags given deliberate defects, so the quality report has something to find.
FROZEN_TAG = "pi:200003"       # constant value -> frozen transmitter
DUPLICATE_TAG = "pi:200002"    # repeated timestamps -> double ingest
GAPPY_TAG = "pi:163661"        # a multi-hour hole -> historian outage
EMPTY_TAG = "pi:200007"        # no datapoints in window -> decommissioned


def _assets() -> list[Asset]:
    out = []
    for aid, xid, name, desc, parent in _ASSET_SPECS:
        out.append(
            Asset(
                id=aid, external_id=xid, name=name, description=desc,
                parent_id=parent, root_id=1, source="mock",
                metadata={"mock": "true", "tag": xid},
                created_time=NOW_MS, last_updated_time=NOW_MS,
            )
        )
    return out


def _timeseries() -> list[TimeSeries]:
    out = []
    for tid, xid, name, desc, unit, aid, is_str, is_step in _TS_SPECS:
        out.append(
            TimeSeries(
                id=tid, external_id=xid, name=name, description=desc, unit=unit,
                asset_id=aid, is_string=is_str, is_step=is_step,
                metadata={"source": "mock-historian"},
                created_time=NOW_MS, last_updated_time=NOW_MS,
            )
        )
    return out


def _events() -> list[Event]:
    """Alarm-like events: instants, no end_time, operational vocabulary.

    Chosen so the event classifier has to reach the *correct* conclusion that
    these are operational records rather than executed work orders.
    """
    out = []
    base = NOW - timedelta(days=45)
    for i in range(14):
        start = base + timedelta(days=i * 3)
        out.append(
            Event(
                id=900 + i,
                external_id=f"mock-event-{i}",
                type="alarm" if i % 2 == 0 else "state_change",
                subtype="high_temperature" if i % 2 == 0 else "compressor_start",
                description=(
                    "Discharge temperature high alarm" if i % 2 == 0 else "Compressor started"
                ),
                start_time=_ms(start),
                end_time=None,
                asset_ids=[3] if i % 3 else [3, 5],
                source="mock-dcs",
                metadata={"priority": "2", "alarm_group": "compressor"},
                created_time=NOW_MS, last_updated_time=NOW_MS,
            )
        )
    return out


def _files() -> list[FileMetadata]:
    specs = [
        (700, "PH-ME-P-0153-001.pdf", "application/pdf", [3], "P&ID first stage compressor"),
        (701, "PH-ME-P-0156-001.pdf", "application/pdf", [2], "P&ID recompression train"),
        (702, "23-KA-9101_datasheet.pdf", "application/pdf", [3], "Compressor datasheet"),
        (703, "PH-25578-P-4110006-001.pdf", "application/pdf", [], "Isometric drawing"),
        (704, "unrelated_report.docx", "application/msword", [], "Monthly production report"),
    ]
    out = []
    for fid, name, mime, assets, desc in specs:
        out.append(
            FileMetadata(
                id=fid, external_id=f"file-{fid}", name=name, mime_type=mime,
                asset_ids=assets, source="mock-docs", directory="/drawings",
                metadata={"description": desc, "document_type": "PID" if "P-01" in name else "OTHER"},
                uploaded=True, uploaded_time=NOW_MS,
                created_time=NOW_MS, last_updated_time=NOW_MS,
            )
        )
    return out


def _generate_datapoints(
    xid: str, start: datetime, end: datetime, interval_s: int = 60
) -> tuple[list[int], list[float]]:
    """Synthesise a plausible process signal, with the defects noted above."""
    spec = next((s for s in _TS_SPECS if s[1] == xid), None)
    if spec is None or xid == EMPTY_TAG:
        return [], []
    if spec[6]:  # is_string
        return [], []

    timestamps: list[int] = []
    values: list[float] = []
    cursor = start
    step = timedelta(seconds=interval_s)
    i = 0
    seed = sum(ord(c) for c in xid)
    while cursor < end:
        if xid == GAPPY_TAG:
            # A 6-hour outage one third of the way in.
            hole_start = start + (end - start) / 3
            if hole_start <= cursor < hole_start + timedelta(hours=6):
                cursor += step
                i += 1
                continue
        ms = _ms(cursor)
        if xid == FROZEN_TAG:
            value = 1.25
        else:
            value = (
                50.0
                + 10.0 * math.sin((i + seed) / 240.0)
                + 0.5 * math.sin((i + seed) / 7.0)
            )
        timestamps.append(ms)
        values.append(value)
        if xid == DUPLICATE_TAG and i % 500 == 0 and i:
            timestamps.append(ms)  # deliberate duplicate ingest
            values.append(value)
        cursor += step
        i += 1
    return timestamps, values


# --------------------------------------------------------------------------
# Fake API surfaces
# --------------------------------------------------------------------------
class _MockAssetsAPI:
    def __init__(self, assets: list[Asset]):
        self._assets = assets
        self.calls: list[str] = []

    def retrieve(self, id=None, external_id=None):
        self.calls.append("retrieve")
        for asset in self._assets:
            if id is not None and asset.id == id:
                return asset
            if external_id is not None and asset.external_id == external_id:
                return asset
        return None

    def list(self, name=None, parent_ids=None, limit=None, root=None, **kwargs):
        self.calls.append("list")
        out = self._assets
        if name is not None:
            out = [a for a in out if a.name == name]
        if parent_ids is not None:
            out = [a for a in out if a.parent_id in set(parent_ids)]
        if root:
            out = [a for a in out if a.parent_id is None]
        return AssetList(list(out))

    def retrieve_subtree(self, id=None, external_id=None, depth=None):
        self.calls.append("retrieve_subtree")
        root = self.retrieve(id=id, external_id=external_id)
        if root is None:
            return AssetList([])
        collected = [root]
        frontier = [root.id]
        while frontier:
            children = [a for a in self._assets if a.parent_id in frontier]
            if not children:
                break
            collected.extend(children)
            frontier = [c.id for c in children]
        return AssetList(collected)

    def search(self, name=None, description=None, query=None, filter=None, limit=25):
        self.calls.append("search")
        if not query:
            return AssetList([])
        needle = query.lower()
        return AssetList(
            [a for a in self._assets if needle in (a.name or "").lower()][:limit]
        )


class _MockDatapointsAPI:
    def __init__(self, series: list[TimeSeries]):
        self._series = {ts.external_id: ts for ts in series}
        self.retrieve_calls: list[dict] = []

    def retrieve(self, *, external_id=None, id=None, start=None, end=None,
                 aggregates=None, granularity=None, limit=None,
                 ignore_unknown_ids=False, **kwargs):
        self.retrieve_calls.append(
            {"n": len(external_id) if isinstance(external_id, list) else 1,
             "start": start, "end": end, "aggregates": aggregates}
        )
        xids = external_id if isinstance(external_id, list) else [external_id]

        # `start=0` means the epoch; the probe uses it to find the oldest point.
        if start == 0 or start is None:
            start_dt = NOW - timedelta(days=365)
        elif isinstance(start, datetime):
            start_dt = start
        else:
            start_dt = NOW - timedelta(days=365)
        end_dt = end if isinstance(end, datetime) else NOW

        interval = 60
        if aggregates and granularity:
            interval = {"1s": 1, "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}.get(
                granularity, 60
            )

        out = []
        for xid in xids:
            ts_meta = self._series.get(xid)
            if ts_meta is None:
                if ignore_unknown_ids:
                    continue
                raise ValueError(f"unknown external_id {xid}")
            timestamps, values = _generate_datapoints(xid, start_dt, end_dt, interval)
            if limit is not None and limit > 0:
                timestamps, values = timestamps[:limit], values[:limit]
            kwargs_dp = dict(
                id=ts_meta.id, external_id=xid, is_string=ts_meta.is_string,
                is_step=ts_meta.is_step, unit=ts_meta.unit, timestamp=timestamps,
                type="string" if ts_meta.is_string else "numeric",
            )
            if aggregates:
                agg = aggregates if isinstance(aggregates, str) else aggregates[0]
                kwargs_dp[agg] = values
                kwargs_dp["granularity"] = granularity
            else:
                kwargs_dp["value"] = values
            out.append(Datapoints(**kwargs_dp))

        if not isinstance(external_id, list):
            return out[0] if out else None
        return DatapointsList(out)

    def retrieve_dataframe(self, *, external_id=None, start=None, end=None,
                           aggregates=None, granularity=None,
                           ignore_unknown_ids=False, include_aggregate_name=True, **kwargs):
        import pandas as pd

        result = self.retrieve(
            external_id=external_id, start=start, end=end, aggregates=aggregates,
            granularity=granularity, ignore_unknown_ids=ignore_unknown_ids,
        )
        items = [result] if isinstance(result, Datapoints) else list(result or [])
        frames = {}
        for dps in items:
            agg = aggregates if isinstance(aggregates, str) else (aggregates or [None])[0]
            values = getattr(dps, agg, None) if agg else dps.value
            if not dps.timestamp:
                continue
            frames[dps.external_id] = pd.Series(
                list(values or []), index=pd.to_datetime(list(dps.timestamp), unit="ms", utc=True)
            )
        return pd.DataFrame(frames) if frames else pd.DataFrame()

    def retrieve_latest(self, id=None, external_id=None, before=None,
                        ignore_unknown_ids=False, **kwargs):
        xids = external_id if isinstance(external_id, list) else [external_id]
        out = []
        for xid in xids:
            ts_meta = self._series.get(xid)
            if ts_meta is None:
                continue
            if xid == EMPTY_TAG:
                out.append(
                    LatestDatapoint(
                        id=ts_meta.id, timestamp=None, value=None,
                        is_string=bool(ts_meta.is_string), type="numeric",
                        before=None, external_id=xid, unit=ts_meta.unit,
                    )
                )
                continue
            out.append(
                LatestDatapoint(
                    id=ts_meta.id,
                    # NOTE: a datetime, matching SDK 8.x semantics.
                    timestamp=NOW,
                    value=42.0,
                    is_string=bool(ts_meta.is_string),
                    type="string" if ts_meta.is_string else "numeric",
                    before=None,
                    external_id=xid,
                    unit=ts_meta.unit,
                )
            )
        return LatestDatapointList(out)


class _MockTimeSeriesAPI:
    def __init__(self, series: list[TimeSeries]):
        self._series = series
        self.data = _MockDatapointsAPI(series)

    def list(self, asset_ids=None, limit=None, **kwargs):
        out = self._series
        if asset_ids is not None:
            out = [ts for ts in out if ts.asset_id in set(asset_ids)]
        return TimeSeriesList(list(out))

    def retrieve(self, id=None, external_id=None, **kwargs):
        for ts in self._series:
            if ts.id == id or ts.external_id == external_id:
                return ts
        return None


class _MockEventsAPI:
    def __init__(self, events: list[Event]):
        self._events = events

    def list(self, asset_ids=None, asset_subtree_ids=None, limit=None, **kwargs):
        out = self._events
        if asset_ids is not None:
            wanted = set(asset_ids)
            out = [e for e in out if wanted & set(e.asset_ids or [])]
        return EventList(list(out))


class _MockFilesAPI:
    def __init__(self, files: list[FileMetadata]):
        self._files = files
        self.downloaded: list[int] = []

    def list(self, asset_ids=None, limit=None, **kwargs):
        out = self._files
        if asset_ids is not None:
            wanted = set(asset_ids)
            out = [f for f in out if wanted & set(f.asset_ids or [])]
        return FileMetadataList(list(out))

    def search(self, name=None, filter=None, limit=25):
        if not name:
            return FileMetadataList([])
        needle = name.lower().replace("&", "")
        hits = [
            f for f in self._files
            if needle in (f.name or "").lower().replace("&", "")
            or needle in str(f.metadata or {}).lower().replace("&", "")
        ]
        return FileMetadataList(hits[:limit])

    def download_to_path(self, path, id=None, external_id=None, instance_id=None):
        self.downloaded.append(id)
        from pathlib import Path

        Path(path).write_bytes(b"%PDF-1.4\n% mock document bytes\n")


class _MockRelationshipsAPI:
    def __init__(self, relationships: list[Relationship]):
        self._relationships = relationships

    def list(self, source_external_ids=None, target_external_ids=None, limit=None, **kwargs):
        out = self._relationships
        if source_external_ids is not None:
            wanted = set(source_external_ids)
            out = [r for r in out if r.source_external_id in wanted]
        elif target_external_ids is not None:
            wanted = set(target_external_ids)
            out = [r for r in out if r.target_external_id in wanted]
        return RelationshipList(list(out))


class _MockSpacesAPI:
    def list(self, limit=25, include_global=False):
        return []


class _MockDataModelsAPI:
    def list(self, **kwargs):
        return []


class _MockDataModelingAPI:
    def __init__(self):
        self.spaces = _MockSpacesAPI()
        self.data_models = _MockDataModelsAPI()


class _MockTokenAPI:
    def inspect(self):
        return TokenInspection(
            subject="mock-user",
            projects=[ProjectSpec(url_name="publicdata", groups=[1])],
            capabilities=[],
        )


class _MockIAM:
    def __init__(self):
        self.token = _MockTokenAPI()


class MockCogniteClient:
    """Quacks like ``CogniteClient`` for exactly the surface this pipeline uses."""

    def __init__(self, with_relationships: bool = False):
        self.assets = _MockAssetsAPI(_assets())
        self.time_series = _MockTimeSeriesAPI(_timeseries())
        self.events = _MockEventsAPI(_events())
        self.files = _MockFilesAPI(_files())
        rels = []
        if with_relationships:
            rels = [
                Relationship(
                    external_id="rel-1", source_external_id="23-VG-9101",
                    source_type="asset", target_external_id="23-KA-9101",
                    target_type="asset", confidence=0.9,
                    created_time=NOW_MS, last_updated_time=NOW_MS,
                )
            ]
        self.relationships = _MockRelationshipsAPI(rels)
        self.data_modeling = _MockDataModelingAPI()
        self.iam = _MockIAM()
