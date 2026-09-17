# Valhall First-Stage Compressor — Cognite Open Industrial Data Extractor

A focused, reproducible extraction of the **Valhall first-stage gas compressor
train** (centred on asset `23-KA-9101`) from Cognite / Aker BP's
[Open Industrial Data](https://openindustrialdata.com) project.

The goal is a *small, coherent, correctly contextualised* production subsystem
rather than a large pile of unrelated data — enough to build this pipeline:

```
SCADA / time-series telemetry
      ↓
anomaly detection / ML
      ↓
asset hierarchy
      ↓
maintenance history
      ↓
P&ID / engineering context
      ↓
knowledge graph
      ↓
troubleshooting recommendations
```

---

## Quick start

```bash
cd cognite_valhall
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # defaults are public and usually fine as-is

python -m src.main
```

On first run the device-code flow prints something like:

```
To sign in, use a web browser to open https://login.microsoft.com/device
and enter the code XXXXXXXXX to authenticate.
```

Approve it in any browser with the account you registered at
[openindustrialdata.com](https://openindustrialdata.com). The token is cached in
`.cognite_token_cache.json` (git-ignored), so later runs are non-interactive.

### Command line

```bash
python -m src.main --help

# defaults: 60 days of 1-minute averages around 23-KA-9101
python -m src.main

# an explicit window
python -m src.main --start 2024-01-01 --end 2024-03-01

# raw resolution — much larger; gated behind a confirmation prompt
python -m src.main --start 2024-02-01 --end 2024-02-08 --raw

# a different machine
python -m src.main --asset 23-KA-9102

# structure only: no telemetry values, no document bytes
python -m src.main --skip-datapoints --skip-files

# expand to two years of hourly averages
python -m src.main --start 2023-01-01 --granularity 1h --yes
```

| Flag | Purpose |
| --- | --- |
| `--asset` | Target tag or external id. Default `23-KA-9101`. |
| `--start` / `--end` | ISO 8601 window. `--end` defaults to the newest datapoint actually present. |
| `--lookback-days` | Days before `--end` when `--start` is omitted. Default 60. |
| `--aggregate` / `--granularity` | Server-side aggregate and bucket. Default `average` @ `1m`. |
| `--raw` | Raw datapoints instead of aggregates. |
| `--skip-datapoints` / `--skip-files` | Metadata only. |
| `--max-files` | Cap on documents downloaded. Default 60. |
| `-y, --yes` | Skip the large-download confirmation. |

### How to expand the dataset

The default is **60 days of 1-minute averages** — a few hundred thousand points
per tag, which downloads in under a minute and is plenty for a first anomaly
model. To scale up, in increasing order of cost:

1. **Longer window, same resolution:** `--start 2023-01-01` (≈12× the data).
2. **Coarser aggregate over years:** `--granularity 1h --start 2020-01-01`.
   Hourly over five years is *smaller* than 1-minute over 60 days.
3. **Raw resolution, short window:** `--raw --start 2024-02-01 --end 2024-02-08`.
   Use this for fast phenomena — surge cycles, vibration transients, trips.
4. **Raw resolution, long window:** possible, but the pipeline will stop and
   ask first. Expect tens of GB.

Runs are incremental at the *file* level, not the row level: re-running
overwrites each tag's Parquet file. To build up history, extract windows into
separate directories (override `VALHALL_*` paths) or add a watermark table —
see the production notes in `src/main.py`.

---

## Output layout

```
data/
├── assets/
│   ├── assets.parquet / .csv            # selected hierarchy, with materialised paths
├── timeseries/
│   ├── timeseries_metadata.parquet/.csv # tag catalogue: units, asset links, is_step
│   ├── datapoints_summary.parquet/.csv  # rows + bytes per tag
│   └── datapoints/
│       └── <external_id>.parquet        # one file per tag, UTC, tz-aware
├── events/
│   └── events.parquet / .csv            # event records + profiled metadata
├── files/
│   ├── files_metadata.parquet / .csv    # document catalogue with relevance scores
│   ├── files_downloaded.parquet / .csv  # what actually landed
│   └── documents/                       # the document bytes
├── relationships/
│   ├── relationships.parquet            # explicit CDF Relationship edges
│   ├── data_model_inventory.parquet     # FDM spaces/models, if any
│   └── implicit_edges.parquet           # PART_OF / MEASURES / AFFECTS / DOCUMENTS
└── data_quality_report.md               # generated assessment
```

Load it:

```python
import pandas as pd, polars as pl

assets = pd.read_parquet("data/assets/assets.parquet")
tags   = pd.read_parquet("data/timeseries/timeseries_metadata.parquet")

# one tag
df = pd.read_parquet("data/timeseries/datapoints/pi__163657.parquet")

# all tags, lazily, without loading them into memory
lf = pl.scan_parquet("data/timeseries/datapoints/*.parquet")
print(lf.group_by("external_id").agg(pl.len()).collect())
```

Build the knowledge graph:

```bash
python notebooks/build_knowledge_graph.py
```

---

## Learning notes

### Asset IDs vs external IDs

Every CDF resource has two identifiers, and conflating them is the most common
way an industrial pipeline breaks.

| | `id` | `external_id` |
| --- | --- | --- |
| Origin | generated by CDF | supplied by the source system |
| Scope | unique within **one project** | stable across projects and re-ingests |
| Stability | changes on re-ingest | survives re-ingest |
| Human meaning | none | the engineering tag (`23-KA-9101`) |
| Used for | most API *filters* | joining to SAP, the DCS, P&IDs |

The rule this project follows: **query with ids, persist both, join on
external_id.** Integer ids are what `asset_ids` and `asset_subtree_ids` filters
accept, so you need them at query time. But `external_id` is what still means
something in six months against a rebuilt project — so it is the key in every
saved table and the node key in the graph.

### Why the asset hierarchy matters

For a troubleshooting agent the hierarchy is the reasoning substrate:

- **Localisation.** A vibration spike on one sensor is meaningless alone. The
  hierarchy says it belongs to the compressor's drive-end bearing, which belongs
  to `23-KA-9101`, which belongs to the first-stage train. That chain is what
  turns an anomaly into a fault hypothesis.
- **Correlation scope.** It tells you which other signals are *physically
  plausible* neighbours. Correlating a compressor bearing with an unrelated
  water-injection pump is noise; with its own lube-oil supply pressure, signal.
- **Aggregation.** Events and documents attach at different levels of the tree,
  so rolling up a subtree is how you collect all the context for one machine.

This extractor therefore takes the target's **ancestors** (which train, which
platform), its **descendants** (motor, bearings, lube oil, seals, anti-surge)
*and* one level of **siblings** — because when discharge temperature rises the
answer often lives in the aftercooler or suction scrubber, which are siblings in
the train, not children of the compressor.

### How SCADA tags relate to assets

```
physical sensor → DCS/SCADA tag → historian series → CDF TimeSeries
                                                          │ asset_id
                                                          ▼
                                                      CDF Asset
```

A CDF `TimeSeries` is the *series*, not the sensor. Its `asset_id` is a
many-to-one foreign key to the equipment it measures. That link — not tag-name
convention — is what lets you ask "every measurement on this machine".

In real projects the link is **incomplete**: tags get ingested before the
hierarchy exists, or attached at the train level rather than the machine. So the
quality report counts unlinked series rather than assuming the link is complete.
Two fields matter more than they look:

- **`unit`** — mixing bar and barg, or °C and K, across tags a model treats as
  interchangeable produces physically nonsensical thresholds.
- **`is_step`** — a step series (a valve setpoint) must be **forward-filled**; a
  continuous one (a pressure) may be interpolated. Getting this backwards
  quietly corrupts every resampled feature.

### Why Parquet, not CSV, for telemetry

| | CSV | Parquet |
| --- | --- | --- |
| Types | text only, re-guessed on every read | schema stored in the file |
| Size | baseline | typically 5–20× smaller for sensor data |
| Column reads | must parse every row | reads only the columns asked for |
| Filtering | full scan | row-group min/max statistics skip irrelevant blocks |
| Nulls | usually indistinguishable from `""` or `0` | explicitly distinct |

The null point is the one that bites hardest in condition monitoring: a sensor
reading `0.0` and a sensor *not reporting* mean completely different things — a
tripped machine versus a dead transmitter. CSV usually renders both as an empty
field.

CSV is still written for the small metadata tables, because being able to open
them in a spreadsheet during development is genuinely useful. It is deliberately
**not** written for datapoints.

### How CDF pagination works

Two different mechanisms, commonly confused:

**Resource listing** (assets, time series, events, files) is **cursor-based**.
Each response carries a `nextCursor` you pass back for the next page. The SDK
hides it: `limit=None` means "follow cursors to exhaustion". Offset pagination
is deliberately not offered — with concurrent writes, offsets skip and duplicate
rows. For big listings, `partitions=N` splits the keyspace so N workers
paginate disjoint slices in parallel.

**Datapoint retrieval** is **time-window-based**. One request returns at most
100,000 raw points (or 10,000 aggregate rows) *per series*, truncating the
window you asked for; you continue from the last timestamp received. The SDK
does this internally and parallelises across series and sub-windows.

### How time-series batching works

The datapoints endpoint accepts **up to 100 series per request**. Batching at
that limit is the single biggest performance lever — it amortises TLS, auth and
HTTP overhead and lets the SDK's thread pool saturate the connection.

But a batch that is both *wide* (100 series) and *long* (2 years at 1 Hz) would
materialise billions of points in RAM. So this pipeline chunks on **both axes**:

- **width** — `DATAPOINTS_TS_PER_REQUEST` = 100 series per request,
- **length** — `RAW_CHUNK_DAYS` = 7 days per request window.

Each block is fetched, converted, appended to a per-tag `ParquetWriter`, and
released. Peak memory is bounded by one block regardless of history length.
The streaming write matters: the naive version (concatenate every chunk, then
`to_parquet` once) defeats the entire point of chunking.

### Storage layout: one Parquet file per tag

```
data/timeseries/datapoints/<external_id>.parquet
```

Chosen because **the access pattern dominates**. Every downstream consumer —
anomaly detection, feature engineering, plotting one trend — asks for *specific
tags over a time range*. One file per tag makes that a single file open with no
filtering. Additionally, tags are heterogeneous (bar, °C, mm/s, plus string
status tags), so per-tag files let each file carry the schema its data actually
needs instead of upcasting everything into one value column. Extending one
tag's history rewrites one small file.

The alternative — Hive partitioning, `external_id=<tag>/year=<y>/part.parquet` —
becomes the better choice once a single tag's history stops fitting in memory,
or you move to Spark/Athena/DuckDB at scale, because partition keys are pruned
before any file is opened. At this subsystem scale it only adds path complexity.

Every datapoint row carries `external_id`, `timeseries_id` and `unit`, which is
redundant *within* a file but means any single file — or any concatenation — is
interpretable with no reference to the metadata table. Dictionary encoding makes
the repeated strings nearly free.

Timestamps are **tz-aware UTC**, always. Naive timestamps are a liability here:
the moment one analyst assumes local Bergen time and another assumes UTC, the
correlation between a vibration spike and a maintenance record is off by an hour
or two and the conclusion is wrong.

### From P&IDs to a NetworkX or Neo4j graph

Everything this pipeline writes is already edge-shaped:

| Nodes | Source | Key |
| --- | --- | --- |
| `Asset` | `assets.parquet` | `external_id` |
| `TimeSeries` | `timeseries_metadata.parquet` | `external_id` |
| `Event` | `events.parquet` | `external_id` / `event_id` |
| `Document` | `files_metadata.parquet` | `external_id` / `file_id` |

| Edge | Derived from |
| --- | --- |
| `(Asset)-[:PART_OF]->(Asset)` | `assets.parent_external_id` |
| `(TimeSeries)-[:MEASURES]->(Asset)` | `timeseries.asset_external_id` |
| `(Event)-[:AFFECTS]->(Asset)` | `events.asset_external_ids` |
| `(Document)-[:DOCUMENTS]->(Asset)` | `files.asset_external_ids` |

`relationships/implicit_edges.parquet` is that edge list, pre-built.
`notebooks/build_knowledge_graph.py` turns it into a `nx.MultiDiGraph` (multi-
because two nodes can share several relationship types; di- because `PART_OF`
is not symmetric) and demonstrates the query that matters:

```python
context_for_tag(graph, "pi:163661")
# → tag description, the asset it measures, that asset's ancestors,
#   its sibling measurements, events that touched it, documents that describe it
```

That single function is why the graph exists: it turns *"pi:163661 looks
wrong"* into *"discharge temperature on the first-stage compressor looks wrong;
here are the other signals on that machine, what happened to it recently, and
the P&ID that describes it."*

For Neo4j, load the same frames — `MERGE` nodes on `external_id` with a
uniqueness constraint, then the edge table. Above a few million edges use
`neo4j-admin database import` rather than `LOAD CSV`.

**The honest limitation:** CDF metadata gives you *containment and attachment*
(what belongs to what, what documents what). It does **not** give you **process
topology** — which stream flows into which vessel, and therefore what can
physically cause what. That has to be extracted from the drawings themselves,
via CDF's P&ID contextualisation service or your own symbol/line detection. That
step is what upgrades the graph from "things belonging to things" to "a process
you can reason about causally", and it is the highest-value piece of work
remaining.

### What would change in production

| Aspect | Here | Production |
| --- | --- | --- |
| Auth | device code (a human approves) | `client_credentials` with a service principal, secret from Key Vault / Secrets Manager, read-only scoped capabilities |
| Scheduling | one-shot CLI | orchestrated task with retries and alerting |
| Incrementality | fixed `--start`/`--end` | per-tag watermark table; resume from last success |
| Ingestion | polled windows | CDF datapoint subscriptions for live monitoring |
| Storage | local Parquet | object storage + Delta/Iceberg (concurrent readers, schema evolution, time travel) |
| Quality | a report a human reads | assertions that **fail the run** |
| Secrets | `.env` | injected at runtime, never on disk |

---

## Testing

The test suite runs entirely offline against a fake CDF built from **real SDK
data classes** (`tests/mock_cdf.py`):

```bash
pytest                                        # 26 tests, no credentials needed
python tests/run_offline_pipeline.py          # full pipeline against the mock
```

Using real `Asset` / `TimeSeries` / `Datapoints` / `LatestDatapoint` objects
rather than `MagicMock` is the point: a mock returns a `Mock` for any attribute
and verifies nothing, whereas this harness fails if the pipeline reads a field
that does not exist or assumes the wrong shape. It caught three genuine bugs
during development:

1. `DatapointsList` is a `collections.UserList`, **not** a builtin `list`, so
   `isinstance(result, list)` is `False` — which would have mis-handled every
   multi-series datapoint batch.
2. `float("nan")` is **truthy**, so `if row["parent_external_id"]:` emitted a
   phantom `"nan"` node from the root asset, corrupting the graph.
3. `LatestDatapoint.timestamp` is a scalar `datetime` in SDK 8.x, not a list as
   in earlier majors.

The synthetic telemetry deliberately embeds a frozen transmitter, duplicate
timestamps, a historian gap and a decommissioned tag, so the quality checks are
verified to fire on real defects — and *not* to fire on healthy channels.

## Project structure

```
cognite_valhall/
├── README.md
├── requirements.txt
├── .env.example                   # public config; copy to .env
├── .gitignore
├── pytest.ini
├── src/
│   ├── config.py                  # all tunables; nothing else reads os.environ
│   ├── cognite_client.py          # OIDC auth (device code / interactive / client creds)
│   ├── discover_assets.py         # find 23-KA-9101, walk ancestors/descendants/siblings
│   ├── download_assets.py         # persist hierarchy with materialised paths
│   ├── download_timeseries.py     # tag catalogue, coverage probe, chunked datapoint download
│   ├── download_events.py         # events + honest classification of what they are
│   ├── download_files.py          # document catalogue, P&ID scoring, selective download
│   ├── download_relationships.py  # explicit + implicit graph edges
│   ├── storage.py                 # Parquet/CSV writers, UTC and filename helpers
│   ├── quality_report.py          # generates data_quality_report.md
│   └── main.py                    # the 9-stage pipeline
├── tests/
│   ├── mock_cdf.py                # offline CDF fake built on real SDK classes
│   ├── test_pipeline.py           # 26 unit + integration tests
│   └── run_offline_pipeline.py    # end-to-end run with no network
├── notebooks/
│   └── build_knowledge_graph.py   # Parquet → NetworkX, with the troubleshooting query
└── data/                          # generated output (git-ignored)
```
