"""Central configuration for the Valhall compressor dataset extraction.

DESIGN NOTE — why a single config module?
-----------------------------------------
Every other module imports its settings from here, and nothing else reads
``os.environ`` directly. That gives us three things that matter in an
industrial data pipeline:

1. **Auditability.** When a dataset lands on disk, you need to be able to say
   exactly which CDF project and which time window produced it. One module
   means one place to look.
2. **Testability.** The pipeline can be pointed at a different project (or a
   mock) by changing environment variables, with no code edits.
3. **No secret sprawl.** Secrets are only ever read from the environment (a
   git-ignored ``.env``), never written into source. The public defaults below
   are *not* secrets: they are the published identifiers of Cognite's Open
   Industrial Data project, deliberately world-readable so anyone can log in
   with their own user account.

The Open Industrial Data values below were verified against Cognite's public
documentation/community and against Microsoft's identity platform (the
device-code endpoint accepts this tenant + client ID + scope combination).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv is a convenience, not a hard requirement
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - the pipeline still works via real env vars
    pass


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Resolve everything relative to the project root (the parent of src/) so the
# pipeline writes to the same place regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

ASSETS_DIR = DATA_DIR / "assets"
TIMESERIES_DIR = DATA_DIR / "timeseries"
DATAPOINTS_DIR = TIMESERIES_DIR / "datapoints"
EVENTS_DIR = DATA_DIR / "events"
FILES_DIR = DATA_DIR / "files"
DOCUMENTS_DIR = FILES_DIR / "documents"
RELATIONSHIPS_DIR = DATA_DIR / "relationships"

ALL_DIRS = [
    ASSETS_DIR,
    TIMESERIES_DIR,
    DATAPOINTS_DIR,
    EVENTS_DIR,
    FILES_DIR,
    DOCUMENTS_DIR,
    RELATIONSHIPS_DIR,
]


def ensure_dirs() -> None:
    """Create the output tree if it does not exist. Idempotent."""
    for directory in ALL_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Cognite connection
# --------------------------------------------------------------------------
# These are the *public* identifiers for Open Industrial Data. They are safe to
# commit; the thing that is never committed is your own token/secret.
COGNITE_PROJECT = os.getenv("COGNITE_PROJECT", "publicdata")
COGNITE_BASE_URL = os.getenv("COGNITE_BASE_URL", "https://api.cognitedata.com")
COGNITE_CLIENT_NAME = os.getenv("COGNITE_CLIENT_NAME", "valhall-compressor-extractor")

# Azure AD (Microsoft Entra ID) tenant that hosts Open Industrial Data logins.
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "48d5043c-cf70-4c49-881c-c638f5796997")

# Public client registration published by Cognite for SDK/device-code logins.
# "Public" here is an OAuth term of art: a client that cannot keep a secret, so
# it authenticates the *user* interactively instead of authenticating itself.
COGNITE_CLIENT_ID = os.getenv("COGNITE_CLIENT_ID", "1b90ede3-271e-401b-81a0-a4d52bea3273")

# Only needed for the non-interactive (client-credentials) flow.
COGNITE_CLIENT_SECRET = os.getenv("COGNITE_CLIENT_SECRET")

AUTHORITY_URL = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}"
TOKEN_URL = f"{AUTHORITY_URL}/oauth2/v2.0/token"

# ``/.default`` asks for every scope already consented to for this resource.
# It is the modern v2.0 equivalent of the older ``user_impersonation`` scope.
SCOPES = [f"{COGNITE_BASE_URL}/.default"]

# "device_code" (default) prints a URL + code for you to approve in a browser.
# "interactive" opens a local browser redirect. "client_credentials" is
# machine-to-machine and needs COGNITE_CLIENT_SECRET.
AUTH_FLOW = os.getenv("COGNITE_AUTH_FLOW", "device_code").strip().lower()

# Caching the token means you authenticate once, not once per run. The cache
# holds a refresh token, so it is treated like a secret and git-ignored.
TOKEN_CACHE_PATH = Path(
    os.getenv("COGNITE_TOKEN_CACHE", str(PROJECT_ROOT / ".cognite_token_cache.json"))
)


# --------------------------------------------------------------------------
# Target system
# --------------------------------------------------------------------------
# The first-stage gas compressor train on the Valhall platform.
DEFAULT_ASSET = os.getenv("VALHALL_TARGET_ASSET", "23-KA-9101")

# How far *up* the hierarchy to walk when collecting context. The compressor's
# ancestors tell us which process train and which platform it belongs to.
ANCESTOR_DEPTH = int(os.getenv("VALHALL_ANCESTOR_DEPTH", "6"))

# Whether to include the target's siblings (assets sharing its parent). On
# Valhall the compressor's siblings are the rest of the first-stage train —
# suction scrubber, discharge cooler, and so on — which is exactly the process
# context we want for troubleshooting.
INCLUDE_SIBLINGS = os.getenv("VALHALL_INCLUDE_SIBLINGS", "true").lower() == "true"


# --------------------------------------------------------------------------
# Time window defaults
# --------------------------------------------------------------------------
# Open Industrial Data streams live 1-second-ish process data and has years of
# history. Pulling all of it at full resolution would be tens of gigabytes, so
# we default to a bounded, useful window and make expansion explicit.
DEFAULT_LOOKBACK_DAYS = int(os.getenv("VALHALL_LOOKBACK_DAYS", "60"))

# Above this estimated raw-point count we stop and require --yes, so nobody
# accidentally starts a multi-hour download.
LARGE_DOWNLOAD_THRESHOLD = int(os.getenv("VALHALL_LARGE_DOWNLOAD_THRESHOLD", "50000000"))


# --------------------------------------------------------------------------
# API batching / limits
# --------------------------------------------------------------------------
# The CDF datapoints endpoint accepts up to 100 time series per request. Asking
# for all of them in one call is what makes the SDK's internal parallel fetcher
# efficient, so we batch at the API limit rather than one series at a time.
DATAPOINTS_TS_PER_REQUEST = int(os.getenv("VALHALL_TS_PER_REQUEST", "100"))

# Raw datapoint requests are capped at 100_000 points per series per request.
# The SDK paginates internally, but we additionally chunk by *time* so that a
# single Python object never holds more than a few days of high-rate data.
RAW_CHUNK_DAYS = int(os.getenv("VALHALL_RAW_CHUNK_DAYS", "7"))

# Resource listing limit sentinel: the SDK treats None as "fetch everything,
# paginating as needed".
LIST_ALL = None

MAX_RETRIES = int(os.getenv("VALHALL_MAX_RETRIES", "8"))
MAX_RETRY_BACKOFF = int(os.getenv("VALHALL_MAX_RETRY_BACKOFF", "60"))


# --------------------------------------------------------------------------
# Discovery vocabulary
# --------------------------------------------------------------------------
# IMPORTANT: these are *search hints*, never assumptions. Nothing in the
# pipeline requires any of these words to exist in the dataset. We use them to
# rank and label what we actually discover, and every one that finds nothing is
# reported as "no match" rather than silently dropped.
SUBSYSTEM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "compressor": ("compressor", "kompressor", "compr"),
    "motor": ("motor", "driver"),
    "turbine": ("turbine", "turbin"),
    "gearbox": ("gear", "gearbox"),
    "bearing": ("bearing", "lager"),
    "lubrication": ("lube", "lubrication", "oil", "olje"),
    "dry_gas_seal": ("seal", "dry gas", "dgs", "tetning"),
    "anti_surge": ("surge", "anti-surge", "antisurge", "recycle"),
    "suction": ("suction", "inlet", "scrubber", "sug"),
    "discharge": ("discharge", "outlet", "cooler", "aftercooler"),
    "instrumentation": ("transmitter", "indicator", "sensor", "gauge"),
}

# Terms used to find engineering drawings in CDF Files. Again: hints only.
PID_SEARCH_TERMS: tuple[str, ...] = (
    "P&ID",
    "PID",
    "piping and instrumentation",
    "process flow",
    "PFD",
)

# Filename/metadata signals that a document is likely a P&ID or engineering
# drawing. Used for *flagging and reporting*, not for filtering downloads.
PID_FILENAME_HINTS: tuple[str, ...] = ("pid", "p&id", "pfd", "iso", "isometric", "drawing", "dwg")

# Cap on how many documents we download, so a stray broad match cannot pull the
# entire public file corpus.
MAX_FILES_TO_DOWNLOAD = int(os.getenv("VALHALL_MAX_FILES", "60"))


@dataclass(slots=True)
class RunConfig:
    """Per-run options, populated from the command line.

    Kept separate from the module-level constants above: constants describe
    *the project*, RunConfig describes *this invocation*. Anything a user would
    reasonably vary between runs belongs here.
    """

    asset: str = DEFAULT_ASSET
    start: str | None = None
    end: str | None = None
    lookback_days: int = DEFAULT_LOOKBACK_DAYS
    aggregate: str | None = None
    granularity: str = "1m"
    skip_datapoints: bool = False
    skip_files: bool = False
    max_files: int = MAX_FILES_TO_DOWNLOAD
    assume_yes: bool = False
    tags: list[str] = field(default_factory=list)
