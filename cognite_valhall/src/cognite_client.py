"""Authenticated CogniteClient construction.

DESIGN NOTE — why three auth flows, and why device code is the default?
-----------------------------------------------------------------------
Cognite Data Fusion uses OpenID Connect. There are two fundamentally different
ways to prove who you are, and the choice is not cosmetic:

* **Delegated / user auth** (device code, interactive browser). The token
  represents *you*. This is correct for Open Industrial Data, because access is
  granted to individual signed-up users, and it is correct for exploratory work
  generally — the audit log shows a human ran the query.

* **Application auth** (client credentials). The token represents a *service
  principal* and needs a client secret. This is what a production extractor or
  scheduled job uses, because there is no human at a browser.

We default to **device code** rather than interactive browser redirect because
device code works over SSH and inside containers: it prints a URL and a short
code, and you approve on any device. Interactive redirect needs a browser on
the same host that can reach ``localhost:53000``.

What would change in production
-------------------------------
A production extractor would use ``client_credentials`` with the secret held in
a managed secret store (Azure Key Vault, AWS Secrets Manager, Kubernetes
secret) and injected as an environment variable at runtime — never a ``.env``
file on disk. It would also run under a service principal scoped to read-only
capabilities on exactly the data sets it needs, rather than a human's full
access.
"""

from __future__ import annotations

import logging
import sys

from cognite.client import ClientConfig, CogniteClient, global_config
from cognite.client.credentials import (
    CredentialProvider,
    OAuthClientCredentials,
    OAuthDeviceCode,
    OAuthInteractive,
)
from cognite.client.exceptions import CogniteAPIError, CogniteAuthError

from . import config

logger = logging.getLogger(__name__)


def _apply_global_settings() -> None:
    """Tune SDK-wide retry behaviour before any client is built.

    The SDK already retries idempotent requests on 429/502/503/504 with
    exponential backoff. We raise the retry count because a long telemetry
    download is a marathon: a single transient 503 two hours in should not
    abort the run. We also silence the PyPI version check so the pipeline does
    not make an outbound call to pypi.org on every start (which would fail in
    a locked-down network and add noise to the logs).
    """
    global_config.max_retries = config.MAX_RETRIES
    global_config.max_retry_backoff = config.MAX_RETRY_BACKOFF
    global_config.disable_pypi_version_check = True


def build_credentials() -> CredentialProvider:
    """Return the credential provider selected by ``COGNITE_AUTH_FLOW``."""
    flow = config.AUTH_FLOW

    if flow == "client_credentials":
        if not config.COGNITE_CLIENT_SECRET:
            raise CogniteAuthError(
                "COGNITE_AUTH_FLOW=client_credentials requires COGNITE_CLIENT_SECRET "
                "to be set in the environment (or .env)."
            )
        logger.info("Auth flow: client credentials (service principal)")
        return OAuthClientCredentials(
            token_url=config.TOKEN_URL,
            client_id=config.COGNITE_CLIENT_ID,
            client_secret=config.COGNITE_CLIENT_SECRET,
            scopes=config.SCOPES,
        )

    if flow == "interactive":
        logger.info("Auth flow: interactive browser redirect")
        return OAuthInteractive(
            authority_url=config.AUTHORITY_URL,
            client_id=config.COGNITE_CLIENT_ID,
            scopes=config.SCOPES,
            token_cache_path=config.TOKEN_CACHE_PATH,
        )

    if flow != "device_code":
        logger.warning("Unknown COGNITE_AUTH_FLOW=%r; falling back to device_code", flow)

    logger.info("Auth flow: device code (browser approval on any device)")
    # The token cache means you approve once and subsequent runs reuse the
    # refresh token silently. Delete the cache file to force a fresh login.
    return OAuthDeviceCode(
        authority_url=config.AUTHORITY_URL,
        client_id=config.COGNITE_CLIENT_ID,
        scopes=config.SCOPES,
        token_cache_path=config.TOKEN_CACHE_PATH,
    )


def get_client() -> CogniteClient:
    """Build a CogniteClient pointed at the configured project.

    Note we pass ``base_url`` explicitly rather than ``cluster``. Open
    Industrial Data lives on the original ``api.cognitedata.com`` host, which is
    not expressible as a modern ``<cluster>.cognitedata.com`` name, so the
    explicit URL is the unambiguous choice.
    """
    _apply_global_settings()

    client_config = ClientConfig(
        client_name=config.COGNITE_CLIENT_NAME,
        project=config.COGNITE_PROJECT,
        credentials=build_credentials(),
        base_url=config.COGNITE_BASE_URL,
    )
    return CogniteClient(client_config)


def verify_connection(client: CogniteClient) -> dict[str, object]:
    """Confirm the token works and report what it can actually see.

    Calling ``token/inspect`` first is a deliberate fail-fast: it is a cheap
    request that distinguishes the three failure modes that otherwise look
    identical several minutes into a download —

      * bad credentials (401),
      * valid token but no access to this project (403 / project absent from
        the inspect response),
      * valid token and access, but missing a specific capability such as
        ``filesAcl:READ``.

    Returning the capability summary also lets later stages degrade gracefully
    instead of crashing: if the token cannot read Files, we skip documents and
    say so in the report.
    """
    try:
        token_info = client.iam.token.inspect()
    except CogniteAPIError as exc:
        raise CogniteAuthError(
            f"Token inspection failed against {config.COGNITE_BASE_URL} "
            f"(project={config.COGNITE_PROJECT}): {exc}"
        ) from exc

    projects = [p.url_name for p in (token_info.projects or [])]
    if config.COGNITE_PROJECT not in projects:
        logger.warning(
            "Token is valid but project %r is not in its project list %s. "
            "Have you signed up at openindustrialdata.com with this account?",
            config.COGNITE_PROJECT,
            projects,
        )

    capabilities: set[str] = set()
    for cap in token_info.capabilities or []:
        raw = cap.dump() if hasattr(cap, "dump") else cap
        if isinstance(raw, dict):
            capabilities.update(raw.keys())

    logger.info(
        "Authenticated to project=%s subject=%s (%d capability groups)",
        config.COGNITE_PROJECT,
        getattr(token_info, "subject", "unknown"),
        len(token_info.capabilities or []),
    )
    return {
        "subject": getattr(token_info, "subject", None),
        "projects": projects,
        "capabilities": sorted(capabilities),
    }


def get_verified_client() -> CogniteClient:
    """``get_client`` plus a connection check, with an actionable error message."""
    client = get_client()
    try:
        verify_connection(client)
    except CogniteAuthError as exc:
        logger.error("Authentication failed: %s", exc)
        logger.error(
            "Checklist:\n"
            "  1. Sign up for Open Industrial Data at https://openindustrialdata.com\n"
            "  2. Confirm COGNITE_PROJECT / COGNITE_BASE_URL in your .env\n"
            "  3. Delete %s to force a fresh login\n"
            "  4. If you are on a locked-down network, confirm %s is reachable",
            config.TOKEN_CACHE_PATH,
            config.COGNITE_BASE_URL,
        )
        sys.exit(2)
    return client
