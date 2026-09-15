"""Stage 11 - Asset context.

Responsible for:
- Loading asset information from data/assets/
- Looking up criticality, redundancy, known failure modes, and operational
  workarounds for a given asset

The same risk score can mean very different things for a critical pump with
no backup versus a non-critical fan with a spare on the shelf.
"""

import json
from pathlib import Path

from src.config import ASSETS_DIR

ASSET_CONTEXT_FILE = ASSETS_DIR / "asset_context.json"


def load_asset_context(asset_id: str, path: Path = ASSET_CONTEXT_FILE) -> dict:
    """Look up asset context for one asset by asset_id.

    Args:
        asset_id: The asset to look up, e.g. "pump-01".
        path: Path to the asset context JSON file.

    Returns:
        The asset's context record (criticality, redundancy, etc.).

    Raises:
        FileNotFoundError: If the asset context file doesn't exist.
        KeyError: If asset_id isn't in the file - fails loudly rather than
            returning a made-up default, since guessing an asset's
            criticality would be a safety-relevant mistake.
    """
    if not path.exists():
        raise FileNotFoundError(f"Asset context file not found: {path}")

    with open(path) as f:
        assets = json.load(f)

    for asset in assets:
        if asset["asset_id"] == asset_id:
            return asset

    known_ids = [a["asset_id"] for a in assets]
    raise KeyError(f"Unknown asset_id: {asset_id!r}. Known assets: {known_ids}")


def list_asset_ids(path: Path = ASSET_CONTEXT_FILE) -> list[str]:
    """List every asset_id in the context file, for populating a picker."""
    if not path.exists():
        raise FileNotFoundError(f"Asset context file not found: {path}")

    with open(path) as f:
        assets = json.load(f)

    return [asset["asset_id"] for asset in assets]


def has_redundancy(asset: dict) -> bool:
    """Interpret an asset's "redundancy" field as a yes/no signal.

    The context file stores redundancy as a human-readable description
    (e.g. "1 standby pump (pump-02b)" or "None") rather than a boolean,
    because that description is also what a person reviewing the
    recommendation wants to read. This is the one place that description
    gets turned into the boolean decide_maintenance_action() needs.
    """
    return asset["redundancy"].strip().lower() != "none"


if __name__ == "__main__":
    context = load_asset_context("pump-01")
    print(context)
