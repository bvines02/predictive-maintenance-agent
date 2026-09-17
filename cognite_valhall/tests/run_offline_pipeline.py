"""Run the whole pipeline against MockCogniteClient. No network, no credentials.

This is the offline verification harness: it patches only the authentication
boundary and lets every other line of the pipeline execute for real.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.mock_cdf import MockCogniteClient  # noqa: E402


def main() -> int:
    from src import cognite_client, main as main_mod

    with_rels = "--with-relationships" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--with-relationships"]

    client = MockCogniteClient(with_relationships=with_rels)
    # Patch the single auth boundary; everything downstream runs unmodified.
    cognite_client.get_verified_client = lambda: client
    main_mod.get_verified_client = lambda: client

    return main_mod.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
