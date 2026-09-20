#!/usr/bin/env python
"""Write the OpenAPI spec from the reference gateway.

The spec is checked in rather than produced at build time, and
``tests/test_openapi_drift.py`` fails if this script would change it, so a
contract change cannot land silently. The phase-4 SDKs will be generated from
the checked-in file. Run this after any change to the contract.
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO / "spec" / "openapi.json"


def render() -> str:
    from trigon.server.app import build_app

    return json.dumps(build_app().openapi(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    out = pathlib.Path(argv[1]) if len(argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render())
    print(f"wrote {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
