"""The checked-in spec is what both SDKs are generated from, so it must match
the reference gateway exactly. If this fails, run scripts/export_openapi.py."""

from __future__ import annotations

import json
import pathlib

SPEC = pathlib.Path(__file__).resolve().parent.parent / "spec" / "openapi.json"


def test_checked_in_spec_matches_the_app():
    import sys

    sys.path.insert(0, str(SPEC.parent.parent / "scripts"))
    from export_openapi import render

    assert SPEC.exists(), "spec/openapi.json is missing; run scripts/export_openapi.py"
    assert SPEC.read_text() == render(), "spec/openapi.json is stale; run scripts/export_openapi.py"


def test_spec_documents_the_three_primitives():
    spec = json.loads(SPEC.read_text())
    schemas = spec["components"]["schemas"]
    for name in ("ChoiceQuestion", "ScoreQuestion", "NoulQuestion"):
        assert name in schemas, f"{name} missing from the published contract"
    assert "/v1/systemone" in spec["paths"]


def test_noul_answer_has_no_confidence_field_in_the_published_contract():
    spec = json.loads(SPEC.read_text())
    assert "confidence" not in spec["components"]["schemas"]["NoulAnswer"]["properties"]
