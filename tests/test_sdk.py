"""The generated Python SDK, run against a real gateway.

The SDK is generated from ``spec/openapi.json`` so there is only ever one
definition of the contract. Two things have to hold for that to be worth
anything: the checked-in file must be what the generator produces from the
current spec, and the client must actually work against the server the spec
describes. A client that parses a fixture proves neither.
"""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from trigon.server.app import build_app
from trigon.server.config import ServerConfig

ROOT = pathlib.Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "scripts" / "generate_sdk.py"
sys.path.insert(0, str(ROOT / "sdk" / "python"))

trigon_client = pytest.importorskip("trigon_client", reason="the SDK has not been generated")


def test_the_checked_in_sdk_is_what_the_spec_produces():
    """A contract change that is not regenerated fails here, in the same
    commit — the discipline test_openapi_drift.py applies to the spec itself."""
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_the_sdk_needs_nothing_but_the_standard_library():
    """The first thing a prospective user installs. Every third-party import
    here is a dependency they did not ask for."""
    source = (ROOT / "sdk" / "python" / "trigon_client" / "_generated.py").read_text()
    imports = {
        line.split()[1].split(".")[0]
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "__future__" not in line
    }
    assert imports <= set(sys.stdlib_module_names), f"non-stdlib imports: {imports}"


def test_the_sdk_pins_the_contract_version_it_was_generated_from():
    import json

    spec = json.loads((ROOT / "spec" / "openapi.json").read_text())
    assert trigon_client.CONTRACT_VERSION == spec["info"]["version"]


# -- against a live gateway --------------------------------------------------


class _TestClientTransport:
    """Point the generated client at an in-process app.

    The client speaks ``urllib`` so it has no injection seam, which is the
    right trade for an SDK and inconvenient here. Patching the one transport
    method keeps every layer above it -- request shaping, parsing, error
    mapping -- under test.
    """

    def __init__(self, client: TestClient) -> None:
        self.client = client

    def __call__(self, method: str, path: str, body: dict | None = None):
        response = self.client.request(method, path, json=body)
        if response.status_code >= 400:
            raise trigon_client.TrigonError(response.status_code, response.json())
        return response.json()


@pytest.fixture
def client() -> trigon_client.TrigonClient:
    sdk = trigon_client.TrigonClient("http://testserver")
    sdk._call = _TestClientTransport(TestClient(build_app(ServerConfig(backend="lexical"))))
    return sdk


def test_it_answers_every_primitive_as_its_declared_type(client):
    response = client.systemone(
        state="the card payment was declined at the till",
        questions={
            "intent": trigon_client.choice(
                "Route this ticket.",
                [
                    {"name": "card_declined", "criteria": "a card transaction was refused"},
                    {"name": "lost_luggage", "criteria": "baggage missing after a flight"},
                ],
            ),
            "severity": trigon_client.score(
                "How severe is this?",
                [{"name": "low", "value": 1.0}, {"name": "high", "value": 5.0}],
            ),
            "urgent": trigon_client.noul("Needs a human within the hour?"),
        },
    )

    assert isinstance(response.answers["intent"], trigon_client.ChoiceAnswer)
    assert isinstance(response.answers["severity"], trigon_client.ScoreAnswer)
    assert isinstance(response.answers["urgent"], trigon_client.NoulAnswer)

    assert response.answers["intent"].selected == "card_declined"
    assert sum(response.answers["intent"].probabilities.values()) == pytest.approx(1.0)
    # Score's expectation cannot land outside the scale the request declared.
    assert 1.0 <= response.answers["severity"].score <= 5.0
    # A Noul has no confidence field, by design, and the SDK must not invent one.
    assert not hasattr(response.answers["urgent"], "confidence")
    assert 0.0 <= response.answers["urgent"].probability <= 1.0


def test_bare_option_names_are_accepted(client):
    """`choice("...", ["a", "b"])` is what someone tries first."""
    response = client.systemone(
        state="a parcel never arrived",
        questions={"q": trigon_client.choice("Pick one.", ["shipping", "billing"])},
    )
    assert set(response.answers["q"].probabilities) == {"shipping", "billing"}


def test_usage_and_timing_come_back_typed(client):
    response = client.systemone(
        state="hello", questions={"q": trigon_client.noul("Is this a greeting?")}
    )
    assert isinstance(response.usage, trigon_client.Usage)
    assert response.usage.prefill_tokens == (
        response.usage.state_tokens + response.usage.schema_tokens + response.usage.readout_tokens
    )
    # Documented as always 0 until a KV cache exists; the SDK must not imply more.
    assert response.usage.cached_schema_tokens == 0
    assert response.timing.total_ms >= 0.0


def test_the_response_names_a_concrete_build(client):
    response = client.systemone(
        state="hello", questions={"q": trigon_client.noul("Is this a greeting?")}
    )
    assert any(ch.isdigit() for ch in response.model)


def test_a_rejected_request_raises_with_its_status(client):
    with pytest.raises(trigon_client.TrigonError) as caught:
        client.systemone(state="x", questions={"q": trigon_client.choice("pick", ["only"])})
    assert caught.value.status == 422
    assert caught.value.payload


def test_an_oversized_state_raises_413_not_422(client):
    with pytest.raises(trigon_client.TrigonError) as caught:
        client.systemone(state="word " * 80000, questions={"q": trigon_client.noul("ok?")})
    assert caught.value.status == 413


def test_ops_endpoints_work(client):
    health = client.healthz()
    assert health["status"] == "ok"
    assert "calibrated" in health and "trained" in health
    assert client.models()["data"][0]["tier"] == "workhorse"


def test_an_unknown_answer_type_is_refused_rather_than_guessed():
    with pytest.raises(trigon_client.TrigonError, match="unknown answer type"):
        trigon_client.parse_answer({"type": "vibes", "probability": 0.5})


def test_the_generator_is_importable_without_running_it():
    """It reads the spec at call time, not at import, so `--check` in CI cannot
    be broken by an unrelated import error."""
    spec = importlib.util.spec_from_file_location("generate_sdk", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.render)


# -- TypeScript --------------------------------------------------------------

TS = ROOT / "sdk" / "typescript"


def _node() -> str | None:
    import shutil

    return shutil.which("node")


def test_the_typescript_client_typechecks_under_strict_mode():
    """It is generated, so nobody reads it — the compiler is the review.
    Strict mode with exactOptionalPropertyTypes caught a real bug the first
    time: `body: undefined` is not an absent key, and RequestInit refuses it.
    """
    if _node() is None:
        pytest.skip("node is not installed")
    if not (TS / "node_modules").exists():
        pytest.skip("run `npm install` in sdk/typescript to typecheck")
    result = subprocess.run(
        [str(TS / "node_modules" / ".bin" / "tsc"), "--noEmit"],
        cwd=TS,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_typescript_client_answers_against_a_live_gateway():
    """Typechecking proves nothing about the server. This starts the real
    gateway and drives it from node."""
    if _node() is None:
        pytest.skip("node is not installed")

    import threading
    import time

    import uvicorn

    app = build_app(ServerConfig(backend="lexical"))
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8933, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 15
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        assert server.started, "the gateway did not come up"

        result = subprocess.run(
            [
                _node(),
                "--experimental-strip-types",
                str(TS / "smoke.mjs"),
                "http://127.0.0.1:8933",
            ],
            cwd=TS,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "confidence field present: false" in result.stdout
    assert "bad request -> 422" in result.stdout


def test_both_sdks_describe_the_same_contract_version():
    source = (TS / "src" / "generated.ts").read_text()
    assert f'CONTRACT_VERSION = "{trigon_client.CONTRACT_VERSION}"' in source


def test_the_typescript_client_has_no_runtime_dependencies():
    import json

    manifest = json.loads((TS / "package.json").read_text())
    assert not manifest.get("dependencies"), manifest.get("dependencies")
