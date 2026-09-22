"""The three status codes a migrating caller's retry loop branches on.

`docs/compat.md` listed 401, 429 and 529 under "what is not there", and that
was a real gap: a caller brings a retry loop written against those codes, and
a server that never emits them leaves that loop untested. A load test against
a gateway that *queues* where the incumbent *sheds* measures the wrong thing,
optimistically.

All three stay off unless configured. The tests below check both halves of
that — that an unconfigured gateway is unchanged, and that a configured one
emits the code *and the headers* a client needs to act on it.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from trigon.server.app import build_app
from trigon.server.compat import build_compat_app
from trigon.server.config import ServerConfig
from trigon.server.limits_middleware import RateLimiter

REQUEST = {
    "state": "my card was declined",
    "questions": {"urgent": {"type": "noul", "instructions": "Needs a human?"}},
}
COMPAT_REQUEST = {**REQUEST, "model": "jev-latest"}


def _native(**kwargs) -> TestClient:
    return TestClient(build_app(ServerConfig(backend="lexical", **kwargs)))


def _compat(**kwargs) -> TestClient:
    return TestClient(build_compat_app(ServerConfig(backend="lexical", **kwargs)))


# -- off unless configured --------------------------------------------------


def test_an_unconfigured_gateway_authenticates_nothing():
    """The honest default: a self-hosted gateway does not invent a policy its
    operator did not choose."""
    with _native() as http:
        assert http.post("/v1/systemone", json=REQUEST).status_code == 200
        assert http.post("/v1/systemone", json=REQUEST, headers={}).status_code == 200


def test_an_empty_key_list_means_no_auth_not_no_access():
    """An empty set would reject every request, which is a configuration
    mistake that looks exactly like a total outage."""
    assert ServerConfig.from_env({"TRIGON_API_KEYS": ""}).api_keys is None
    assert ServerConfig.from_env({"TRIGON_API_KEYS": "   "}).api_keys is None
    assert ServerConfig.from_env({"TRIGON_API_KEYS": "a, b ,"}).api_keys == frozenset({"a", "b"})


@pytest.mark.parametrize("raw", ["0", "-1"])
def test_a_nonsense_limit_is_refused_at_startup_not_at_request_time(raw):
    with pytest.raises(ValueError):
        ServerConfig.from_env({"TRIGON_RATE_PER_MINUTE": raw})


def test_a_rate_of_zero_is_refused_rather_than_rejecting_everything():
    with pytest.raises(ValueError):
        RateLimiter(0)


# -- 401 --------------------------------------------------------------------


def test_a_missing_or_wrong_key_is_401_with_a_scheme_a_client_can_act_on():
    with _native(api_keys=frozenset({"secret"})) as http:
        for headers in ({}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic x"}):
            response = http.post("/v1/systemone", json=REQUEST, headers=headers)
            assert response.status_code == 401
            # Without this a client cannot tell which scheme to retry under.
            assert response.headers["WWW-Authenticate"] == "Bearer"
            assert response.json()["error"]["type"] == "authentication_error"
        assert (
            http.post(
                "/v1/systemone", json=REQUEST, headers={"Authorization": "Bearer secret"}
            ).status_code
            == 200
        )


def test_the_compat_path_authenticates_too():
    """It is the path a migrating caller actually points at."""
    with _compat(api_keys=frozenset({"secret"})) as http:
        assert http.post("/v1/systemone", json=COMPAT_REQUEST).status_code == 401
        assert (
            http.post(
                "/v1/systemone", json=COMPAT_REQUEST, headers={"Authorization": "Bearer secret"}
            ).status_code
            == 200
        )


def test_healthz_stays_open():
    """A liveness probe that needs a credential reports the credential's
    health. An operator debugging a 401 storm needs one endpoint that answers.
    """
    with _native(api_keys=frozenset({"secret"})) as http:
        assert http.get("/healthz").status_code == 200


# -- 429 --------------------------------------------------------------------


def test_the_limit_sheds_with_retry_after_and_a_budget():
    with _native(rate_per_minute=3) as http:
        for _ in range(3):
            assert http.post("/v1/systemone", json=REQUEST).status_code == 200
        response = http.post("/v1/systemone", json=REQUEST)
        assert response.status_code == 429
        assert response.json()["error"]["type"] == "rate_limit_error"
        assert int(response.headers["Retry-After"]) >= 1
        assert response.headers["X-RateLimit-Limit"] == "3"
        assert response.headers["X-RateLimit-Remaining"] == "0"


def test_retry_after_never_lands_on_the_boundary_it_is_racing():
    """A client retrying at exactly the reset instant races the window and is
    rejected again, which reads to them as the limiter lying. Rounded up."""
    limiter = RateLimiter(1, window_seconds=60.0)
    limiter.check("k", now=10.5)
    _, _, reset_in = limiter.check("k", now=10.5)
    assert reset_in == pytest.approx(49.5)
    assert int(reset_in) + 1 > reset_in


def test_the_window_resets():
    limiter = RateLimiter(2, window_seconds=10.0)
    assert limiter.check("k", now=0.0)[0]
    assert limiter.check("k", now=1.0)[0]
    assert not limiter.check("k", now=2.0)[0]
    # Next window.
    assert limiter.check("k", now=11.0)[0]


def test_callers_are_counted_separately():
    limiter = RateLimiter(1, window_seconds=10.0)
    assert limiter.check("alice", now=0.0)[0]
    assert not limiter.check("alice", now=0.1)[0]
    assert limiter.check("bob", now=0.1)[0], "one caller's burst must not shed another's traffic"


def test_a_key_is_counted_by_token_not_by_token_and_address():
    """A caller rotating keys behind one address would otherwise get a fresh
    budget per rotation, which is the failure a limiter exists to prevent."""
    with _native(api_keys=frozenset({"a", "b"}), rate_per_minute=1) as http:
        assert (
            http.post(
                "/v1/systemone", json=REQUEST, headers={"Authorization": "Bearer a"}
            ).status_code
            == 200
        )
        assert (
            http.post(
                "/v1/systemone", json=REQUEST, headers={"Authorization": "Bearer a"}
            ).status_code
            == 429
        )
        # A different key is a different budget, from the same address.
        assert (
            http.post(
                "/v1/systemone", json=REQUEST, headers={"Authorization": "Bearer b"}
            ).status_code
            == 200
        )


# -- 529 --------------------------------------------------------------------


def test_overload_sheds_as_529_not_503():
    """The contract this mirrors uses 529 for overload, and a caller's backoff
    branches on that number rather than on the word.

    Driven against a real uvicorn socket with two concurrent callers, because
    the first version of this test asserted that a *lone* request succeeded
    and called that a shedding test. It passed without the guard installed at
    all, which is the definition of a test that measures nothing.
    """
    import threading
    import urllib.error
    import urllib.request

    import uvicorn
    from fastapi import FastAPI

    from trigon.server.limits_middleware import install_guards

    holding = threading.Event()
    release = threading.Event()
    app = FastAPI()

    @app.get("/v1/slow")
    def slow() -> dict:
        holding.set()
        release.wait(timeout=10)
        return {"ok": True}

    install_guards(app, max_concurrent=1)
    config = uvicorn.Config(app, host="127.0.0.1", port=8123, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen("http://127.0.0.1:8123/healthz", timeout=0.5)
                break
            except Exception:  # noqa: BLE001 - waiting for the socket to bind
                time.sleep(0.05)

        codes: list[int] = []

        def call() -> None:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8123/v1/slow", timeout=15) as r:
                    codes.append(r.status)
            except urllib.error.HTTPError as exc:
                codes.append(exc.code)
                assert exc.headers["Retry-After"] == "1"

        first = threading.Thread(target=call)
        first.start()
        assert holding.wait(timeout=10), "the first request never reached the handler"
        # The first is now in flight and holding the only slot.
        second = threading.Thread(target=call)
        second.start()
        second.join(timeout=10)
        release.set()
        first.join(timeout=10)
    finally:
        release.set()
        server.should_exit = True
        thread.join(timeout=10)

    assert 529 in codes, f"the second caller was queued rather than shed: {codes}"
    assert 200 in codes, f"the first caller should have been served: {codes}"


def test_the_counter_is_locked_because_a_gateway_is_threaded():
    """Two callers sharing a key read-modify-write one counter. Without the
    lock the limit is approximately enforced, which for a shedder means not
    enforced at the moment it matters most."""
    import threading

    limiter = RateLimiter(500, window_seconds=3600.0)
    allowed = []

    def hammer() -> None:
        for _ in range(100):
            allowed.append(limiter.check("shared")[0])

    threads = [threading.Thread(target=hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(allowed) == 500, f"the limit leaked: {sum(allowed)} of 800 allowed, wanted 500"
