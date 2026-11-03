"""Error envelope {"error","code","details"}: checked on a minimal app wired with core.errors,
and on the real app when it can be imported on this machine."""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

errors = pytest.importorskip("core.errors")


def check_shape(body):
    assert {"error", "code", "details"} <= set(body)
    assert isinstance(body["error"], str) and isinstance(body["code"], str)


@pytest.fixture()
def mini():
    app = FastAPI()
    errors.register_exception_handlers(app)

    @app.get("/voices/{vid}")
    def get_voice(vid: str):
        raise HTTPException(404, "Voice not found")

    @app.get("/app-error")
    def app_error():
        raise errors.AppError("CUSTOM", "custom failure", 418, {"x": 1})

    @app.get("/boom")
    def boom():
        raise RuntimeError("secret internals")

    @app.get("/q")
    def q(n: int):
        return n

    return TestClient(app, raise_server_exceptions=False)


def test_404_shape(mini):
    r = mini.get("/voices/unknown")
    assert r.status_code == 404
    body = r.json()
    check_shape(body)
    assert body["code"] == "NOT_FOUND"


def test_app_error_shape(mini):
    r = mini.get("/app-error")
    assert r.status_code == 418
    assert r.json()["code"] == "CUSTOM" and r.json()["details"] == {"x": 1}


def test_validation_shape(mini):
    r = mini.get("/q", params={"n": "abc"})
    assert r.status_code == 422
    check_shape(r.json())
    assert r.json()["code"] == "VALIDATION_ERROR"


def test_unhandled_does_not_leak(mini):
    r = mini.get("/boom")
    assert r.status_code == 500
    check_shape(r.json())
    assert "secret internals" not in r.text


def test_real_app_unknown_voice_404(app_client):
    r = app_client.delete("/api/voices/does-not-exist", headers={"storage-mode": "local"})
    assert r.status_code == 404, r.text
    check_shape(r.json())
