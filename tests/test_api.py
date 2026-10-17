"""API contract tests with a stub analyzer (no models needed): auth, schema, upload limits, persistence.

python -m pytest tests/test_api.py -q
"""
import importlib
import json
import os
import sys
import time
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# exact schema from the spec (section 10); extra keys allowed only inside violation objects
SPEC_SCHEMA = {
    "type": "object",
    "required": ["job_id", "status", "verdict", "violations"],
    "additionalProperties": False,
    "properties": {
        "job_id": {"type": "string"}, "status": {"const": "done"},
        "verdict": {"enum": ["violation", "no_violation"]},
        "violations": {"type": "array", "items": {
            "type": "object", "required": ["type", "timecode_sec", "description", "evidence_frames"],
            "properties": {
                "type": {"type": "string"}, "timecode_sec": {"type": "number"}, "description": {"type": "string"},
                "evidence_frames": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
                    "type": "object", "required": ["timestamp_sec", "image_base64"],
                    "properties": {"timestamp_sec": {"type": "number"}, "image_base64": {"type": "string", "pattern": "^/9j/"}}}},
            }}},
    },
}

FAKE = {"job_id": "", "status": "done", "verdict": "violation", "violations": [{
    "type": "Нарушение порядка хранения наличных денег", "timecode_sec": 10.0, "description": "test",
    "evidence_frames": [{"timestamp_sec": 10.0, "image_base64": "/9j/4AAQSkZJRg=="}], "type_code": "V03"}],
    "_debug": {"timing_sec": {"total": 0.1}}}


class StubAnalyzer:
    def analyze(self, path, job_id="", progress=None):
        progress("checks", 0.6)
        if Path(path).read_bytes()[:4] == b"FAIL":
            raise RuntimeError("decoder error")
        return json.loads(json.dumps(FAKE))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    monkeypatch.setenv("FRONTEND_DIR", str(tmp_path / "no-ui"))
    import service.app as app_mod
    app_mod = importlib.reload(app_mod)

    def fake_load():
        app_mod.state.update(analyzer=StubAnalyzer(), models={"vlm": "stub"})
        app_mod.worker.ready.set()
    monkeypatch.setattr(app_mod, "load_models", fake_load)
    from fastapi.testclient import TestClient
    with TestClient(app_mod.app) as c:
        yield c


def wait_done(c, jid, key="test-key"):
    for _ in range(100):
        r = c.get(f"/api/v1/jobs/{jid}", headers={"X-API-Key": key}).json()
        if r["status"] != "processing":
            return r
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_auth_required(client):
    assert client.post("/api/v1/analyze", files={"file": ("a.mov", b"x")}).status_code == 401
    assert client.post("/api/v1/analyze", headers={"X-API-Key": "wrong"}, files={"file": ("a.mov", b"x")}).status_code == 401
    assert client.get("/api/v1/jobs/deadbeef").status_code == 401
    assert client.get("/api/v1/catalog").status_code == 401


def test_full_flow_matches_spec_schema(client):
    r = client.post("/api/v1/analyze", headers={"X-API-Key": "test-key"}, files={"file": ("video_017.mov", b"\x00" * 1000)})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"job_id", "status"} and body["status"] == "processing"
    res = wait_done(client, body["job_id"])
    jsonschema.validate(res, SPEC_SCHEMA)
    assert res["job_id"] == body["job_id"]
    p = client.get(f"/api/v1/jobs/{body['job_id']}/progress", headers={"X-API-Key": "test-key"}).json()
    assert p["status"] == "done" and p["progress"] == 1.0


def test_failed_job_reports_error(client):
    r = client.post("/api/v1/analyze", headers={"X-API-Key": "test-key"}, files={"file": ("bad.mov", b"FAIL....")})
    res = wait_done(client, r.json()["job_id"])
    assert res["status"] == "failed" and "decoder error" in res["error"]


def test_upload_validation(client):
    h = {"X-API-Key": "test-key"}
    assert client.post("/api/v1/analyze", headers=h, files={"file": ("a.txt", b"x")}).status_code == 415
    assert client.post("/api/v1/analyze", headers=h, files={"file": ("a.mov", b"")}).status_code == 400
    assert client.post("/api/v1/analyze", headers=h, files={"file": ("a.mov", b"\x00" * (1024 * 1024 + 1))}).status_code == 413
    assert client.get("/api/v1/jobs/notajob1", headers=h).status_code == 404


def test_catalog(client):
    items = client.get("/api/v1/catalog", headers={"X-API-Key": "test-key"}).json()["items"]
    codes = {i["code"]: i for i in items}
    assert {"V01", "V03", "V17", "X01", "X02"} <= set(codes)
    assert codes["V10"]["status"] == "disabled"


def signup(c, name, pw="secret123"):
    r = c.post("/api/v1/auth/register", json={"username": name, "password": pw})
    assert r.status_code == 200, r.text
    return r


def test_register_login_logout(client):
    assert client.get("/api/v1/auth/me").status_code == 401
    signup(client, "alice")
    assert client.get("/api/v1/auth/me").json() == {"username": "alice"}
    assert client.post("/api/v1/auth/register", json={"username": "ALICE", "password": "secret123"}).status_code == 400
    assert client.post("/api/v1/auth/register", json={"username": "bob", "password": "123"}).status_code == 400
    assert client.post("/api/v1/auth/logout").status_code == 200
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/catalog").status_code == 401
    assert client.post("/api/v1/auth/login", json={"username": "alice", "password": "wrong"}).status_code == 401
    assert client.post("/api/v1/auth/login", json={"username": "alice", "password": "secret123"}).status_code == 200
    assert client.get("/api/v1/catalog").status_code == 200


def test_history_is_per_user_and_video_is_deleted(client):
    signup(client, "alice")
    jid = client.post("/api/v1/analyze", files={"file": ("X-47.mov", b"\x00" * 1000)}).json()["job_id"]
    res = wait_done(client, jid)
    jsonschema.validate(res, SPEC_SCHEMA)
    items = client.get("/api/v1/jobs").json()["items"]
    assert [(i["job_id"], i["filename"], i["status"]) for i in items] == [(jid, "X-47.mov", "done")]
    for _ in range(50):  # summary is written right after the worker finishes
        item = client.get("/api/v1/jobs").json()["items"][0]
        if item["verdict"]:
            break
        time.sleep(0.02)
    assert item["verdict"] == "violation" and item["n_violations"] == 1
    import service.app as app_mod
    d = app_mod.store.dir(jid)
    assert not list(d.glob("input.*")), "video must be deleted after analysis"
    assert (d / "result.json").exists() and "X-47.mov" in (d / "log.txt").read_text()

    client.post("/api/v1/auth/logout")
    signup(client, "bob")
    assert client.get("/api/v1/jobs").json()["items"] == []
    assert client.get(f"/api/v1/jobs/{jid}").status_code == 404
    assert client.get(f"/api/v1/jobs/{jid}", headers={"X-API-Key": "test-key"}).json()["status"] == "done"


def test_archive_prefix():
    from service.archive import prefix
    p = prefix({"job_id": "ab12cd34", "owner": "alice", "filename": "Х-47 копия.mov", "created": 0})
    assert p == "reports/alice/1970-01-01_ab12cd34_Х-47_копия/"
    assert prefix({"job_id": "ab12cd34", "created": 0}).startswith("reports/api/")
