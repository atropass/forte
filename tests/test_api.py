"""API contract + security tests with a stub analyzer (no models, no Redis server, no GPU).

python -m pytest tests/test_api.py -q
"""
import jsonschema

from conftest import MOV

K = {"X-API-Key": "test-key"}

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


def upload(c, name="video_017.mov", data=MOV, headers=K):
    return c.post("/api/v1/analyze", headers=headers, files={"file": (name, data)})


def test_auth_required(client):
    assert client.post("/api/v1/analyze", files={"file": ("a.mov", MOV)}).status_code == 401
    assert upload(client, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/api/v1/jobs/deadbeef").status_code == 401
    assert client.get("/api/v1/catalog").status_code == 401
    assert client.post("/api/v1/auth/register", json={"username": "x", "password": "y"}).status_code in (404, 405)


def test_full_flow_matches_spec_schema(client):
    r = upload(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"job_id", "status"} and body["status"] == "processing"
    assert client.get(f"/api/v1/jobs/{body['job_id']}", headers=K).json() == {"job_id": body["job_id"], "status": "processing"}
    client.env.work()
    res = client.get(f"/api/v1/jobs/{body['job_id']}", headers=K).json()
    jsonschema.validate(res, SPEC_SCHEMA)
    assert res["job_id"] == body["job_id"]
    p = client.get(f"/api/v1/jobs/{body['job_id']}/progress", headers=K).json()
    assert p["status"] == "done" and p["progress"] == 1.0 and p["queue_position"] == 0
    d = client.get(f"/api/v1/jobs/{body['job_id']}/details", headers=K).json()
    assert d["debug"]["timing_sec"]["total"] == 0.1 and "video_key" not in d["job"]


def test_video_deleted_and_log_kept(client):
    jid = upload(client).json()["job_id"]
    client.env.work()
    jobs_dir = client.env.data_dir / "blobs" / "jobs" / jid
    assert not list(jobs_dir.glob("input.*")), "video must be deleted after analysis"
    assert (jobs_dir / "result.json").exists() and f"job {jid} done" in (jobs_dir / "log.txt").read_text()


def test_failed_job_reports_generic_error(client):
    jid = upload(client, "bad.mov", MOV + b"FAIL").json()["job_id"]
    client.env.work()
    res = client.get(f"/api/v1/jobs/{jid}", headers=K).json()
    assert res == {"job_id": jid, "status": "failed", "error": "internal_error"}
    assert not list((client.env.data_dir / "blobs" / "jobs" / jid).glob("input.*"))


def test_upload_validation(client):
    assert upload(client, "a.txt", b"x").status_code == 415
    assert upload(client, "a.mov", b"").status_code == 400
    assert upload(client, "a.mov", b"<html>not a video</html>").status_code == 415
    assert upload(client, "a.mov", MOV + b"\x00" * (1024 * 1024 + 1)).status_code == 413
    assert client.get("/api/v1/jobs/notajob1", headers=K).status_code == 404
    assert client.get("/api/v1/jobs/..%2F..%2Fetc", headers=K).status_code == 404


def test_ffprobe_validation(client, monkeypatch):
    from service import app as app_mod
    from service.video import InvalidVideo, VideoInfo
    client.env.set(VALIDATE_VIDEO="1")

    def bad(_p):
        raise InvalidVideo("not a readable video")
    monkeypatch.setattr(app_mod, "probe", bad)
    r = upload(client)
    assert r.status_code == 422 and r.json()["detail"] == "not a readable video"
    monkeypatch.setattr(app_mod, "probe", lambda _p: VideoInfo(5000, 848, 464, "h264"))
    assert upload(client).status_code == 422
    monkeypatch.setattr(app_mod, "probe", lambda _p: VideoInfo(30, 848, 464, "h264"))
    assert upload(client).status_code == 200


def test_idempotent_reupload(client):
    a = upload(client).json()["job_id"]
    b = upload(client).json()["job_id"]
    assert a == b
    c = upload(client, data=MOV + b"other").json()["job_id"]
    assert c != a


def test_catalog(client):
    items = client.get("/api/v1/catalog", headers=K).json()["items"]
    codes = {i["code"]: i for i in items}
    assert {"V01", "V03", "V17", "X01", "X02"} <= set(codes)
    assert codes["V10"]["status"] == "disabled"


def make_user(name="alice", pw="correct-horse-battery"):
    from service import auth
    auth.create_user(name, pw)
    return pw


def login(c, name="alice", pw="correct-horse-battery"):
    return c.post("/api/v1/auth/login", json={"username": name, "password": pw})


def test_login_logout_and_session_revocation(client):
    make_user()
    assert client.get("/api/v1/auth/me").status_code == 401
    assert login(client, pw="wrong-password-123").status_code == 401
    r = login(client)
    assert r.status_code == 200 and "httponly" in r.headers["set-cookie"].lower()
    assert "samesite=strict" in r.headers["set-cookie"].lower()
    token = client.cookies.get("fv_session")
    assert client.get("/api/v1/auth/me").json() == {"username": "alice"}
    assert client.post("/api/v1/auth/logout").status_code == 200
    assert client.get("/api/v1/auth/me").status_code == 401
    client.cookies.set("fv_session", token)  # replaying the old cookie must not work
    assert client.get("/api/v1/auth/me").status_code == 401


def test_password_reset_revokes_sessions(client):
    from service import auth
    make_user()
    login(client)
    auth.set_password("alice", "another-long-password")
    assert client.get("/api/v1/auth/me").status_code == 401


def test_login_lockout(client):
    make_user()
    for _ in range(5):
        assert login(client, pw="wrong-password-123").status_code == 401
    assert login(client).status_code == 429


def test_csrf_origin_checks(client):
    make_user()
    assert client.post("/api/v1/auth/login", json={"username": "alice", "password": "correct-horse-battery"},
                       headers={"Origin": "https://evil.example"}).status_code == 403
    login(client)
    r = client.post("/api/v1/analyze", files={"file": ("a.mov", MOV)}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert client.post("/api/v1/analyze", files={"file": ("a.mov", MOV)}).status_code == 200


def test_history_and_isolation(client):
    make_user("alice")
    make_user("bob")
    login(client, "alice")
    jid = client.post("/api/v1/analyze", files={"file": ("X-47.mov", MOV)}).json()["job_id"]
    client.env.work()
    items = client.get("/api/v1/jobs").json()["items"]
    assert [(i["job_id"], i["filename"], i["status"], i["verdict"], i["n_violations"]) for i in items] == \
        [(jid, "X-47.mov", "done", "violation", 1)]
    client.post("/api/v1/auth/logout")
    login(client, "bob")
    assert client.get("/api/v1/jobs").json()["items"] == []
    for path in ("", "/progress", "/details"):
        assert client.get(f"/api/v1/jobs/{jid}{path}").status_code == 404
    client.post("/api/v1/auth/logout")
    assert client.get(f"/api/v1/jobs/{jid}", headers=K).status_code == 404  # API keys see API jobs only
    assert client.get("/api/v1/jobs", headers=K).status_code == 401  # history needs a session


def test_ui_active_job_cap(svc_env):
    svc_env.set(UI_MAX_ACTIVE_JOBS="1")
    from fastapi.testclient import TestClient

    from service.app import create_app
    make_user()
    with TestClient(create_app(), headers={"Origin": "http://testserver"}) as c:
        login(c)
        assert c.post("/api/v1/analyze", files={"file": ("a.mov", MOV)}).status_code == 200
        assert c.post("/api/v1/analyze", files={"file": ("b.mov", MOV + b"b")}).status_code == 429


def test_managed_api_keys(client):
    from service import auth
    key = auth.create_api_key("jury")
    assert upload(client, headers={"X-API-Key": key}).status_code == 200
    assert auth.revoke_api_key("jury") == 1
    assert upload(client, headers={"X-API-Key": key}).status_code == 401


def test_security_headers_and_ops(client):
    r = client.get("/healthz")
    assert r.json() == {"ok": True}
    for h in ("x-content-type-options", "x-frame-options", "content-security-policy", "referrer-policy", "x-request-id"):
        assert h in r.headers
    assert r.headers["cache-control"] == "no-store"
    assert client.get("/api/docs").status_code == 404
    assert client.get("/api/openapi.json").status_code == 404
    assert client.get("/metrics").status_code == 404
    rz = client.get("/readyz")
    assert rz.status_code == 200 and rz.json()["checks"] == {"redis": True, "storage": True}
    assert client.get("/readyz?strict=1").status_code == 503  # no workers registered


def test_metrics_with_token(client):
    client.env.set(METRICS_TOKEN="m-token")
    from fastapi.testclient import TestClient

    from service.app import create_app
    with TestClient(create_app()) as c:
        assert c.get("/metrics").status_code == 401
        r = c.get("/metrics", headers={"Authorization": "Bearer m-token"})
        assert r.status_code == 200 and "fv_queue_depth" in r.text


def test_api_key_never_logged(client, caplog):
    import logging
    caplog.set_level(logging.INFO)
    upload(client)
    client.env.work()
    assert "test-key" not in caplog.text
