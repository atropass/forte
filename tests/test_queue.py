"""Job lifecycle: timeouts, retries, reaper recovery, storage backends."""
import time

import pytest

from conftest import MOV

K = {"X-API-Key": "test-key"}


def upload(c, data=MOV):
    return c.post("/api/v1/analyze", headers=K, files={"file": ("v.mov", data)}).json()["job_id"]


def test_infra_error_is_retried_once(client):
    import httpx

    def boom():
        raise httpx.ConnectError("pod down")
    client.env.analyzer.behaviour = [boom]
    jid = upload(client)
    client.env.work()
    from service import state
    job = state.get(jid)
    assert job["status"] == "done" and job["attempts"] == 2


def test_infra_error_fails_after_max_attempts(client):
    import httpx

    def boom():
        raise httpx.ConnectError("pod down")
    client.env.analyzer.behaviour = [boom, boom, boom]
    jid = upload(client)
    client.env.work()
    res = client.get(f"/api/v1/jobs/{jid}", headers=K).json()
    assert res == {"job_id": jid, "status": "failed", "error": "inference_unavailable"}


def test_logic_error_not_retried(client):
    jid = upload(client, MOV + b"FAIL")
    client.env.work()
    from service import state
    assert state.get(jid)["attempts"] == 1 and state.get(jid)["error"] == "internal_error"


def test_watchdog_timeout(client):
    client.env.set(JOB_TIMEOUT_S="1")
    client.env.analyzer.behaviour = [lambda: time.sleep(2.5)]
    jid = upload(client)
    client.env.work()
    res = client.get(f"/api/v1/jobs/{jid}", headers=K).json()
    assert res == {"job_id": jid, "status": "failed", "error": "timeout"}
    assert not list((client.env.data_dir / "blobs" / "jobs" / jid).glob("input.*"))


def test_upstream_down_fails_fast(client, monkeypatch):
    from service import upstream
    monkeypatch.setattr(upstream, "targets", lambda: {"vlm": ("http://127.0.0.1:9/health", "")})
    jid = upload(client)
    client.env.work()
    assert client.get(f"/api/v1/jobs/{jid}", headers=K).json()["error"] == "inference_unavailable"


def test_reaper_requeues_lost_job_then_fails(client):
    from service import reaper, state
    jid = upload(client)
    # simulate: a worker picked the job and died (no heartbeat, started long ago)
    state.update(jid, stage="grounding", started=time.time() - 60, attempts=1)
    assert reaper.reconcile() == 1
    job = state.get(jid)
    assert job["stage"] == "queued" and job["status"] == "processing"
    state.update(jid, stage="grounding", started=time.time() - 60, attempts=2)
    state.get_redis().delete("fv:lock:reaper")
    assert reaper.reconcile() == 1
    assert state.get(jid)["status"] == "failed" and state.get(jid)["error"] == "worker_lost"
    assert not list((client.env.data_dir / "blobs" / "jobs" / jid).glob("input.*"))


def test_reaper_requeues_job_with_lost_queue_entry(client):
    from service import jobqueue, reaper, state
    jid = upload(client)
    from service.redis_conn import get_rq_redis
    get_rq_redis().delete(jobqueue.queue("api").key)
    get_rq_redis().delete(f"rq:job:{state.get(jid)['rq_id']}")
    state.update(jid, queued_at=time.time() - 120)
    assert reaper.reconcile() == 1
    client.env.work()
    assert state.get(jid)["status"] == "done"


def test_jury_queue_has_priority(client):
    from service import auth, jobqueue
    auth.create_user("alice", "correct-horse-battery")
    client.post("/api/v1/auth/login", json={"username": "alice", "password": "correct-horse-battery"})
    ui_job = client.post("/api/v1/analyze", files={"file": ("a.mov", MOV + b"ui")}).json()["job_id"]
    client.cookies.clear()
    api_job = upload(client, MOV + b"api")
    assert jobqueue.position(api_job) == 1 and jobqueue.position(ui_job) == 2


def test_s3_blob_roundtrip(tmp_path, monkeypatch):
    moto = pytest.importorskip("moto")
    with moto.mock_aws():
        monkeypatch.setenv("S3_BUCKET", "fv-test")
        monkeypatch.setenv("S3_REGION", "us-east-1")
        monkeypatch.setenv("S3_ACCESS_KEY", "x")
        monkeypatch.setenv("S3_SECRET_KEY", "y")
        monkeypatch.delenv("S3_ENDPOINT", raising=False)
        monkeypatch.delenv("AWS_ENDPOINT_URL_S3", raising=False)
        from service import blob, config
        config.reset_settings()
        b = blob.S3Blob(config.get_settings())
        b.ensure_bucket()
        b.check()
        src = tmp_path / "v.mov"
        src.write_bytes(MOV)
        b.put_file(src, blob.job_key("abcd1234", "input.mov"))
        b.get_file(blob.job_key("abcd1234", "input.mov"), tmp_path / "back.mov")
        assert (tmp_path / "back.mov").read_bytes() == MOV
        b.put_json(blob.job_key("abcd1234", "result.json"), {"a": 1})
        assert b.get_json(blob.job_key("abcd1234", "result.json")) == {"a": 1}
        assert b.get_json(blob.job_key("abcd1234", "missing.json")) is None
        assert b.delete_older_than("jobs/", -1) == 2
        config.reset_settings()


def test_storage_keys_are_validated():
    from service import blob
    with pytest.raises(ValueError):
        blob.job_key("../etc", "passwd")
    with pytest.raises(ValueError):
        blob.job_key("abc", "../../x")
