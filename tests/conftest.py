"""Shared fixtures: in-memory Redis (fakeredis), storage in tmp, stub analyzer, inline RQ worker. No models."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MOV = b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00qt  " + b"\x00" * 1000

FAKE = {"job_id": "", "status": "done", "verdict": "violation", "violations": [{
    "type": "Нарушение порядка хранения наличных денег", "timecode_sec": 10.0, "description": "test",
    "evidence_frames": [{"timestamp_sec": 10.0, "image_base64": "/9j/4AAQSkZJRg=="}], "type_code": "V03"}],
    "_debug": {"timing_sec": {"total": 0.1}}}


class StubAnalyzer:
    def __init__(self):
        self.calls = 0
        self.behaviour = []  # list of callables run per call (index = call number), e.g. raise

    def analyze(self, path, job_id="", progress=None):
        self.calls += 1
        progress("checks", 0.6)
        if self.behaviour and len(self.behaviour) >= self.calls and self.behaviour[self.calls - 1]:
            self.behaviour[self.calls - 1]()
        if b"FAIL" in Path(path).read_bytes():
            raise RuntimeError("decoder error")
        return json.loads(json.dumps(FAKE))


@pytest.fixture()
def svc_env(tmp_path, monkeypatch):
    """Environment + fresh fake Redis + stub analyzer; returns a namespace with helpers."""
    for k in ("VLM_API_BASE", "DET_API_BASE", "API_KEYS", "METRICS_TOKEN", "PUBLIC_ORIGIN", "CORS_ORIGINS"):
        monkeypatch.delenv(k, raising=False)
    env = {"API_KEY": "test-key", "DATA_DIR": str(tmp_path / "data"), "MAX_UPLOAD_MB": "1",
           "FRONTEND_DIR": str(tmp_path / "no-ui"), "COOKIE_SECURE": "0", "VALIDATE_VIDEO": "0",
           "WORKER_MODE": "simple", "REAPER_IN_API": "0", "APP_ENV": "test", "STORAGE_BACKEND": "fs",
           "INFRA_WAIT_S": "0", "LOG_LEVEL": "INFO"}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import fakeredis

    from service import blob, config, redis_conn, worker
    config.reset_settings()
    blob.reset_blob()
    server = fakeredis.FakeServer()
    redis_conn.set_clients(fakeredis.FakeRedis(server=server, decode_responses=True), fakeredis.FakeRedis(server=server))
    stub = StubAnalyzer()
    worker.set_analyzer(stub)

    class NS:
        analyzer = stub
        data_dir = tmp_path / "data"

        @staticmethod
        def set(**kw):
            for k, v in kw.items():
                monkeypatch.setenv(k, str(v))
            config.reset_settings()

        @staticmethod
        def work():
            """Run queued jobs inline until both queues are empty."""
            from rq import SimpleWorker

            from service import jobqueue
            SimpleWorker([jobqueue.queue(q) for q in jobqueue.QUEUES],
                         connection=redis_conn.get_rq_redis()).work(burst=True)

    yield NS
    worker.set_analyzer(None)
    redis_conn.set_clients(None, None)
    config.reset_settings()
    blob.reset_blob()


@pytest.fixture()
def client(svc_env):
    from fastapi.testclient import TestClient

    from service.app import create_app
    with TestClient(create_app(), headers={"Origin": "http://testserver"}) as c:
        c.env = svc_env
        yield c
