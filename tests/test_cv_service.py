"""CV service (GPU pod) with a stub detector: auth, input validation, limits, no URL fetching."""
import base64
import io
import sys

import numpy as np
import pytest
from PIL import Image

KEY = "k" * 32


class StubDetector:
    device, half = "cpu", False

    def __init__(self, threshold=0.35):
        self.threshold = threshold
        self.seen = []

    def __call__(self, frames):
        from pipeline.perception import Det
        self.seen.append(self.threshold)
        return [[Det("person", 0.9, np.array([1, 2, 30, 40], dtype=np.float32))] for _ in frames]


@pytest.fixture()
def cv(monkeypatch):
    monkeypatch.setenv("CV_API_KEY", KEY)
    monkeypatch.setenv("MAX_IMAGE_MB", "1")
    sys.modules.pop("service.cv_service", None)
    import service.cv_service as m
    monkeypatch.setattr(m, "Detector", StubDetector)
    from fastapi.testclient import TestClient
    with TestClient(m.app) as c:
        c.mod = m
        yield c


def jpeg(w=64, h=64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 120, 120)).save(buf, format="JPEG")
    return buf.getvalue()


H = {"Authorization": f"Bearer {KEY}"}


def test_refuses_to_start_without_key(monkeypatch):
    monkeypatch.delenv("CV_API_KEY", raising=False)
    sys.modules.pop("service.cv_service", None)
    import service.cv_service as m
    monkeypatch.setattr(m, "Detector", StubDetector)
    from fastapi.testclient import TestClient
    with pytest.raises(RuntimeError):
        with TestClient(m.app):
            pass


def test_auth(cv):
    assert cv.post("/detect", files={"file": ("f.jpg", jpeg())}).status_code == 401
    assert cv.post("/detect", files={"file": ("f.jpg", jpeg())}, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert cv.get("/health").json()["ready"] is True
    assert cv.get("/docs").status_code == 404 and cv.get("/openapi.json").status_code == 404


def test_detect_formats(cv):
    r = cv.post("/detect", files={"file": ("f.jpg", jpeg())}, headers=H)
    assert r.status_code == 200 and r.json()["detections"][0]["label"] == "person"
    r = cv.post("/detect", content=jpeg(), headers={**H, "Content-Type": "image/jpeg"})
    assert r.status_code == 200
    r = cv.post("/detect", json={"image_base64": base64.b64encode(jpeg()).decode(), "threshold": 0.5}, headers=H)
    assert r.status_code == 200
    det = cv.mod.detector
    assert det.seen[-1] == 0.5 and det.threshold == 0.35  # per-request threshold never leaks into shared state


def test_no_url_fetching(cv):
    r = cv.post("/detect", json={"url": "http://169.254.169.254/latest/meta-data"}, headers=H)
    assert r.status_code == 400


def test_bad_inputs(cv):
    assert cv.post("/detect", files={"file": ("f.jpg", b"not an image")}, headers=H).status_code == 400
    assert cv.post("/detect", json={"image_base64": "%%%"}, headers=H).status_code == 400
    assert cv.post("/detect", json={"image_base64": "x", "threshold": "high"}, headers=H).status_code == 400
    assert cv.post("/detect", content=b"x" * (3 << 20), headers={**H, "Content-Type": "image/jpeg"}).status_code == 413
    assert cv.post("/detect", content=b"{}", headers={**H, "Content-Type": "text/plain"}).status_code == 415


def test_track_validation(cv):
    ok = {"frames": [{"frame_index": 0, "person_boxes": [[0, 0, 10, 10]]},
                     {"frame_index": 1, "person_boxes": [[1, 1, 11, 11]]}]}
    r = cv.post("/track", json=ok, headers=H)
    assert r.status_code == 200 and r.json()["total_tracks"] == 1
    assert cv.post("/track", json={"frames": [{"frame_index": 0, "person_boxes": [[1, 2]]}]}, headers=H).status_code == 400
    assert cv.post("/track", json={**ok, "max_misses": 10**6}, headers=H).status_code == 422


