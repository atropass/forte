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
        self.calls = []

    def __call__(self, frames, batch=16):
        from pipeline.perception import Det
        self.seen.append(self.threshold)
        self.calls.append(len(frames))
        if getattr(self, "fail", False):
            raise RuntimeError("cuda error")
        scores = getattr(self, "scores", [0.9])
        return [[Det("person", s, np.array([1, 2, 30, 40], dtype=np.float32)) for s in scores if s > self.threshold]
                for _ in frames]


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
    b64 = base64.b64encode(jpeg()).decode()
    r = cv.post("/detect", json={"image_base64": b64, "threshold": 0.5}, headers=H)
    assert r.status_code == 200 and len(r.json()["detections"]) == 1       # stub score 0.9 > 0.5
    r = cv.post("/detect", json={"image_base64": b64, "threshold": 0.95}, headers=H)
    assert r.status_code == 200 and r.json()["detections"] == []          # filtered per request
    assert cv.get("/health").json()["batching"]["requests"] >= 4


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




def test_batcher_groups_requests_and_keeps_per_request_thresholds(monkeypatch):
    import asyncio
    monkeypatch.setenv("CV_API_KEY", KEY)
    import service.cv_service as m
    det = StubDetector(threshold=0.35)
    det.scores = [0.9, 0.4, 0.32]
    frame = np.zeros((64, 64, 3), np.uint8)

    async def run():
        b = m.Batcher(det, max_batch=8, wait_s=0.05, workers=1, floor=0.1)
        b.start()
        res = await asyncio.gather(b.detect(frame, 0.5), b.detect(frame, 0.3), b.detect(frame, None))
        await b.stop()
        return res, b.stats
    (r50, r30, rdef), stats = asyncio.run(run())
    assert det.calls == [3] and det.seen == [0.1]          # one GPU pass for all three, at the score floor
    assert det.threshold == 0.1                             # fixed at the floor, never swapped per request
    assert [d.score for d in r50] == [0.9]
    assert [d.score for d in r30] == [0.9, 0.4, 0.32]
    assert [d.score for d in rdef] == [0.9, 0.4]            # None = the service default (0.35)
    assert stats == {"requests": 3, "batches": 1, "largest_batch": 3}


def test_batcher_failure_fails_only_that_batch(monkeypatch):
    import asyncio
    monkeypatch.setenv("CV_API_KEY", KEY)
    import service.cv_service as m
    det = StubDetector()
    frame = np.zeros((64, 64, 3), np.uint8)

    async def run():
        b = m.Batcher(det, max_batch=4, wait_s=0.01, workers=2)
        b.start()
        det.fail = True
        with pytest.raises(RuntimeError):
            await b.detect(frame, None)
        det.fail = False
        ok = await b.detect(frame, None)   # the GPU task survived the failure
        await b.stop()
        return ok
    assert len(asyncio.run(run())) == 1


def test_batch_size_cap(monkeypatch):
    import asyncio
    monkeypatch.setenv("CV_API_KEY", KEY)
    import service.cv_service as m
    det = StubDetector()
    frame = np.zeros((64, 64, 3), np.uint8)

    async def run():
        b = m.Batcher(det, max_batch=4, wait_s=0.05, workers=2)
        b.start()
        await asyncio.gather(*[b.detect(frame, None) for _ in range(10)])
        await b.stop()
    asyncio.run(run())
    assert sum(det.calls) == 10 and max(det.calls) <= 4


def test_threshold_below_default_floor_is_raised(monkeypatch):
    import asyncio
    monkeypatch.setenv("CV_API_KEY", KEY)
    import service.cv_service as m
    det = StubDetector(threshold=0.35)
    det.scores = [0.9, 0.4, 0.32]
    frame = np.zeros((64, 64, 3), np.uint8)

    async def run():
        b = m.Batcher(det, max_batch=8, wait_s=0.01, workers=1)   # floor defaults to the service threshold
        b.start()
        r = await b.detect(frame, 0.2)
        await b.stop()
        return r
    assert [d.score for d in asyncio.run(run())] == [0.9, 0.4] and det.seen == [0.35]
