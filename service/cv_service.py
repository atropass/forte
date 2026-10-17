"""CV microservice on the GPU pod: RT-DETRv2 object detection + IoU tracking.

Server-to-server only (called by the analysis workers). Auth: `Authorization: Bearer <CV_API_KEY>`; the service
refuses to start without a key. Images are accepted as multipart `file`, raw image body, or JSON base64. The service
never fetches remote URLs. Inference runs in a worker thread (the event loop stays responsive for health checks)
and is serialized on the single GPU.

Env: CV_API_KEY (required, >= 24 chars), DET_THRESHOLD (0.35), DET_MODEL, MAX_IMAGE_MB (20), MAX_IMAGE_PIXELS (40e6).
"""
from __future__ import annotations

import base64
import binascii
import hmac
import io
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.perception import COCO_KEEP, Detector, track  # noqa: E402

log = logging.getLogger("cv_service")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_IMAGE_BYTES = int(float(os.getenv("MAX_IMAGE_MB", "20")) * 1024 * 1024)
Image.MAX_IMAGE_PIXELS = int(float(os.getenv("MAX_IMAGE_PIXELS", "40000000")))
MAX_TRACK_FRAMES = 20000
MAX_BOXES_PER_FRAME = 200


def _api_key() -> str:
    key = os.getenv("CV_API_KEY", "")
    if len(key) < 24:
        raise RuntimeError("CV_API_KEY must be set (at least 24 characters); refusing to start without auth")
    return key


detector: Detector | None = None
_gpu_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app):
    global detector
    _api_key()
    log.info("loading RT-DETRv2")
    detector = Detector(threshold=float(os.getenv("DET_THRESHOLD", "0.35")))
    detector(np.full((640, 640, 3), 128, dtype=np.uint8)[None])  # warm-up
    log.info("RT-DETRv2 ready on %s (half=%s)", detector.device, detector.half)
    yield


app = FastAPI(title="ForteVision CV Service", version="2.0.0", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url=None)


def check_auth(authorization: str | None):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="unauthorized")
    token = authorization[7:].strip()
    if not hmac.compare_digest(token.encode(), _api_key().encode()):
        raise HTTPException(status_code=401, detail="unauthorized")


class BBox(BaseModel):
    label: str
    confidence: float
    box: list[float]  # [x1, y1, x2, y2] in pixels


class DetectionResponse(BaseModel):
    elapsed_ms: float
    detections: list[BBox]


class DetectJSON(BaseModel):
    image_base64: str | None = Field(default=None, max_length=MAX_IMAGE_BYTES * 4 // 3 + 64)
    image: str | None = Field(default=None, max_length=MAX_IMAGE_BYTES * 4 // 3 + 64)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    classes: list[str] | None = Field(default=None, max_length=50)


class FrameDetections(BaseModel):
    frame_index: int = Field(ge=0)
    person_boxes: list[list[float]] = Field(max_length=MAX_BOXES_PER_FRAME)


class TrackRequest(BaseModel):
    frames: list[FrameDetections] = Field(max_length=MAX_TRACK_FRAMES)
    iou_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    max_misses: int = Field(default=4, ge=0, le=100)


@app.get("/")
def root():
    return {"service": "ForteVision CV Service", "version": "2.0.0", "model": "RT-DETRv2-r18vd",
            "ready": detector is not None}


@app.get("/health")
@app.get("/cv/health")
def health():
    return {"status": "ok", "ready": detector is not None, "classes": list(COCO_KEEP)}


def _decode(content: bytes) -> Image.Image:
    if not content:
        raise HTTPException(status_code=400, detail="empty image")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image too large")
    try:
        img = Image.open(io.BytesIO(content))
        img.load()
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        raise HTTPException(status_code=400, detail="not a valid image") from None


async def _parse(request: Request) -> tuple[Image.Image, float | None, list[str] | None]:
    ctype = request.headers.get("content-type", "").lower()
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_IMAGE_BYTES * 2:
        raise HTTPException(status_code=413, detail="request too large")
    threshold, classes = None, None
    if "multipart/form-data" in ctype:
        form = await request.form(max_files=1, max_fields=10)
        upload = form.get("file") or form.get("image")
        if not hasattr(upload, "read"):
            raise HTTPException(status_code=400, detail="multipart field 'file' is required")
        img = _decode(await upload.read(MAX_IMAGE_BYTES + 1))
        if form.get("threshold"):
            try:
                threshold = float(form.get("threshold"))
            except ValueError:
                raise HTTPException(status_code=400, detail="threshold must be a number") from None
        if form.get("classes"):
            classes = [c.strip() for c in str(form.get("classes")).split(",") if c.strip()][:50]
    elif "application/json" in ctype:
        try:
            body = DetectJSON.model_validate_json(await request.body())
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid JSON body") from None
        raw = body.image_base64 or body.image
        if not raw:
            raise HTTPException(status_code=400, detail="image_base64 is required")
        if "," in raw[:100]:
            raw = raw.split(",", 1)[1]
        try:
            img = _decode(base64.b64decode(raw, validate=True))
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="invalid base64") from None
        threshold, classes = body.threshold, body.classes
    elif ctype.startswith("image/"):
        img = _decode(await request.body())
    else:
        raise HTTPException(status_code=415, detail="send multipart 'file', an image body, or JSON image_base64")
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="threshold must be in [0, 1]")
    w, h = img.size
    if w < 32 or h < 32:
        img = img.resize((max(32, w), max(32, h)), Image.NEAREST)
    return img, threshold, classes


def _infer(frame: np.ndarray, threshold: float | None):
    with _gpu_lock:  # one GPU: serialize; the per-request threshold is applied only while holding the lock
        base = detector.threshold
        try:
            if threshold is not None:
                detector.threshold = threshold
            return detector(frame[None])[0]
        finally:
            detector.threshold = base


@app.post("/cv/detect", response_model=DetectionResponse)
@app.post("/detect", response_model=DetectionResponse)
async def detect_image(request: Request, authorization: str | None = Header(None)):
    check_auth(authorization)
    if detector is None:
        raise HTTPException(status_code=503, detail="detector not ready")
    t0 = time.time()
    img, threshold, classes = await _parse(request)
    results = await run_in_threadpool(_infer, np.asarray(img), threshold)
    if classes:
        allowed = set(classes)
        results = [d for d in results if d.label in allowed]
    return DetectionResponse(
        elapsed_ms=round((time.time() - t0) * 1000, 2),
        detections=[BBox(label=d.label, confidence=round(d.score, 4), box=[round(float(c), 2) for c in d.box.tolist()])
                    for d in results])


@app.post("/cv/track")
@app.post("/track")
def track_people(req: TrackRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    frames = sorted(req.frames, key=lambda f: f.frame_index)
    for f in frames:
        if any(len(b) != 4 for b in f.person_boxes):
            raise HTTPException(status_code=400, detail="each box must be [x1, y1, x2, y2]")
    t0 = time.time()
    tracks = track([[np.array(b, dtype=np.float32) for b in f.person_boxes] for f in frames],
                   iou_thr=req.iou_threshold, max_misses=req.max_misses)
    out = [{"track_id": t.tid,
            "first_frame": min(t.boxes) if t.boxes else -1,
            "last_frame": max(t.boxes) if t.boxes else -1,
            "trajectory": {i: [round(float(c), 2) for c in box.tolist()] for i, box in t.boxes.items()}}
           for t in tracks]
    return {"elapsed_ms": round((time.time() - t0) * 1000, 2), "total_tracks": len(out), "tracks": out}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8005")))
