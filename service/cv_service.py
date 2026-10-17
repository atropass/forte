"""CV microservice on the GPU pod: RT-DETRv2 object detection + IoU tracking.

Server-to-server only (called by the analysis workers). Auth: `Authorization: Bearer <CV_API_KEY>`; the service
refuses to start without a key. Images are accepted as multipart `file`, raw image body, or JSON base64. The service
never fetches remote URLs. Decoding and inference run in worker threads (the event loop stays responsive for health
checks), and inference is batched dynamically: concurrent requests share one forward pass (CV_MAX_BATCH,
CV_BATCH_WAIT_MS).

Env: CV_API_KEY (required, >= 24 chars), DET_THRESHOLD (0.35), DET_MODEL, MAX_IMAGE_MB (20), MAX_IMAGE_PIXELS (40e6),
CV_MAX_BATCH (32), CV_BATCH_WAIT_MS (5), CV_INFER_WORKERS (2), CV_THRESHOLD_FLOOR (= DET_THRESHOLD).
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import hmac
import io
import logging
import os
import sys
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
MAX_BATCH = int(os.getenv("CV_MAX_BATCH", "32"))
BATCH_WAIT_S = float(os.getenv("CV_BATCH_WAIT_MS", "5")) / 1000
# batches in flight: while one runs on the GPU the next is pre-processed on the CPU
INFER_WORKERS = int(os.getenv("CV_INFER_WORKERS", "2"))
# the model always runs at this score floor; each request then keeps scores > its own threshold. RT-DETR picks its
# top-k boxes before thresholding, so this equals running at the request's threshold (thresholds below the floor
# are raised to it).
THRESHOLD_FLOOR = float(os.getenv("CV_THRESHOLD_FLOOR", os.getenv("DET_THRESHOLD", "0.35")))


class Batcher:
    """Dynamic batching: requests that arrive together run as ONE forward pass on the GPU.

    Each request queues (frame, threshold). Each of `workers` GPU tasks takes the first waiting request, collects
    whatever else arrives within BATCH_WAIT_S (up to MAX_BATCH frames) and runs the batch; with 2+ workers the CPU
    pre-processing of one batch overlaps the GPU pass of another. The detector always runs at the score floor and each
    request keeps detections with score > its own threshold (None = the service default), which equals single-request
    inference because RT-DETR selects its top-k boxes before thresholding. The detector is never mutated at runtime.
    """

    def __init__(self, det, max_batch: int = MAX_BATCH, wait_s: float = BATCH_WAIT_S, workers: int = INFER_WORKERS,
                 floor: float = THRESHOLD_FLOOR):
        self.det, self.max_batch, self.wait_s, self.workers = det, max_batch, wait_s, max(1, workers)
        self.default = det.threshold
        self.floor = min(floor, self.default)
        det.threshold = self.floor
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: list[asyncio.Task] = []
        self.stats = {"requests": 0, "batches": 0, "largest_batch": 0}

    def start(self):
        self.tasks = [asyncio.create_task(self._run()) for _ in range(self.workers)]

    async def stop(self):
        for t in self.tasks:
            t.cancel()
        for t in self.tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t

    async def detect(self, frame: np.ndarray, threshold: float | None):
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((frame, threshold, fut))
        return await fut

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            deadline = loop.time() + self.wait_s
            while len(batch) < self.max_batch:
                timeout = deadline - loop.time()
                if timeout <= 0:
                    try:
                        batch.append(self.queue.get_nowait())
                        continue
                    except asyncio.QueueEmpty:
                        break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), timeout))
                except asyncio.TimeoutError:
                    break
            thrs = [self.default if t is None else max(t, self.floor) for _, t, _ in batch]
            try:
                results = await run_in_threadpool(self.det, [f for f, _, _ in batch], self.max_batch)
            except Exception as e:  # noqa: BLE001 - fail every request of the batch, keep serving
                log.exception("batch inference failed")
                for _, _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)
                continue
            self.stats["requests"] += len(batch)
            self.stats["batches"] += 1
            self.stats["largest_batch"] = max(self.stats["largest_batch"], len(batch))
            for (_, _, fut), dets, thr in zip(batch, results, thrs):
                if not fut.done():
                    fut.set_result([d for d in dets if d.score > thr])


batcher: Batcher | None = None


@asynccontextmanager
async def lifespan(_app):
    global detector, batcher
    _api_key()
    log.info("loading RT-DETRv2")
    detector = Detector(threshold=float(os.getenv("DET_THRESHOLD", "0.35")))
    detector([np.full((640, 640, 3), 128, dtype=np.uint8)] * min(MAX_BATCH, 8))  # warm-up incl. a batched pass
    log.info("RT-DETRv2 ready on %s (half=%s), max batch %d, wait %.0f ms, %d batches in flight", detector.device,
             detector.half, MAX_BATCH, BATCH_WAIT_S * 1000, INFER_WORKERS)
    batcher = Batcher(detector)
    batcher.start()
    yield
    await batcher.stop()


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
    return {"status": "ok", "ready": detector is not None and batcher is not None, "classes": list(COCO_KEEP),
            "batching": {"max_batch": MAX_BATCH, "wait_ms": BATCH_WAIT_S * 1000, "workers": INFER_WORKERS, **(batcher.stats if batcher else {})}}


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
        img = await run_in_threadpool(_decode, await upload.read(MAX_IMAGE_BYTES + 1))
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
            img = await run_in_threadpool(_decode, base64.b64decode(raw, validate=True))
        except (binascii.Error, ValueError):
            raise HTTPException(status_code=400, detail="invalid base64") from None
        threshold, classes = body.threshold, body.classes
    elif ctype.startswith("image/"):
        img = await run_in_threadpool(_decode, await request.body())
    else:
        raise HTTPException(status_code=415, detail="send multipart 'file', an image body, or JSON image_base64")
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=400, detail="threshold must be in [0, 1]")
    w, h = img.size
    if w < 32 or h < 32:
        img = img.resize((max(32, w), max(32, h)), Image.NEAREST)
    return img, threshold, classes


@app.post("/cv/detect", response_model=DetectionResponse)
@app.post("/detect", response_model=DetectionResponse)
async def detect_image(request: Request, authorization: str | None = Header(None)):
    check_auth(authorization)
    if detector is None:
        raise HTTPException(status_code=503, detail="detector not ready")
    t0 = time.time()
    img, threshold, classes = await _parse(request)
    results = await batcher.detect(np.asarray(img), threshold)
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
