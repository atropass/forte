"""High-Speed Computer Vision Microservice: RT-DETRv2 Object Detection & Tracking."""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.perception import Detector, track, COCO_KEEP  # noqa: E402

app = FastAPI(
    title="ForteVision CV Service",
    version="1.0.0",
    description="Real-time Computer Vision Microservice with RT-DETRv2 (Object Detection) & IoU Tracking on NVIDIA GPU.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.getenv("CV_API_KEY", "fv-prod-secret-forte-2026")
detector: Detector | None = None


@app.on_event("startup")
def load_models():
    global detector
    print("==> Loading RT-DETRv2-r18vd into CUDA...")
    detector = Detector(threshold=float(os.getenv("DET_THRESHOLD", "0.35")))
    # Warmup with dummy frame
    dummy = np.full((640, 640, 3), 128, dtype=np.uint8)
    detector(dummy[None])
    print(f"==> RT-DETRv2 loaded on device: {detector.device} (half={detector.half})")


def check_auth(authorization: str | None):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Bearer token")
    token = authorization.split("Bearer ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


class BBox(BaseModel):
    label: str
    confidence: float
    box: list[float]  # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    elapsed_ms: float
    detections: list[BBox]


class FrameDetections(BaseModel):
    frame_index: int
    person_boxes: list[list[float]]  # list of [x1, y1, x2, y2]


class TrackRequest(BaseModel):
    frames: list[FrameDetections]
    iou_threshold: float = 0.25
    max_misses: int = 4


@app.get("/")
def root():
    return {
        "service": "ForteVision Computer Vision Microservice",
        "version": "1.0.0",
        "status": "online",
        "model": "RT-DETRv2-r18vd",
        "device": detector.device if detector else None,
        "supported_classes": list(COCO_KEEP),
        "endpoints": {
            "health": ["GET /health", "GET /cv/health"],
            "detection": ["POST /cv/detect", "POST /detect"],
            "tracking": ["POST /cv/track", "POST /track"],
            "interactive_docs": "/docs",
            "openapi_spec": "/openapi.json",
        },
    }


@app.get("/health")
@app.get("/cv/health")
def health():
    return {
        "status": "ok",
        "ready": detector is not None,
        "device": detector.device if detector else None,
        "classes": list(COCO_KEEP),
    }


async def _parse_image_from_request(request: Request) -> tuple[Image.Image, float | None, list[str] | None]:
    """Parse image from either multipart/form-data, application/json, or raw binary."""
    content_type = request.headers.get("content-type", "").lower()
    pil_img: Image.Image | None = None
    req_threshold: float | None = None
    target_classes: list[str] | None = None

    if "multipart/form-data" in content_type:
        form = await request.form()
        # Look for file under 'file', 'image', or any UploadFile
        for key in ("file", "image"):
            field = form.get(key)
            if hasattr(field, "read"):
                content = await field.read()
                pil_img = Image.open(io.BytesIO(content)).convert("RGB")
                break
        if pil_img is None:
            for val in form.values():
                if hasattr(val, "read"):
                    content = await val.read()
                    pil_img = Image.open(io.BytesIO(content)).convert("RGB")
                    break

        if form.get("threshold"):
            try:
                req_threshold = float(form.get("threshold"))
            except ValueError:
                pass
        if form.get("classes"):
            target_classes = [c.strip() for c in str(form.get("classes")).split(",") if c.strip()]

    elif "application/json" in content_type:
        try:
            data = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON payload: {e}")

        req_threshold = data.get("threshold")
        target_classes = data.get("classes")

        raw_b64 = data.get("image_base64") or data.get("image")
        image_url = data.get("url")

        if raw_b64:
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                content = base64.b64decode(raw_b64)
                pil_img = Image.open(io.BytesIO(content)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {e}")
        elif image_url:
            try:
                resp = requests.get(image_url, timeout=10)
                resp.raise_for_status()
                pil_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to fetch image from URL: {e}")
    elif any(img_type in content_type for img_type in ("image/jpeg", "image/png", "image/webp", "image/bmp")):
        body_bytes = await request.body()
        if body_bytes:
            pil_img = Image.open(io.BytesIO(body_bytes)).convert("RGB")

    if pil_img is None:
        raise HTTPException(
            status_code=400,
            detail="No valid image found. Provide multipart file ('file' or 'image'), JSON {'image_base64': '...'} or {'url': '...'}, or raw image bytes.",
        )

    # Guard against 1x1 or sub-minimal dummy images
    w, h = pil_img.size
    if w < 32 or h < 32:
        pil_img = pil_img.resize((max(32, w), max(32, h)), Image.NEAREST)

    return pil_img, req_threshold, target_classes


@app.post("/cv/detect", response_model=DetectionResponse)
@app.post("/detect", response_model=DetectionResponse)
async def detect_image(
    request: Request,
    authorization: str | None = Header(None),
):
    check_auth(authorization)
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not ready")

    t0 = time.time()
    pil_img, req_threshold, target_classes = await _parse_image_from_request(request)

    frame = np.asarray(pil_img)

    # Run RT-DETRv2 inference
    orig_threshold = detector.threshold
    if req_threshold is not None and 0.0 <= req_threshold <= 1.0:
        detector.threshold = req_threshold

    try:
        results = detector(frame[None])[0]
    finally:
        detector.threshold = orig_threshold

    elapsed = (time.time() - t0) * 1000

    # Filter by classes if specified
    if target_classes:
        allowed = set(target_classes)
        results = [d for d in results if d.label in allowed]

    bboxes = [
        BBox(
            label=d.label,
            confidence=round(d.score, 4),
            box=[round(float(coord), 2) for coord in d.box.tolist()],
        )
        for d in results
    ]

    return DetectionResponse(elapsed_ms=round(elapsed, 2), detections=bboxes)


@app.post("/cv/track")
@app.post("/track")
def track_people(req: TrackRequest, authorization: str | None = Header(None)):
    check_auth(authorization)
    sorted_frames = sorted(req.frames, key=lambda f: f.frame_index)
    person_boxes = [
        [np.array(b, dtype=np.float32) for b in f.person_boxes]
        for f in sorted_frames
    ]

    t0 = time.time()
    tracks = track(person_boxes, iou_thr=req.iou_threshold, max_misses=req.max_misses)
    elapsed = (time.time() - t0) * 1000

    out_tracks = []
    for t in tracks:
        out_tracks.append({
            "track_id": t.tid,
            "first_frame": min(t.boxes.keys()) if t.boxes else -1,
            "last_frame": max(t.boxes.keys()) if t.boxes else -1,
            "trajectory": {
                frame_idx: [round(float(c), 2) for c in box.tolist()]
                for frame_idx, box in t.boxes.items()
            },
        })

    return {
        "elapsed_ms": round(elapsed, 2),
        "total_tracks": len(out_tracks),
        "tracks": out_tracks,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8005")))
