"""High-Speed Computer Vision Microservice: RT-DETRv2 Object Detection & Tracking."""
from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.perception import Detector, track, COCO_KEEP  # noqa: E402

app = FastAPI(title="ForteVision CV Service", version="1.0.0")

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


def check_auth(authorization: str | None = Header(None)):
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


class Base64DetectRequest(BaseModel):
    image_base64: str
    threshold: float | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ready": detector is not None,
        "device": detector.device if detector else None,
        "classes": list(COCO_KEEP),
    }


@app.post("/cv/detect", response_model=DetectionResponse)
async def detect_image(
    file: UploadFile | None = File(None),
    body: Base64DetectRequest | None = None,
    authorization: str | None = Header(None),
):
    check_auth(authorization)
    if detector is None:
        raise HTTPException(status_code=503, detail="Detector not ready")

    t0 = time.time()
    if file:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    elif body and body.image_base64:
        raw = body.image_base64
        if "," in raw:
            raw = raw.split(",", 1)[1]
        content = base64.b64decode(raw)
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    else:
        raise HTTPException(status_code=400, detail="Must provide 'file' multipart or 'image_base64' JSON body")

    # Guard against 1x1 or sub-minimal dummy images
    w, h = pil_img.size
    if w < 32 or h < 32:
        pil_img = pil_img.resize((max(32, w), max(32, h)), Image.NEAREST)

    frame = np.asarray(pil_img)
    results = detector(frame[None])[0]
    elapsed = (time.time() - t0) * 1000

    bboxes = [
        BBox(
            label=d.label,
            confidence=round(d.score, 4),
            box=[round(float(coord), 2) for coord in d.box.tolist()],
        )
        for d in results
    ]

    return DetectionResponse(elapsed_ms=round(elapsed, 2), detections=bboxes)


class FrameDetections(BaseModel):
    frame_index: int
    person_boxes: list[list[float]]  # list of [x1, y1, x2, y2]


class TrackRequest(BaseModel):
    frames: list[FrameDetections]
    iou_threshold: float = 0.25
    max_misses: int = 4


@app.post("/cv/track")
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
