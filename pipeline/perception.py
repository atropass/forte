"""Person detection (RT-DETRv2, Apache-2.0), lightweight IoU tracking, in-room filtering, roles.

No face recognition or identification: people are only boxes; roles come from position/behaviour.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

COCO_KEEP = {"person", "laptop", "handbag", "backpack", "suitcase", "cup", "bottle", "cell phone", "book"}


@dataclass
class Det:
    label: str
    score: float
    box: np.ndarray  # x1,y1,x2,y2 in pixels of the analysed frame


@dataclass
class Track:
    tid: int
    boxes: dict = field(default_factory=dict)  # frame_idx -> box
    last: int = -1
    misses: int = 0

    def box_at(self, i):
        return self.boxes.get(i)


class Detector:
    def __init__(self, model_id: str | None = None, device: str | None = None, threshold: float = 0.4):
        import torch
        from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection
        self.torch = torch
        self.model_id = model_id or os.getenv("DET_MODEL", "PekingU/rtdetr_v2_r18vd")
        self.device = device or os.getenv("DEVICE") or (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(self.model_id).to(self.device).eval()
        self.half = self.device == "cuda"
        if self.half:
            self.model = self.model.half()
        self.id2label = self.model.config.id2label
        self.threshold = threshold

    def __call__(self, frames: np.ndarray, batch: int = 16) -> list[list[Det]]:
        out_all = []
        with self.torch.no_grad():
            for i in range(0, len(frames), batch):
                chunk = list(frames[i:i + batch])
                inp = self.proc(images=chunk, return_tensors="pt").to(self.device)
                if self.half:
                    inp["pixel_values"] = inp["pixel_values"].half()
                out = self.model(**inp)
                res = self.proc.post_process_object_detection(
                    out, target_sizes=[f.shape[:2] for f in chunk], threshold=self.threshold)
                for r in res:
                    dets = []
                    for s, l, b in zip(r["scores"], r["labels"], r["boxes"]):
                        lab = self.id2label[int(l)]
                        if lab in COCO_KEEP:
                            dets.append(Det(lab, float(s), b.float().cpu().numpy()))
                    out_all.append(dets)
        return out_all


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def track(person_boxes: list[list[np.ndarray]], iou_thr: float = 0.25, max_misses: int = 4) -> list[Track]:
    """Greedy IoU tracker (ByteTrack-style association without the low-score pass)."""
    tracks: list[Track] = []
    active: list[Track] = []
    nid = 0
    for i, boxes in enumerate(person_boxes):
        unmatched = list(range(len(boxes)))
        pairs = sorted(((iou(t.boxes[t.last], boxes[j]), ti, j) for ti, t in enumerate(active) for j in unmatched),
                       reverse=True)
        used_t, used_d = set(), set()
        for s, ti, j in pairs:
            if s < iou_thr or ti in used_t or j in used_d:
                continue
            t = active[ti]
            t.boxes[i] = boxes[j]
            t.last, t.misses = i, 0
            used_t.add(ti)
            used_d.add(j)
        for ti, t in enumerate(active):
            if ti not in used_t:
                t.misses += 1
        active = [t for t in active if t.misses <= max_misses]
        for j in range(len(boxes)):
            if j not in used_d:
                t = Track(nid, {i: boxes[j]}, i)
                nid += 1
                tracks.append(t)
                active.append(t)
    return tracks


@dataclass
class Scene:
    """Per-video scene layout in 0-1 relative coordinates (from VLM grounding, with fallbacks)."""
    safe: list | None = None
    desk: list | None = None
    door: list | None = None
    laptop: list | None = None
    counting_machine: list | None = None

    @classmethod
    def from_grounding(cls, g: dict) -> "Scene":
        rel = {k: [v / 1000 for v in b] for k, b in g.items()}
        return cls(**{k: rel.get(k) for k in ("safe", "desk", "door", "laptop", "counting_machine")})

    def floor_line(self) -> float:
        """y (0-1) below which a person's lowest visible point must be to count as inside the room.

        People seen through the glass wall stand behind it, so their boxes end above the wall base,
        which is approximately the bottom of the grounded door box. Seated people inside the room are
        cut by the desk at about the same height, hence the 0.06 margin. Fallback: 0.44.
        """
        if self.door:
            return min(max(self.door[3] - 0.06, 0.30), 0.60)
        return 0.44


MIN_PERSON_SCORE = 0.5
MIN_PERSON_H = 0.05   # relative height; smaller boxes are reflections / false positives
TALL_PERSON_H = 0.15  # a box this tall is a person near the camera, i.e. inside the room


def in_room(box: np.ndarray, w: int, h: int, scene: Scene, score: float = 1.0) -> bool:
    bh = (box[3] - box[1]) / h
    if score < MIN_PERSON_SCORE or bh < MIN_PERSON_H:
        return False
    return box[3] / h >= scene.floor_line() or bh >= TALL_PERSON_H


class RemoteDetector:
    """Our RT-DETRv2 CV service on a GPU pod (`ForteVision CV Service`, POST /detect, multipart `file`).
    Same output as Detector: per frame a list of Det with boxes in pixels of the sent frame.
    Env: DET_API_BASE (https://<pod>-8000.proxy.runpod.net), DET_API_KEY (default VLM_API_KEY),
    DET_CONCURRENCY (parallel requests, default 8), DET_JPEG_QUALITY (90)."""

    def __init__(self, base_url: str | None = None, api_key: str | None = None, threshold: float = 0.4):
        import httpx
        self.base = (base_url or os.getenv("DET_API_BASE", "")).rstrip("/")
        if not self.base:
            raise RuntimeError("DET_API_BASE is not set")
        key = api_key if api_key is not None else os.getenv("DET_API_KEY") or os.getenv("VLM_API_KEY", "")
        self.concurrency = int(os.getenv("DET_CONCURRENCY", "8"))
        self.quality = int(os.getenv("DET_JPEG_QUALITY", "90"))
        self.threshold = threshold
        self.model_id = f"remote:{self.base}"
        self.device = "remote"
        self.client = httpx.Client(timeout=float(os.getenv("DET_TIMEOUT", "60")),
                                   headers={"Authorization": f"Bearer {key}"} if key else {},
                                   limits=httpx.Limits(max_connections=self.concurrency))

    def _one(self, frame: np.ndarray) -> list[Det]:
        import io
        import time as _t

        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(np.asarray(frame)).save(buf, format="JPEG", quality=self.quality)
        delay = 1.0
        for attempt in range(5):
            try:
                r = self.client.post(f"{self.base}/detect", files={"file": ("frame.jpg", buf.getvalue(), "image/jpeg")})
                if r.status_code in (429, 500, 502, 503, 504):
                    raise RuntimeError(f"HTTP {r.status_code}")
                r.raise_for_status()
                out = []
                for d in r.json().get("detections", []):
                    if d.get("label") in COCO_KEEP and float(d.get("confidence", 0)) >= self.threshold:
                        out.append(Det(d["label"], float(d["confidence"]), np.array(d["box"], dtype=np.float32)))
                return out
            except Exception:  # noqa: BLE001 - transient gateway errors are retried
                if attempt == 4:
                    raise
                _t.sleep(delay)
                delay *= 2
        return []

    def __call__(self, frames, batch: int = 16) -> list[list[Det]]:
        from concurrent.futures import ThreadPoolExecutor
        frames = list(frames)
        if len(frames) <= 1:
            return [self._one(f) for f in frames]
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(self._one, frames))


def detector_backend(vlm_backend: str | None = None) -> str:
    """DET_BACKEND = rtdetr (local GPU/CPU) | remote (CV service, DET_API_BASE) | auto (default):
    remote when DET_API_BASE is set, otherwise local rtdetr."""
    choice = os.getenv("DET_BACKEND", "auto").lower()
    if choice == "auto":
        choice = "remote" if os.getenv("DET_API_BASE") else "rtdetr"
    return choice


def make_detector(vlm_backend: str | None = None):
    kind = detector_backend(vlm_backend)
    if kind == "remote":
        return RemoteDetector()
    return Detector()
