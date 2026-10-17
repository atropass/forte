"""Key-in-safe-lock classifier.

The lock on the safe door is ~3 px wide at 464x848, too small for the VLM (Phase 2: 0/12 even on 10x
patches), but a key/tag leaves a short bar next to the lock that is visible in the raw pixels.
Pipeline: VLM grounds the lock inside an upscaled safe crop (1 call per video, cached) -> fixed-size
native-resolution patch around the lock -> HOG + intensity features -> logistic regression.
Trained by scripts/train_keyclf.py with leave-one-video-out validation; weights in models/keyclf.joblib.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .vlm import Answer, crop_box, parse_json, to_pil

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "keyclf.joblib"
PATCH = 32  # output patch side (pixels) after resampling
LOCATE_PROMPT = ('This is a close-up of a grey metal safe. Locate the lock / keyhole on the safe door. '
                 'Return JSON only: {"lock": [x1, y1, x2, y2]} in 0-1000 coordinates relative to this image.')


def safe_crop(frame: np.ndarray, safe_box: list) -> np.ndarray:
    return crop_box(frame, safe_box, pad=0.05, min_side=8)


def locate_lock(vlm, safe_img: np.ndarray) -> list | None:
    big = Image.fromarray(safe_img).resize((safe_img.shape[1] * 3, safe_img.shape[0] * 3), Image.LANCZOS)
    j = parse_json(vlm.raw([(LOCATE_PROMPT, [to_pil(np.asarray(big), 896)])], max_tokens=60)[0]) or {}
    lock = j.get("lock")
    if isinstance(lock, list) and len(lock) == 4 and all(isinstance(v, (int, float)) for v in lock):
        return [float(v) for v in lock]
    return None


def lock_patch(safe_img: np.ndarray, lock_rel: list | None) -> np.ndarray:
    """Square patch centred on the lock, side = 18% of the safe-crop width (covers lock + protruding key)."""
    hh, ww = safe_img.shape[:2]
    if lock_rel:
        cx, cy = (lock_rel[0] + lock_rel[2]) / 2000 * ww, (lock_rel[1] + lock_rel[3]) / 2000 * hh
    else:  # fallback: left-middle of the door, where the lock is in all training setups
        cx, cy = 0.15 * ww, 0.5 * hh
    side = max(8, int(round(0.18 * ww)))
    x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    p = safe_img[y1:y1 + side, x1:x1 + side]
    if p.size == 0:
        p = safe_img
    return np.asarray(Image.fromarray(p).convert("L").resize((PATCH, PATCH), Image.BICUBIC), dtype=np.float32) / 255.0


def features(patch: np.ndarray) -> np.ndarray:
    from skimage.feature import hog
    p = (patch - patch.mean()) / (patch.std() + 1e-4)
    h = hog(p, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    small = np.asarray(Image.fromarray(((p - p.min()) / (np.ptp(p) + 1e-6) * 255).astype(np.uint8))
                       .resize((8, 8), Image.BILINEAR), dtype=np.float32).ravel() / 255.0
    return np.concatenate([h, small])


class KeyClassifier:
    def __init__(self, model, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold
        self._lock_cache: dict[tuple, list | None] = {}

    @classmethod
    def load_default(cls) -> "KeyClassifier | None":
        if not MODEL_PATH.exists():
            return None
        import joblib
        d = joblib.load(MODEL_PATH)
        return cls(d["model"], d.get("threshold", 0.5))

    def prob(self, frame: np.ndarray, safe_box: list, vlm) -> float:
        s = safe_crop(frame, safe_box)
        key = tuple(round(v) for v in safe_box)
        if key not in self._lock_cache:
            self._lock_cache[key] = locate_lock(vlm, s)
        return float(self.model.predict_proba(features(lock_patch(s, self._lock_cache[key]))[None])[0, 1])

    def answer(self, frame: np.ndarray, safe_box: list, vlm) -> Answer:
        p = self.prob(frame, safe_box, vlm)
        a = "yes" if p >= self.threshold else "no"
        return Answer(a, f"классификатор замка: p(ключ)={p:.2f}", f"keyclf p={p:.3f}")
