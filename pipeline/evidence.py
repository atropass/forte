"""Evidence frames: full-resolution JPEG, thin box + short label on the relevant region, base64."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .frames import frame_at

_FONT = None


def _font(size: int = 14):
    global _FONT
    if _FONT is None:
        for path in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            try:
                _FONT = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _FONT = ImageFont.load_default()
    return _FONT


def annotate(img: np.ndarray, box_rel: list | None = None, label: str | None = None,
             color=(255, 64, 64)) -> Image.Image:
    im = Image.fromarray(img).convert("RGB")
    if box_rel:
        d = ImageDraw.Draw(im)
        x1, y1, x2, y2 = (box_rel[0] * im.width, box_rel[1] * im.height, box_rel[2] * im.width, box_rel[3] * im.height)
        d.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if label:
            f = _font()
            tw = d.textlength(label, font=f)
            ty = max(0, y1 - 18)
            d.rectangle([x1, ty, x1 + tw + 6, ty + 17], fill=color)
            d.text((x1 + 3, ty + 1), label, fill=(255, 255, 255), font=f)
    return im


def to_b64_jpeg(im: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def evidence_frames(video_path, times: list[float], box_rel: list | None = None, label: str | None = None,
                    duration: float | None = None) -> list[dict]:
    out = []
    seen = set()
    for t in times[:3]:
        if duration is not None:
            t = min(max(0.0, t), max(0.0, duration - 0.05))
        t = round(t, 2)
        if t in seen:
            continue
        seen.add(t)
        img = frame_at(video_path, t)
        out.append({"timestamp_sec": t, "image_base64": to_b64_jpeg(annotate(img, box_rel, label))})
    return out
