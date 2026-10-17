"""Question bank, prompt building, caching and structured parsing on top of a VLMBackend."""
from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .vlm_backends import VLMBackend, VLMRequest

QUESTIONS = yaml.safe_load((Path(__file__).parent / "questions.yaml").read_text())


@dataclass
class Answer:
    answer: str  # yes | no | unclear
    evidence: str
    raw: str

    @property
    def yes(self) -> bool:
        return self.answer == "yes"


def to_pil(img, max_side: int | None = None) -> Image.Image:
    im = img if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img))
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC)
    return im


def crop_box(img: np.ndarray, box_rel: list[float] | None, pad: float = 0.35, min_side: int = 160) -> np.ndarray:
    """Crop around a 0-1000 relative box with padding; returns full image when box is None."""
    if box_rel is None:
        return img
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [v / 1000 for v in box_rel]
    bw, bh = (x2 - x1) * w, (y2 - y1) * h
    cx, cy = (x1 + x2) / 2 * w, (y1 + y2) / 2 * h
    side_w = max(bw * (1 + 2 * pad), min_side)
    side_h = max(bh * (1 + 2 * pad), min_side)
    xa, xb = int(max(0, cx - side_w / 2)), int(min(w, cx + side_w / 2))
    ya, yb = int(max(0, cy - side_h / 2)), int(min(h, cy + side_h / 2))
    return img[ya:yb, xa:xb]


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json(text: str) -> dict | None:
    m = _JSON_RE.search(text.replace("```json", "").replace("```", ""))
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def parse_answer(text: str) -> Answer:
    j = parse_json(text)
    if j and isinstance(j.get("answer"), str):
        a = j["answer"].strip().lower()
        a = a if a in ("yes", "no", "unclear") else "unclear"
        return Answer(a, str(j.get("evidence", ""))[:200], text)
    low = text.strip().lower()
    for a in ("yes", "no"):
        if low.startswith(a) or f'"{a}"' in low:
            return Answer(a, "", text)
    return Answer("unclear", "", text)


class VLM:
    def __init__(self, backend: VLMBackend, max_side: int = 640, cache_dir: str | Path | None = None):
        self.backend = backend
        self.max_side = max_side
        self.cache: dict[str, str] = {}
        self.cache_file = Path(cache_dir) / f"{backend.model_id.replace('/', '_')}.jsonl" if cache_dir else None
        if self.cache_file and self.cache_file.exists():
            for line in self.cache_file.read_text().splitlines():
                k, v = json.loads(line)
                self.cache[k] = v

    def _key(self, prompt: str, images: list[Image.Image]) -> str:
        h = hashlib.sha1(prompt.encode())
        h.update(self.backend.model_id.encode())
        for im in images:
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            h.update(buf.getvalue())
        return h.hexdigest()

    def raw(self, items: list[tuple[str, list]], max_tokens: int = 64) -> list[str]:
        """items: [(prompt, [images])]. Cached + batched."""
        reqs, keys, todo = [], [], []
        for i, (prompt, imgs) in enumerate(items):
            pil = [to_pil(im, self.max_side) for im in imgs]
            k = self._key(prompt, pil)
            keys.append(k)
            if k not in self.cache:
                todo.append(i)
                reqs.append(VLMRequest(prompt, pil, max_tokens))
        if reqs:
            outs = self.backend.generate(reqs)
            for i, o in zip(todo, outs):
                self.cache[keys[i]] = o
                if self.cache_file:
                    self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with self.cache_file.open("a") as f:
                        f.write(json.dumps([keys[i], o], ensure_ascii=False) + "\n")
        return [self.cache[k] for k in keys]

    def prompt_for(self, qid: str, text_override: str | None = None) -> str:
        q = QUESTIONS["questions"][qid]
        return QUESTIONS["preamble"].strip() + "\nQuestion: " + (text_override or q["text"]).strip()

    def ask(self, batch: list[tuple[str, list]], text_override: dict | None = None) -> list[Answer]:
        """batch: [(question_id, [images])]"""
        items = [(self.prompt_for(q, (text_override or {}).get(q)), imgs) for q, imgs in batch]
        return [parse_answer(t) for t in self.raw(items)]

    def ground(self, frame: np.ndarray) -> dict:
        txt = self.raw([(QUESTIONS["grounding"]["text"].strip(), [frame])], max_tokens=200)[0]
        j = parse_json(txt) or {}
        out = {}
        for k in ("safe", "desk", "door", "laptop", "counting_machine"):
            v = j.get(k)
            if isinstance(v, list) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v):
                out[k] = [float(x) for x in v]
        return out


def upscale_min(img: np.ndarray, min_side: int = 448) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) >= min_side:
        return img
    s = min_side / min(h, w)
    return np.asarray(Image.fromarray(img).resize((round(w * s), round(h * s)), Image.BICUBIC))


def image_for(qid: str, frame: np.ndarray, scene_boxes: dict) -> np.ndarray:
    """Pick the image for a question per its scope (crop around a grounded object, or the full frame)."""
    scope = QUESTIONS["questions"].get(qid, {}).get("scope", "full")
    if scope.startswith("crop:"):
        box = scene_boxes.get(scope.split(":", 1)[1])
        if box:
            return upscale_min(crop_box(frame, box))
    return frame
