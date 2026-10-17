"""End-to-end analysis of one video -> dict in the exact API schema (plus extra per-violation fields).

Stages
  1. perception   frames at 2 fps (640 px) -> RT-DETRv2 person boxes (positions / motion only)
  2. grounding    1 VLM call: safe / desk / door / laptop / counting machine boxes (per video, no fixed zones)
  3. scene state  VLM every STATE_STEP s: people INSIDE the room, cashier seated, visitor at desk
  4. events       room empty -> cashier_left (start refined by detector motion), entries, visitors, ...
  5. checks       narrow yes/no VLM questions on crops at trigger moments + periodic checks, one batch
  6. rules        deterministic voting (rules.yaml) -> violations, evidence frames, legal references
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

from . import evidence as ev
from .frames import extract_frames, frame_at, probe
from .keyclf import KeyClassifier
from .legal import legal_ref
from .perception import Detector, Scene, in_room
from .rules import decide, plan
from .timeline import StateSample, events_from_state
from .vlm import QUESTIONS, VLM, image_for, parse_json, to_pil

log = logging.getLogger("analyze")

PIPE_FPS = 2.0
DET_SIDE = 640

# object to outline on evidence frames, per rule
EVIDENCE_OBJECT = {
    "key_left_in_safe": "safe", "safe_left_open": "safe", "left_without_safe_key": "safe",
    "money_left_unattended": "desk", "bag_left_on_desk": "desk", "prev_client_money_on_desk": "desk",
    "personal_items_on_desk": "desk", "unsecured_valuables_generic": "desk",
    "laptop_unlocked_unattended": "laptop", "door_left_open": "door",
}
EVIDENCE_LABEL = {"safe": "сейф", "desk": "стол", "laptop": "ноутбук", "door": "дверь"}

Progress = Callable[[str, float], None]


def state_step(duration: float) -> float:
    if duration <= 40:
        return 1.0
    if duration <= 90:
        return 2.0
    return 2.5


class Analyzer:
    def __init__(self, vlm: VLM, detector: Detector, key_clf: KeyClassifier | None = None):
        self.vlm = vlm
        self.det = detector
        self.key_clf = key_clf if key_clf is not None else KeyClassifier.load_default()

    # ------------------------------------------------------------------ stages
    def scene_state(self, path: Path, duration: float) -> list[StateSample]:
        step = state_step(duration)
        times = list(np.arange(0.5, max(duration - 0.2, 0.6), step))
        frames = [frame_at(path, float(t)) for t in times]
        raws = self.vlm.raw([(QUESTIONS["state"]["text"].strip(), [f]) for f in frames], max_tokens=60)
        out = []
        for t, r in zip(times, raws):
            j = parse_json(r) or {}
            try:
                n = int(j.get("inside", 0))
            except (TypeError, ValueError):
                n = 0
            out.append(StateSample(round(float(t), 2), max(0, min(n, 9)), bool(j.get("cashier_seated")),
                                   bool(j.get("visitor_at_desk"))))
        return out

    def analyze(self, path: str | Path, job_id: str = "", progress: Progress | None = None) -> dict:
        path = Path(path)
        prog = progress or (lambda stage, pct: None)
        t0 = time.time()
        timing = {}

        prog("perception", 0.05)
        info = probe(path)
        ts, frames = extract_frames(path, fps=PIPE_FPS, max_side=DET_SIDE)
        dets = self.det(frames)
        h, w = frames.shape[1:3]
        timing["perception"] = round(time.time() - t0, 1)

        prog("grounding", 0.20)
        g = self.vlm.ground(frame_at(path, info.duration * 0.5))
        scene = Scene.from_grounding(g)
        rel_boxes = {k: v for k, v in scene.__dict__.items() if v}  # 0-1, for drawing evidence boxes
        scene_boxes = dict(g)  # 0-1000, as expected by crop_box / image_for

        prog("scene_state", 0.30)
        t1 = time.time()
        states = self.scene_state(path, info.duration)
        timing["scene_state"] = round(time.time() - t1, 1)
        # detector boxes of plausible in-room people, for leave-start refinement
        det_boxes = [[d.box for d in fd if d.label == "person" and in_room(d.box, w, h, scene, d.score)] for fd in dets]
        events, dbg = events_from_state(states, info.duration, ts, det_boxes, (w, h))

        prog("checks", 0.60)
        t2 = time.time()
        checks = plan(events, info.duration)
        self._answer(path, checks, scene_boxes)
        timing["checks"] = round(time.time() - t2, 1)

        def seated_at(t: float) -> bool | None:
            if not states:
                return None
            s = min(states, key=lambda x: abs(x.t - t))
            return s.seated

        violations = decide(checks, seated_at=seated_at)

        prog("evidence", 0.90)
        out_v = []
        for v in violations:
            ref = legal_ref(v.legal_ref)
            obj = EVIDENCE_OBJECT.get(v.type_id)
            frames_b64 = ev.evidence_frames(path, v.evidence_times, rel_boxes.get(obj) if obj else None,
                                            EVIDENCE_LABEL.get(obj), info.duration)
            desc = f"{v.subtype}. {v.observed} {v.reason}"
            if ref:
                desc += f" Основание: {ref['short']}."
            out_v.append({
                "type": v.title,
                "timecode_sec": float(v.timecode_sec),
                "description": " ".join(desc.split()),
                "evidence_frames": frames_b64,
                # extra fields (allowed inside each violation object)
                "type_code": v.code,
                "subtype": v.subtype,
                "type_id": v.type_id,
                "rule": ref,
                "source": v.source,
                "confidence": v.confidence,
            })
        timing["total"] = round(time.time() - t0, 1)
        log.info("%s: %d events, %d checks, %d violations, timing %s", path.name, len(events), len(checks),
                 len(out_v), timing)
        prog("done", 1.0)
        return {
            "job_id": job_id,
            "status": "done",
            "verdict": "violation" if out_v else "no_violation",
            "violations": out_v,
            "_debug": {"events": [e.__dict__ for e in events], "state": dbg, "grounding": g, "timing_sec": timing,
                       "n_vlm_checks": len(checks), "duration_sec": round(info.duration, 2),
                       "checks": [{"rule": c.rule, "q": c.question, "t": round(c.trigger_t, 2), "times": [round(x, 2) for x in c.times],
                                   "answers": [getattr(a, "answer", a) for a in c.answers]} for c in checks]},
        }

    # ------------------------------------------------------------------ VLM batch
    def _answer(self, path: Path, checks, scene_boxes):
        cache: dict[float, np.ndarray] = {}

        def fr(t):
            k = round(t, 2)
            if k not in cache:
                cache[k] = frame_at(path, k)
            return cache[k]

        single, multi_items, open_items = [], [], []
        for ci, c in enumerate(checks):
            if not c.question:
                continue
            if c.question == "handover_first_item":
                imgs = [to_pil(image_for("bag_handover", fr(t), scene_boxes), 512) for t in c.times]
                multi_items.append((ci, imgs))
            elif c.question == "open_check":
                open_items.append((ci, fr(c.times[0])))
            elif c.question == "key_in_safe" and self.key_clf is not None and scene_boxes.get("safe"):
                for t in c.times:  # VLM cannot see the 3-px key; the lock-patch classifier answers instead
                    checks[ci].answers.append(self.key_clf.answer(fr(t), scene_boxes["safe"], self.vlm))
            else:
                for t in c.times:
                    single.append((ci, c.question, image_for(c.question, fr(t), scene_boxes)))
        if single:
            answers = self.vlm.ask([(q, [img]) for _, q, img in single])
            for (ci, _, _), a in zip(single, answers):
                checks[ci].answers.append(a)
        if multi_items:
            prompt = QUESTIONS["handover_first_item"]["text"].strip() + \
                " Also give \"frame\": <1-based index of the frame where it is handed over>."
            raws = self.vlm.raw([(prompt, imgs) for _, imgs in multi_items], max_tokens=80)
            for (ci, _), r in zip(multi_items, raws):
                checks[ci].answers.append(parse_json(r) or {})
        if open_items:
            raws = self.vlm.raw([(QUESTIONS["open_check"]["text"].strip(), [img]) for _, img in open_items], max_tokens=160)
            for (ci, _), r in zip(open_items, raws):
                checks[ci].answers.append(parse_json(r) or {})
