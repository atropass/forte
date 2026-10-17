"""End-to-end analysis of one video -> dict in the exact API schema (plus extra per-violation fields)."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from . import evidence as ev
from .frames import extract_frames, frame_at, probe
from .legal import legal_ref
from .perception import Detector, Scene, in_room
from .rules import RULES, decide, plan
from .timeline import build_timeline
from .vlm import QUESTIONS, VLM, image_for, parse_json, to_pil

log = logging.getLogger("analyze")

PIPE_FPS = 2.0
DET_SIDE = 640


class Analyzer:
    def __init__(self, vlm: VLM, detector: Detector):
        self.vlm = vlm
        self.det = detector

    # ------------------------------------------------------------------ perception
    def perceive(self, path: Path):
        info = probe(path)
        ts, frames = extract_frames(path, fps=PIPE_FPS, max_side=DET_SIDE)
        dets = self.det(frames)
        return info, ts, frames, dets

    def ground(self, path: Path, duration: float) -> dict:
        return self.vlm.ground(frame_at(path, duration * 0.5))

    # ------------------------------------------------------------------ main
    def analyze(self, path: str | Path, job_id: str = "") -> dict:
        path = Path(path)
        t0 = time.time()
        info, ts, frames, dets = self.perceive(path)
        g = self.ground(path, info.duration)
        scene = Scene.from_grounding(g)
        scene_boxes = {k: v for k, v in scene.__dict__.items() if v}
        h, w = frames.shape[1:3]
        inroom = [[d.box for d in fd if d.label == "person" and in_room(d.box, w, h, scene, d.score)] for fd in dets]
        events, dbg = build_timeline(ts, inroom, (w, h), PIPE_FPS, scene.desk)
        t_perc = time.time() - t0

        checks = plan(events, info.duration)
        self._answer(path, checks, scene_boxes)
        violations = decide(checks)
        t_total = time.time() - t0
        log.info("%s: %d events, %d checks, %d violations, perception %.1fs total %.1fs",
                 path.name, len(events), len(checks), len(violations), t_perc, t_total)

        out_v = []
        for v in violations:
            ref = legal_ref(v.legal_ref)
            box_obj = {"key_left_in_safe": "safe", "safe_left_open": "safe", "money_left_unattended": "desk",
                       "bag_left_on_desk": "desk", "laptop_unlocked_unattended": "laptop",
                       "prev_client_money_on_desk": "desk", "door_left_open": "door",
                       "personal_items_on_desk": "desk"}.get(v.type_id)
            frames_b64 = ev.evidence_frames(path, v.evidence_times, scene_boxes.get(box_obj) if box_obj else None,
                                            None, info.duration)
            desc = f"{v.observed} {v.reason}"
            if ref:
                desc += f" Основание: {ref['short']}."
            out_v.append({
                "type": v.title,
                "timecode_sec": float(v.timecode_sec),
                "description": desc.strip(),
                "evidence_frames": frames_b64,
                # extra fields (allowed inside each violation object)
                "type_id": v.type_id,
                "rule": ref,
                "source": v.source,
                "confidence": v.confidence,
            })
        return {
            "job_id": job_id,
            "status": "done",
            "verdict": "violation" if out_v else "no_violation",
            "violations": out_v,
            "_debug": {"events": [e.__dict__ for e in events], "occ": dbg["occ"], "grounding": g,
                       "timing_sec": round(t_total, 1), "n_vlm_checks": len(checks)},
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
