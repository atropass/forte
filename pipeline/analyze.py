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
from . import handover
from .frames import extract_frames, frame_at, probe
from .keyclf import KeyClassifier
from .legal import legal_ref
from .perception import Detector, Scene, in_room
from .rules import decide, plan
from .timeline import StateSample, events_from_state
from .vlm import QUESTIONS, VLM, crop_box, image_for, parse_json, to_pil, upscale_min

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


def _dbg_answer(a):
    if isinstance(a, handover.EpisodeResult):
        return {"kind": a.kind, "t": a.t, "conf": a.confidence, "id_seen_at": a.id_seen_at,
                "first_cash": a.first_cash, "first_bag": a.first_bag, "trace": a.trace}
    if getattr(a, "p", None) is not None:
        return round(a.p, 3)
    return getattr(a, "answer", a)


def state_step(duration: float) -> float:
    """Coarse step of the scene-state pass; changes are then refined to REFINE_STEP."""
    if duration <= 40:
        return 1.5
    if duration <= 90:
        return 2.5
    return 3.0


REFINE_STEP = 0.5


def handover_crop(frame: np.ndarray, scene_boxes: dict) -> np.ndarray:
    """Desk plus the space above it where hands meet, native resolution, upscaled so small items get pixels."""
    desk = scene_boxes.get("desk")
    if not desk:
        return frame
    x1, y1, x2, y2 = desk
    zone = [x1, max(0.0, y1 - 0.6 * (y2 - y1)), x2, y2]  # extend upwards: visitor's and cashier's hands
    return upscale_min(crop_box(frame, zone, pad=0.12), min_side=560)


def refine_times(samples: list[StateSample], have: set[float]) -> list[float]:
    """Extra sample times between neighbours whose state differs (who is inside / seated / visitor)."""
    extra = []
    for a, b in zip(samples, samples[1:]):
        if (a.inside, a.seated, a.visitor) != (b.inside, b.seated, b.visitor):
            t = a.t + REFINE_STEP
            while t < b.t - 1e-6:
                if round(t, 2) not in have:
                    extra.append(round(t, 2))
                t += REFINE_STEP
    return extra


class Analyzer:
    def __init__(self, vlm: VLM, detector: Detector, key_clf: KeyClassifier | None = None, aux_vlm: VLM | None = None):
        self.vlm = vlm
        self.det = detector
        self.key_clf = key_clf if key_clf is not None else KeyClassifier.load_default()
        self.aux_vlm = aux_vlm  # optional second model, used only on hand-over frames (VLM_AUX_MODEL)

    # ------------------------------------------------------------------ stages
    def _state_at(self, path: Path, times: list[float]) -> list[StateSample]:
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

    def scene_state(self, path: Path, duration: float) -> list[StateSample]:
        """Adaptive: coarse pass over the whole video, then 0.5-s refinement only where the state changes.
        ~2-3x fewer VLM calls than a dense pass, and event times (timecodes) at 0.5-s resolution."""
        coarse = [round(float(t), 2) for t in np.arange(0.5, max(duration - 0.2, 0.6), state_step(duration))]
        samples = self._state_at(path, coarse)
        extra = refine_times(samples, set(coarse))
        if extra:
            samples = sorted(samples + self._state_at(path, extra), key=lambda s: s.t)
        return samples

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
                                   "answers": [_dbg_answer(a) for a in c.answers]} for c in checks]},
        }

    # ------------------------------------------------------------------ VLM batch
    def _answer(self, path: Path, checks, scene_boxes):
        cache: dict[float, np.ndarray] = {}

        def fr(t):
            k = round(t, 2)
            if k not in cache:
                cache[k] = frame_at(path, k)
            return cache[k]

        single, open_items, episodes = [], [], {}
        for ci, c in enumerate(checks):
            if not c.question:
                continue
            if c.question == "handover_sequence":
                episodes.setdefault((round(c.trigger_t, 2), tuple(c.times)), []).append(ci)
            elif c.question == "open_check":
                open_items.append((ci, fr(c.times[0])))
            elif c.question == "key_in_safe" and self.key_clf is not None and scene_boxes.get("safe"):
                for t in c.times:  # VLM cannot see the 3-px key; the lock-patch classifier answers instead
                    checks[ci].answers.append(self.key_clf.answer(fr(t), scene_boxes["safe"], self.vlm))
            else:
                for t in c.times:
                    single.append((ci, c.question, image_for(c.question, fr(t), scene_boxes)))
        if single:  # decision mode: one forward pass per frame, P(yes)
            answers = self.vlm.prob([(q, [img]) for _, q, img in single])
            for (ci, _, _), a in zip(single, answers):
                checks[ci].answers.append(a)
        if episodes:  # one sequence analysis per visitor episode, shared by V05 and V13
            items, index = [], []
            for key, cis in episodes.items():
                for t in key[1]:
                    items.append((handover.PROMPT, [handover_crop(fr(t), scene_boxes)], handover.OPTIONS))
                    index.append(key)
            probs = self.vlm.choose(items)
            if self.aux_vlm is not None:  # ensemble: average the two models' probabilities frame by frame
                aux = self.aux_vlm.choose(items)
                probs = [{o: (p.get(o, 0) + q.get(o, 0)) / 2 for o in handover.OPTIONS} for p, q in zip(probs, aux)]
            for key, cis in episodes.items():
                ep = [p for k, p in zip(index, probs) if k == key]
                res = handover.decide_episode(list(key[1]), ep)
                for ci in cis:
                    checks[ci].answers.append(res)
        if open_items:
            raws = self.vlm.raw([(QUESTIONS["open_check"]["text"].strip(), [img]) for _, img in open_items], max_tokens=160)
            for (ci, _), r in zip(open_items, raws):
                checks[ci].answers.append(parse_json(r) or {})
