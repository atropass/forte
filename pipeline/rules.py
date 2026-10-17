"""Deterministic rule engine: timeline events + VLM checks -> violations.

Flow: plan() turns events into VLM check requests (question, frame times, crop), the caller answers
them in one batch, then decide() applies voting thresholds from rules.yaml and builds violations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .timeline import Event

RULES = yaml.safe_load((Path(__file__).parent / "rules.yaml").read_text())["types"]

PERIODIC_STEP = 5.0  # seconds between periodic workplace checks
HANDOVER_WINDOW = (2.0, 14.0)  # seconds after a visitor arrives in which the first handover is searched
HANDOVER_FRAMES = 6


# for cashier_left triggers, check frames are taken relative to the moment the room is empty (the cashier
# is out of the picture), while the timecode stays at the leave start
CHECK_BASE: dict[float, float] = {}


@dataclass
class Check:
    rule: str
    question: str
    trigger_t: float
    times: list[float]
    min_yes: int = 2
    multi: bool = False  # all times go into ONE multi-frame question
    expect: str | None = None
    periodic: bool = False
    answers: list = field(default_factory=list)


def _triggers(rule: dict) -> list[str]:
    t = rule.get("trigger", [])
    return t if isinstance(t, list) else [t]


def expand_triggers(events: list[Event], duration: float) -> dict[str, list[float]]:
    """Map trigger names used in rules.yaml to lists of trigger times (timecodes)."""
    trig: dict[str, list[float]] = {k: [] for k in (
        "cashier_left", "cashier_entered", "visitor_arrived", "second_visitor",
        "visitor_arrived_after_visitor", "visitor_episode", "periodic", "cashier_absent_with_visitor")}
    n_visitors = 0
    for e in events:
        if e.kind == "cashier_absent_with_visitor":
            trig["cashier_absent_with_visitor"].append(e.t)
        elif e.kind == "cashier_left":
            trig["cashier_left"].append(e.t)
            CHECK_BASE[round(e.t, 2)] = e.info.get("t_room_empty", e.t)
        elif e.kind == "cashier_entered":
            trig["cashier_entered"].append(e.t)
        elif e.kind in ("visitor_arrived", "second_visitor"):
            if e.kind == "visitor_arrived":
                trig["visitor_arrived"].append(e.t)
                trig["visitor_episode"].append(e.t)
            else:
                trig["second_visitor"].append(e.t)
            if n_visitors >= 1:
                trig["visitor_arrived_after_visitor"].append(e.t)
            n_visitors += 1
    t = 1.0
    while t < duration - 1.0:
        trig["periodic"].append(t)
        t += PERIODIC_STEP
    return trig


def plan(events: list[Event], duration: float) -> list[Check]:
    CHECK_BASE.clear()
    trig = expand_triggers(events, duration)
    checks: list[Check] = []
    for rid, rule in RULES.items():
        if rule.get("enabled", True) is False:
            continue
        for tname in _triggers(rule):
            for t0 in trig.get(tname, []):
                for c in rule.get("checks", []):
                    q = c["q"]
                    if q == "handover_first_item":
                        a, b = HANDOVER_WINDOW
                        times = [t0 + a + i * (b - a) / (HANDOVER_FRAMES - 1) for i in range(HANDOVER_FRAMES)]
                        times = [min(x, duration - 0.1) for x in times]
                        checks.append(Check(rid, q, t0, times, multi=True, expect=c.get("expect")))
                    elif q == "open_check":
                        checks.append(Check(rid, q, t0, [min(t0 + 3.5, duration - 0.1)], min_yes=1))
                    else:
                        offs = c.get("offsets", [0.0])
                        if tname == "periodic":
                            offs = [0.0]  # periodic checks: one frame each, voting across neighbours
                        base = t0
                        if tname == "cashier_left":
                            base = CHECK_BASE.get(round(t0, 2), t0)
                            offs = [0.5, 1.5, 2.5]
                        times = [min(max(0.0, base + o), duration - 0.1) for o in offs]
                        if tname == "periodic":
                            checks.append(Check(rid, q, t0, times, min_yes=1, periodic=True))
                        else:
                            checks.append(Check(rid, q, t0, times, min_yes=c.get("min_yes", 1)))
                if not rule.get("checks"):
                    checks.append(Check(rid, "", t0, [t0], min_yes=0))
    return checks


@dataclass
class Violation:
    type_id: str
    code: str
    title: str  # catalog type_ru (API "type")
    subtype: str
    timecode_sec: float
    observed: str
    reason: str
    legal_ref: str
    evidence_times: list[float]
    confidence: float
    source: str
    extra: dict = field(default_factory=dict)


def decide(checks: list[Check], periodic_min_consecutive: int = 2, seated_at=None) -> list[Violation]:
    """seated_at(t) -> bool | None: cashier seated at that time (from the scene-state pass)."""
    out: list[Violation] = []
    periodic_hits: dict[str, list[Check]] = {}
    for c in checks:
        rule = RULES[c.rule]
        tc = round(c.trigger_t + rule.get("timecode_offset", 0.0), 1)
        if c.question == "":  # detector-only rule
            out.append(_mk(c, rule, tc, [c.trigger_t, c.trigger_t + 1.0, c.trigger_t + 2.0], 0.6, "detector"))
            continue
        if c.question == "handover_first_item":
            ans = c.answers[0] if c.answers else {}
            if ans.get("first_item") == c.expect:
                idx = ans.get("frame")
                t_hand = c.times[idx - 1] if isinstance(idx, int) and 1 <= idx <= len(c.times) else c.times[len(c.times) // 2]
                v = _mk(c, rule, round(t_hand, 1), [t_hand - 1.0, t_hand, t_hand + 1.0], 0.6, "detector+vlm")
                v.extra["vlm_evidence"] = ans.get("evidence", "")
                out.append(v)
            continue
        if c.question == "open_check":
            ans = c.answers[0] if c.answers else {}
            items = [i for i in ans.get("items", []) if isinstance(i, dict) and i.get("item")]
            if ans.get("confidence") == "high" and items:
                v = _mk(c, rule, tc, c.times, 0.5, "vlm-open-check")
                v.observed = v.observed.replace("{items}", ", ".join(f"{i['item']} ({i.get('where', '')})" for i in items))
                out.append(v)
            continue
        yes = sum(1 for a in c.answers if getattr(a, "answer", None) == "yes")
        if c.periodic:
            gate = RULES[c.rule].get("periodic_requires_seated", False)
            if gate and seated_at is not None and not seated_at(c.trigger_t):
                continue
            if yes:
                periodic_hits.setdefault(c.rule, []).append(c)
            continue
        if yes >= c.min_yes:
            ev_times = [tc] + [t for t in c.times if abs(t - tc) > 0.3][:2]  # the moment itself + confirming frames
            v = _mk(c, rule, tc, ev_times, yes / len(c.times), "detector+vlm")
            v.extra["vlm_evidence"] = next((a.evidence for a in c.answers if a.answer == "yes"), "")
            out.append(v)
    # periodic rules need >= N consecutive positive checks (two independent frames 10 s apart)
    for rid, hits in periodic_hits.items():
        hits.sort(key=lambda c: c.trigger_t)
        run = [hits[0]]
        for c in hits[1:]:
            if abs(c.trigger_t - run[-1].trigger_t - PERIODIC_STEP) < 0.5:
                run.append(c)
            else:
                run = [c]
            if len(run) >= periodic_min_consecutive:
                break
        if len(run) >= periodic_min_consecutive:
            first = run[0]
            v = _mk(first, RULES[rid], round(first.trigger_t, 1), [c.trigger_t for c in run[:3]], 0.6, "vlm-periodic")
            v.extra["vlm_evidence"] = next((a.evidence for a in first.answers if a.answer == "yes"), "")
            out.append(v)
    return dedupe(out)


def _mk(c: Check, rule: dict, tc: float, ev_times: list[float], conf: float, source: str) -> Violation:
    return Violation(c.rule, rule.get("code", ""), rule["type"], rule.get("subtype", ""), tc, rule.get("observed", "").replace("{t}", f"{tc:.1f}"),
                     rule.get("reason", ""), rule.get("legal", ""), [max(0.0, t) for t in ev_times], round(conf, 2), source)


def dedupe(vs: list[Violation], window: float = 6.0) -> list[Violation]:
    vs = sorted(vs, key=lambda v: (v.type_id, v.timecode_sec))
    out: list[Violation] = []
    for v in vs:
        if out and out[-1].type_id == v.type_id and v.timecode_sec - out[-1].timecode_sec < window:
            continue
        out.append(v)
    # the generic open check is only reported if no specific rule fired for the same moment
    specific = [v for v in out if v.type_id != "unsecured_valuables_generic"]
    out = [v for v in out if v.type_id != "unsecured_valuables_generic"
           or not any(abs(s.timecode_sec - v.timecode_sec) < window for s in specific)]
    return sorted(out, key=lambda v: v.timecode_sec)
