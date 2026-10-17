"""Deterministic rule engine: timeline events + VLM checks -> violations.

Flow: plan() turns events into VLM check requests (question, frame times, crop), the caller answers
them in one batch, then decide() applies the voting thresholds and builds violations.

Voting (yes/no checks, decision mode): every frame gives p = P(yes). A check fires when at least
`min_yes` frames have p >= threshold(question) OR the mean p over its frames is >= threshold.
Periodic checks fire after 2 consecutive positive samples. Thresholds: pipeline/thresholds.yaml
(written by scripts/calibrate.py) or the defaults in questions.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import handover
from .timeline import Event
from .vlm import THRESHOLDS, threshold

RULES = yaml.safe_load((Path(__file__).parent / "rules.yaml").read_text())["types"]

PERIODIC_STEP = 5.0  # seconds between periodic workplace checks

# for cashier_left triggers, check frames are taken relative to the moment the room is empty (the cashier
# is out of the picture), while the timecode stays at the leave start
CHECK_BASE: dict[float, float] = {}
EPISODE_END: dict[float, float | None] = {}


@dataclass
class Check:
    rule: str
    question: str
    trigger_t: float
    times: list[float]
    min_yes: int = 2
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
    arrivals = [e for e in events if e.kind in ("visitor_arrived", "second_visitor")]
    for e in events:
        if e.kind == "cashier_absent_with_visitor":
            trig["cashier_absent_with_visitor"].append(e.t)
        elif e.kind == "cashier_left":
            trig["cashier_left"].append(e.t)
            CHECK_BASE[round(e.t, 2)] = e.info.get("t_room_empty", e.t)
        elif e.kind == "cashier_entered":
            trig["cashier_entered"].append(e.t)
        elif e.kind in ("visitor_arrived", "second_visitor"):
            trig["visitor_arrived" if e.kind == "visitor_arrived" else "second_visitor"].append(e.t)
            # every visitor (first or second) is an episode in which money or a bag may be handed over;
            # it ends when somebody leaves or the next visitor arrives
            trig["visitor_episode"].append(e.t)
            ends = [x.t for x in events if x.kind in ("visitor_left", "cashier_left") and x.t > e.t + 1.0]
            ends += [a.t for a in arrivals if a.t > e.t + 1.0]
            EPISODE_END[round(e.t, 2)] = min(ends) if ends else None
            if n_visitors >= 1:
                trig["visitor_arrived_after_visitor"].append(e.t)
            n_visitors += 1
    t = 1.0
    while t < duration - 1.0:
        trig["periodic"].append(t)
        t += PERIODIC_STEP
    return trig


def rule_enabled(rid: str, rule: dict) -> bool:
    """Rules marked requires_calibration run only after scripts/calibrate.py has shown that their check
    separates violation from clean videos on the deployed model (listed in thresholds.yaml: enabled_rules)."""
    if rule.get("enabled", True) is False:
        return False
    if rule.get("requires_calibration"):
        return rid in ((THRESHOLDS or {}).get("enabled_rules") or [])
    return True


def plan(events: list[Event], duration: float, visitor_at_desk: list[float] | None = None) -> list[Check]:
    """visitor_at_desk: scene-state times at which a visitor is at the desk (episode frames start there)."""
    CHECK_BASE.clear()
    EPISODE_END.clear()
    trig = expand_triggers(events, duration)
    checks: list[Check] = []
    for rid, rule in RULES.items():
        if not rule_enabled(rid, rule):
            continue
        for tname in _triggers(rule):
            for t0 in trig.get(tname, []):
                for c in rule.get("checks", []):
                    q = c["q"]
                    if q == "handover_sequence":
                        end = EPISODE_END.get(round(t0, 2))
                        at_desk = [t for t in (visitor_at_desk or []) if t >= t0 and (end is None or t < end)]
                        seated = at_desk[0] if at_desk else t0 + 2.0
                        times = handover.episode_times(t0, end, duration, seated_from=seated)
                        checks.append(Check(rid, q, t0, times, min_yes=0))
                    elif q == "open_check":
                        checks.append(Check(rid, q, t0, [min(CHECK_BASE.get(round(t0, 2), t0) + 1.0, duration - 0.1)],
                                            min_yes=1))
                    else:
                        offs = c.get("offsets", [0.0])
                        base = t0
                        if tname == "periodic":
                            offs = [0.0]  # periodic checks: one frame each, voting across neighbours
                        elif tname == "cashier_left":
                            base = CHECK_BASE.get(round(t0, 2), t0)
                            offs = [0.5, 1.5, 2.5]
                        times = [min(max(0.0, base + o), duration - 0.1) for o in offs]
                        checks.append(Check(rid, q, t0, times, min_yes=1 if tname == "periodic" else c.get("min_yes", 1),
                                            periodic=tname == "periodic"))
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


def _p(a) -> float:
    """P(yes) of an answer; legacy text answers count as 1/0."""
    p = getattr(a, "p", None)
    if p is not None:
        return float(p)
    return 1.0 if getattr(a, "answer", None) == "yes" else 0.0


def votes(c: Check) -> tuple[bool, float]:
    ps = [_p(a) for a in c.answers]
    if not ps:
        return False, 0.0
    thr = threshold(c.question)
    mean = sum(ps) / len(ps)
    return (sum(p >= thr for p in ps) >= max(1, c.min_yes) or mean >= thr), mean


def decide(checks: list[Check], periodic_min_consecutive: int = 2, seated_at=None, alone_at=None) -> list[Violation]:
    """seated_at(t) / alone_at(t) -> bool | None: cashier seated / cashier alone in the room at time t
    (from the scene-state pass)."""
    out: list[Violation] = []
    periodic_hits: dict[str, list[tuple[Check, float]]] = {}
    for c in checks:
        rule = RULES[c.rule]
        tc = round(c.trigger_t + rule.get("timecode_offset", 0.0), 1)
        if c.question == "":  # timeline-only rule
            out.append(_mk(c, rule, tc, [c.trigger_t, c.trigger_t + 1.0, c.trigger_t + 2.0], 0.6, "timeline"))
            continue
        if c.question == "handover_sequence":
            res = c.answers[0] if c.answers else None
            if res is not None and c.rule == "partial_disbursement":
                if res.partial_at is not None:
                    v = _mk(c, rule, round(res.partial_at, 1), [res.give_onsets[0], res.partial_at, res.partial_at + 1.0],
                            0.6, "timeline+vlm-sequence")
                    v.extra.update(give_onsets=res.give_onsets)
                    out.append(v)
                continue
            if res is not None and res.kind == c.rule and res.t is not None:
                v = _mk(c, rule, round(res.t, 1), [res.t, res.t + 0.5, res.t + 1.5], res.confidence, "timeline+vlm-sequence")
                v.extra.update(first_cash=res.first_cash, first_bag=res.first_bag)
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
        if rule.get("requires_cashier_not_seated") and seated_at is not None:
            # two independent signals: the scene-state pass must also see the cashier away from her seat
            if any(seated_at(t) for t in c.times):
                continue
        fired, mean = votes(c)
        if c.periodic:
            if rule.get("periodic_requires_seated") and seated_at is not None and not seated_at(c.trigger_t):
                continue
            if rule.get("periodic_requires_alone") and alone_at is not None and not alone_at(c.trigger_t):
                continue
            if fired:
                periodic_hits.setdefault(c.rule, []).append((c, mean))
            continue
        if fired:
            ev_times = [tc] + [t for t in c.times if abs(t - tc) > 0.3][:2]  # the moment itself + confirming frames
            v = _mk(c, rule, tc, ev_times, mean, "timeline+vlm")
            v.extra["p_yes"] = [round(_p(a), 2) for a in c.answers]
            out.append(v)
    # periodic rules need >= N consecutive positive samples (independent frames PERIODIC_STEP apart)
    for rid, hits in periodic_hits.items():
        hits.sort(key=lambda h: h[0].trigger_t)
        run = [hits[0]]
        for h in hits[1:]:
            if abs(h[0].trigger_t - run[-1][0].trigger_t - PERIODIC_STEP) < 0.5:
                run.append(h)
            else:
                run = [h]
            if len(run) >= periodic_min_consecutive:
                break
        if len(run) >= periodic_min_consecutive:
            first = run[0][0]
            conf = sum(m for _, m in run) / len(run)
            v = _mk(first, RULES[rid], round(first.trigger_t, 1), [h[0].trigger_t for h in run[:3]], conf, "vlm-periodic")
            out.append(v)
    return dedupe(out)


def _mk(c: Check, rule: dict, tc: float, ev_times: list[float], conf: float, source: str) -> Violation:
    return Violation(c.rule, rule.get("code", ""), rule["type"], rule.get("subtype", ""), tc,
                     rule.get("observed", "").replace("{t}", f"{tc:.1f}"), rule.get("reason", ""), rule.get("legal", ""),
                     [max(0.0, t) for t in ev_times], round(float(conf), 2), source)


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
