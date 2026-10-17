"""Visitor-episode analysis for V05 (client did not show ID) and V13 (CIT bag accepted without documents).

A single frame cannot separate an ID card from a small stack of banknotes at 464x848 (~10 px each).
The *sequence of actions* is visible though, so every frame of the episode (2 fps, desk crop at native
resolution) gets one multiple-choice decision with probabilities:

  A  visitor hands over / holds out a small card or ID document
  B  visitor shows a phone screen to the cashier (digital ID — valid under п.19 пп.1 of Rules No. 231)
  C  banknotes are handed over, put on the desk or put into the counting machine
  D  a bag or sack is handed over
  E  cashier holds a small card/document and looks at it (examining an ID)
  F  none of these

Decision (decide_episode):
  * first_cash = onset of C (high, confirmed within the next 2 frames) -> money handed over (V05 timecode)
  * first_bag  = onset of D, same rule                                  -> bag handed over (V13 timecode)
  * id_seen    = A+B+E high on >= 2 frames before (or at) the first handover
  * bag first and no id_seen  -> collector_no_id (V13)
  * cash first and no id_seen -> client_no_id (V05)
Evidence of ID anywhere before the handover suppresses the violation (false alarms on clean client
episodes are worse than they look: 6 clean training videos contain client episodes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .vlm import THRESHOLDS

OPTIONS = ["A", "B", "C", "D", "E", "F"]
PROMPT = (
    "Ceiling CCTV frame of a bank cashier's desk: a visitor on one side, the cashier on the other. "
    "Which ONE statement best describes what is happening in this frame?\n"
    "A) the visitor hands over or holds out a small card or ID document\n"
    "B) the visitor shows a phone screen to the cashier\n"
    "C) banknotes are handed over, put on the desk, or put into the counting machine\n"
    "D) a bag or sack is handed over\n"
    "E) the cashier holds a small card or document and looks at it\n"
    "F) none of these\n"
    "Answer with exactly one letter."
)

STEP = 0.5          # seconds between frames
MAX_WINDOW = 25.0   # analyse at most this long after the visitor arrives
ID_THR = float((THRESHOLDS or {}).get("handover_id", 0.40))  # P(A+B+E) on >= ID_MIN_FRAMES frames before the hand-over
ID_MIN_FRAMES = 2
ACT_THR = float((THRESHOLDS or {}).get("handover_act", 0.50))  # P(C)/P(D) confirmed within 2 frames -> onset


@dataclass
class EpisodeResult:
    kind: str | None            # "client_no_id" | "collector_no_id" | None
    t: float | None             # timecode of the handover
    confidence: float           # 1 - strongest ID evidence before the handover
    id_seen_at: float | None = None
    first_cash: float | None = None
    first_bag: float | None = None
    trace: list = field(default_factory=list)


def episode_times(t0: float, t_end: float | None, duration: float) -> list[float]:
    end = min(t_end if t_end is not None else duration, t0 + MAX_WINDOW, duration - 0.1)
    n = max(0, int((end - t0) / STEP) + 1)
    return [round(t0 + i * STEP, 2) for i in range(n)]


def _first_confirmed(xs: list[float], thr: float) -> int | None:
    """Onset of an action: first frame >= thr that is confirmed by one of the next two frames
    (a single-frame flicker is ignored, the timecode stays at the real start)."""
    for i in range(len(xs)):
        if xs[i] >= thr and any(x >= thr for x in xs[i + 1:i + 3]):
            return i
    return None


def decide_episode(times: list[float], probs: list[dict[str, float]], id_thr: float | None = None,
                   act_thr: float | None = None) -> EpisodeResult:
    id_thr = ID_THR if id_thr is None else id_thr
    act_thr = ACT_THR if act_thr is None else act_thr
    if not times:
        return EpisodeResult(None, None, 0.0)
    idp = [p.get("A", 0) + p.get("B", 0) + p.get("E", 0) for p in probs]
    cash = [p.get("C", 0) for p in probs]
    bag = [p.get("D", 0) for p in probs]
    ic, ib = _first_confirmed(cash, act_thr), _first_confirmed(bag, act_thr)
    first_cash = times[ic] if ic is not None else None
    first_bag = times[ib] if ib is not None else None
    trace = [(t, round(a, 2), round(c, 2), round(b, 2)) for t, a, c, b in zip(times, idp, cash, bag)]
    if ic is None and ib is None:
        return EpisodeResult(None, None, 0.0, trace=trace)
    bag_first = ib is not None and (ic is None or ib <= ic)
    i_hand = ib if bag_first else ic
    assert i_hand is not None
    before = idp[: i_hand + 1]  # ID shown at the same moment as the handover still counts
    strong = [i for i, v in enumerate(before) if v >= id_thr]
    id_seen = len(strong) >= ID_MIN_FRAMES
    conf = round(1.0 - (max(before) if before else 0.0), 2)
    res = EpisodeResult(None, times[i_hand], conf, times[strong[0]] if strong else None, first_cash, first_bag, trace)
    if not id_seen:
        res.kind = "collector_no_id" if bag_first else "client_no_id"
    return res
