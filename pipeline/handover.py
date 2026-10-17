"""Visitor-episode analysis: V05 (client did not show ID), V13 (CIT bag accepted without documents) and
X03 (cash handed to the client in parts instead of the whole prepared amount at once).

A single frame cannot separate an ID card from a small stack of banknotes at 464x848 (~10 px each).
The *sequence of actions* is visible though, so every frame of the episode (2 fps, desk-zone crop at
native resolution) gets one multiple-choice decision with probabilities:

  A  visitor hands over / holds out a small card or ID document
  B  visitor shows a phone screen to the cashier (digital ID — valid under п.19 пп.1 of Rules No. 231)
  C  the visitor hands banknotes to the cashier / puts them on the desk / into the counting machine
  D  a bag or sack is handed over
  E  cashier holds a small card/document and looks at it (examining an ID)
  F  the cashier hands banknotes to the visitor (disbursement)
  G  none of these

Decision (decide_episode):
  * first_cash = onset of C or F (high, confirmed within the next 2 frames) -> money changes hands
    (п.19 пп.1 for acceptance and п.31 пп.1 for disbursement both require the ID check first)
  * first_bag  = onset of D, same rule -> bag handed over (V13 timecode)
  * id_seen    = A+B+E high on >= 2 frames anywhere in the episode (see note in decide_episode)
  * bag first and no id_seen  -> collector_no_id (V13)
  * cash first and no id_seen -> client_no_id (V05)
  * F has >= 2 separate onsets (a gap of >= GAP_S without F between them) -> partial_at = 2nd onset (X03)
Evidence of ID anywhere in the episode suppresses V05 (false alarms on clean client episodes are
costly: 6 clean training videos contain client episodes).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .vlm import THRESHOLDS

OPTIONS = ["A", "B", "C", "D", "E", "F", "G"]
PROMPT = (
    "Ceiling CCTV frame of a bank cashier's desk: a visitor on one side, the cashier on the other. "
    "Which ONE statement best describes what is happening in this frame?\n"
    "A) the visitor hands over or holds out a small card or ID document\n"
    "B) the visitor shows a phone screen to the cashier\n"
    "C) right now a hand is passing banknotes from the visitor to the cashier (money lying still on the desk does not count)\n"
    "D) a bag or sack is handed over\n"
    "E) the cashier holds a small card or document and looks at it\n"
    "F) right now the cashier's hand is passing banknotes to the visitor\n"
    "G) nothing is being handed over right now (people sit, talk, write, count or use the machine)\n"
    "Answer with exactly one letter."
)

STEP = 0.5          # seconds between frames
MAX_WINDOW = 25.0   # analyse at most this long after the visitor arrives
GAP_S = 1.0         # two disbursement onsets must be separated by at least this long without F
ID_MIN_FRAMES = 2


def _thr(name: str, default: float) -> float:
    return float((THRESHOLDS or {}).get(name, default))


ID_THR = _thr("handover_id", 0.40)    # P(A+B+E) on >= ID_MIN_FRAMES frames before the hand-over
ACT_THR = _thr("handover_act", 0.50)  # P(C)/P(D)/P(F) confirmed within 2 frames -> onset


@dataclass
class EpisodeResult:
    kind: str | None            # "client_no_id" | "collector_no_id" | None
    t: float | None             # timecode of the first hand-over
    confidence: float           # 1 - strongest ID evidence before the hand-over
    id_seen_at: float | None = None
    first_cash: float | None = None
    first_bag: float | None = None
    partial_at: float | None = None      # 2nd separate cashier->visitor cash hand-over (X03)
    give_onsets: list = field(default_factory=list)
    trace: list = field(default_factory=list)


def episode_times(t0: float, t_end: float | None, duration: float, seated_from: float | None = None) -> list[float]:
    """Frames of a visitor episode. Starts when the visitor is at the desk (scene state), not at the door:
    while walking in nothing can be handed over, and the model mislabels those frames."""
    end = min(t_end if t_end is not None else duration, t0 + MAX_WINDOW + 3.0, duration - 0.1)
    if seated_from is not None and t0 <= seated_from < end:
        t0 = seated_from
    n = max(0, int((end - t0) / STEP) + 1)
    return [round(t0 + i * STEP, 2) for i in range(n)]


def _first_confirmed(xs: list[float], thr: float) -> int | None:
    """Onset of an action: first frame >= thr that is confirmed by one of the next two frames
    (a single-frame flicker is ignored, the timecode stays at the real start)."""
    for i in range(len(xs)):
        if xs[i] >= thr and any(x >= thr for x in xs[i + 1:i + 3]):
            return i
    return None


def _onsets(xs: list[float], thr: float, gap_frames: int) -> list[int]:
    """All confirmed onsets; a new onset needs >= gap_frames frames below thr since the previous one
    (a one-frame dip inside a single hand-over does not split it in two)."""
    out, below, in_run = [], gap_frames, False
    for i, x in enumerate(xs):
        if x >= thr:
            if not in_run and below >= gap_frames and any(v >= thr for v in xs[i + 1:i + 3]):
                out.append(i)
            if out:
                in_run, below = True, 0
        else:
            in_run = False
            below += 1
    return out


def decide_episode(times: list[float], probs: list[dict[str, float]], id_thr: float | None = None,
                   act_thr: float | None = None) -> EpisodeResult:
    id_thr = ID_THR if id_thr is None else id_thr
    act_thr = ACT_THR if act_thr is None else act_thr
    if not times:
        return EpisodeResult(None, None, 0.0)
    idp = [p.get("A", 0) + p.get("B", 0) + p.get("E", 0) for p in probs]
    take = [p.get("C", 0) for p in probs]
    give = [p.get("F", 0) for p in probs]
    cash = [max(a, b) for a, b in zip(take, give)]
    bag = [p.get("D", 0) for p in probs]
    ic, ib = _first_confirmed(cash, act_thr), _first_confirmed(bag, act_thr)
    first_cash = times[ic] if ic is not None else None
    first_bag = times[ib] if ib is not None else None
    gives = _onsets(give, act_thr, max(1, int(round(GAP_S / STEP))))
    partial_at = times[gives[1]] if len(gives) >= 2 else None
    trace = [(t, round(a, 2), round(c, 2), round(g, 2), round(b, 2)) for t, a, c, g, b in zip(times, idp, take, give, bag)]
    base = dict(first_cash=first_cash, first_bag=first_bag, partial_at=partial_at,
                give_onsets=[times[i] for i in gives], trace=trace)
    if ic is None and ib is None:
        return EpisodeResult(None, None, 0.0, **base)
    bag_first = ib is not None and (ic is None or ib <= ic)
    i_hand = ib if bag_first else ic
    assert i_hand is not None
    # The model cannot order hand-overs to the second (it labels a visitor sitting down as "cash" and keeps
    # "ID" for many frames), so ID evidence ANYWHERE in the episode clears it: a false alarm on a clean client
    # is worse than missing a doubtful case (FAQ #16 + 6 clean client videos in training).
    strong = [i for i, v in enumerate(idp) if v >= id_thr]
    id_seen = len(strong) >= ID_MIN_FRAMES
    conf = round(1.0 - (max(idp) if idp else 0.0), 2)
    res = EpisodeResult(None, times[i_hand], conf, times[strong[0]] if strong else None, **base)
    if not id_seen:
        res.kind = "collector_no_id" if bag_first else "client_no_id"
    return res
