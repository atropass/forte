"""Event timeline from per-frame in-room person boxes.

Events (all with timestamp in seconds and a confidence):
  cashier_left      room becomes empty; t = moment the cashier started moving away from her workstation
  cashier_entered   someone enters an empty room (the cashier)
  visitor_arrived   occupancy grows while the cashier is present (client / collector)
  visitor_left      occupancy shrinks back while someone stays
  second_visitor    two visitors at the same time (occupancy >= 3)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Event:
    kind: str
    t: float
    t_end: float | None = None
    conf: float = 1.0
    info: dict = field(default_factory=dict)


def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) == 0:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, pad, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))]).round().astype(int)


def runs(mask: np.ndarray):
    """Yield (start, end_exclusive) index runs where mask is True."""
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            yield i, j
            i = j
        else:
            i += 1


def center(b):
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


def build_timeline(ts: np.ndarray, boxes: list[list[np.ndarray]], frame_wh: tuple[int, int],
                   fps: float, desk_rel: list | None = None) -> tuple[list[Event], dict]:
    """boxes: in-room person boxes per frame (pixels)."""
    w, h = frame_wh
    n = len(ts)
    raw = np.array([len(b) for b in boxes], dtype=int)
    k = max(3, int(round(fps * 2)) | 1)  # ~2 s median window, odd
    occ = median_filter(raw, k)

    # workstation point W: where the single occupant (= cashier) usually is
    solo = [center(b[0]) for b, o in zip(boxes, occ) if o == 1 and len(b) == 1]
    if len(solo) >= 3:
        W = np.median(np.array(solo), axis=0)
    elif desk_rel:
        W = np.array([(desk_rel[0] + desk_rel[2]) / 2 * w, (desk_rel[1] + desk_rel[3]) / 2 * h])
    else:
        W = np.array([w / 2, h / 2])
    radius = 0.12 * float(np.hypot(w, h))

    def near_W(i):
        return any(np.linalg.norm(center(b) - W) < radius for b in boxes[i])

    events: list[Event] = []
    min_empty = max(2, int(round(fps * 1.5)))
    empty_runs = [(a, b) for a, b in runs(occ == 0) if b - a >= min_empty]
    for a, b in empty_runs:
        if a == 0:
            continue  # video starts empty -> nobody left
        # walk back to the moment the cashier left the workstation
        j = a - 1
        while j > 0 and not near_W(j):
            j -= 1
        t_leave = ts[min(j + 1, a - 1)]
        events.append(Event("cashier_left", float(t_leave), float(ts[min(b, n - 1)]),
                            conf=min(1.0, (b - a) / (fps * 3)), info={"t_room_empty": float(ts[a])}))
        if b < n:
            events.append(Event("cashier_entered", float(ts[b]), info={}))
    if len(occ) and occ[0] == 0:
        first_in = next((i for i in range(n) if occ[i] > 0), None)
        if first_in is not None and not any(e.kind == "cashier_entered" and abs(e.t - ts[first_in]) < 0.01 for e in events):
            events.append(Event("cashier_entered", float(ts[first_in])))

    # visitors: occupancy increases while someone is already inside
    prev = occ[0] if n else 0
    for i in range(1, n):
        if occ[i] > prev and prev >= 1:
            # first raw frame of the increase (more precise than the filtered one)
            j = i
            while j > 0 and raw[j - 1] > prev:
                j -= 1
            kind = "second_visitor" if occ[i] >= 3 else "visitor_arrived"
            events.append(Event(kind, float(ts[j]), info={"occupancy": int(occ[i])}))
        elif occ[i] < prev and occ[i] >= 1:
            events.append(Event("visitor_left", float(ts[i]), info={"occupancy": int(occ[i])}))
        prev = occ[i]

    events.sort(key=lambda e: e.t)
    debug = {"occ": occ.tolist(), "raw": raw.tolist(), "W": W.tolist(), "radius": radius}
    return events, debug


# ---------------------------------------------------------------------------------------------------------
# v2: events from the VLM scene-state pass (inside count, cashier seated, visitor at desk), refined by the
# detector. The VLM decides WHO is inside (it separates the room from the corridor behind the glass); the
# detector, sampled more densely, gives WHEN the cashier started moving away.

@dataclass
class StateSample:
    t: float
    inside: int
    seated: bool
    visitor: bool


def _smooth_counts(samples: list[StateSample]) -> list[int]:
    """Median-of-3 over the inside count; isolated single-sample blips are removed."""
    c = [s.inside for s in samples]
    if len(c) < 3:
        return c
    out = [c[0]]
    for i in range(1, len(c) - 1):
        out.append(int(np.median(c[i - 1:i + 2])))
    out.append(c[-1])
    return out


def _leave_start(t_empty: float, det_ts: np.ndarray, det_boxes: list[list[np.ndarray]], frame_wh,
                 max_back: float = 8.0, move_thr: float = 0.012) -> float:
    """Walk back from the moment the room is empty to the last moment the cashier was still (= stood up /
    turned away). Uses the largest person box per detector frame; relative center motion per sample."""
    w, h = frame_wh
    idx = [i for i, t in enumerate(det_ts) if t_empty - max_back <= t <= t_empty]
    if len(idx) < 3:
        return max(0.0, t_empty - 2.0)
    track = []
    for i in idx:
        bs = det_boxes[i]
        if not bs:
            track.append(None)
            continue
        b = max(bs, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        track.append(np.array([(b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h, (b[3] - b[1]) / h]))
    # last index with a box before empty
    last = max((k for k, v in enumerate(track) if v is not None), default=None)
    if last is None:
        return max(0.0, t_empty - 2.0)
    k = last
    while k > 0 and track[k - 1] is not None:
        d = np.abs(track[k] - track[k - 1]).max()
        if d < move_thr:  # still -> movement starts at k
            break
        k -= 1
    return float(det_ts[idx[k]])


LEAVE_DIRECT_S = 4.0  # stand-up this close to the room becoming empty = walked straight out


def events_from_state(samples: list[StateSample], duration: float, det_ts: np.ndarray,
                      det_boxes: list[list[np.ndarray]], frame_wh,
                      leave_onsets: dict[float, float] | None = None) -> tuple[list[Event], dict]:
    """leave_onsets: {t_room_empty: start of the walk to the door} from the VLM (used instead of detector boxes)."""
    events: list[Event] = []
    if not samples:
        return events, {}
    occ = _smooth_counts(samples)
    ts = [s.t for s in samples]
    n = len(samples)
    # room empty runs -> cashier_left / cashier_entered
    i = 0
    while i < n:
        if occ[i] == 0:
            j = i
            while j < n and occ[j] == 0:
                j += 1
            if i > 0:
                t_empty = ts[i]
                # leave start: the last "seated -> not seated" change before the room got empty (stood up);
                # if she was not seated (e.g. at the safe), the detector motion onset
                k = i - 1
                while k > 0 and not samples[k].seated and t_empty - ts[k] < 10.0:
                    k -= 1
                stood_up = (ts[k] + ts[k + 1]) / 2 if samples[k].seated and k < i - 1 else None
                if stood_up is not None and t_empty - stood_up <= LEAVE_DIRECT_S:
                    t_leave = stood_up  # stood up and walked straight out
                else:
                    # stood up long before leaving (went to the safe first) or never seen seated: the timecode is
                    # the start of the walk to the door = last moment the cashier stood still (detector)
                    if leave_onsets is not None and round(t_empty, 2) in leave_onsets:
                        t_leave = leave_onsets[round(t_empty, 2)]
                    else:
                        t_leave = _leave_start(t_empty, det_ts, det_boxes, frame_wh)
                    t_leave = max(t_leave, ts[i - 1] - 6.0, stood_up or 0.0)
                events.append(Event("cashier_left", round(t_leave, 2), ts[j] if j < n else duration,
                                    conf=1.0 if j - i >= 2 else 0.6, info={"t_room_empty": t_empty}))
            if j < n:
                events.append(Event("cashier_entered", ts[j - 1] + (ts[j] - ts[j - 1]) / 2 if j > 0 else ts[j]))
            i = j
        else:
            i += 1
    # visitors: count increases while somebody is inside
    for k in range(1, n):
        if occ[k] > occ[k - 1] and occ[k - 1] >= 1:
            t_arr = ts[k - 1] + (ts[k] - ts[k - 1]) / 2
            if occ[k] >= 3:
                sustained = k + 1 < n and occ[k + 1] >= 3
                if sustained:
                    events.append(Event("second_visitor", round(t_arr, 2), info={"occupancy": occ[k]}))
            else:
                events.append(Event("visitor_arrived", round(t_arr, 2), info={"occupancy": occ[k]}))
        elif occ[k] < occ[k - 1] and occ[k] >= 1:
            events.append(Event("visitor_left", ts[k], info={"occupancy": occ[k]}))
    # visitor at the desk while the cashier's seat is empty, for >= 2 consecutive samples (V02)
    k = 0
    while k < n - 1:
        if samples[k].visitor and not samples[k].seated and samples[k + 1].visitor and not samples[k + 1].seated \
                and occ[k] >= 1:
            events.append(Event("cashier_absent_with_visitor", ts[k]))
            while k < n and samples[k].visitor and not samples[k].seated:
                k += 1
        k += 1
    # a video that starts with a visitor already present counts as one visitor episode at t=0
    if occ[0] >= 2:
        events.append(Event("visitor_arrived", 0.0, info={"occupancy": occ[0], "at_start": True}))
    events.sort(key=lambda e: e.t)
    return events, {"occ": occ, "t": ts, "seated": [s.seated for s in samples], "visitor": [s.visitor for s in samples]}
