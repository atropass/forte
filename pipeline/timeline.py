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
