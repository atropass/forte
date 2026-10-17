"""Frame extraction with ffmpeg.

Frames are sampled on a fixed time grid (t = k / fps). ffmpeg's fps filter picks, for each grid
point, the source frame whose PTS is closest, so timestamps are correct for VFR sources (all
training videos are VFR, see DECISIONS.md #4).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    rotation: int = 0


def probe(path: str | Path) -> VideoInfo:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height:stream_side_data=rotation:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    st = j["streams"][0]
    rot = 0
    for sd in st.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    w, h = st["width"], st["height"]
    if abs(rot) in (90, 270):  # ffmpeg autorotates on decode
        w, h = h, w
    return VideoInfo(w, h, float(j["format"]["duration"]), rot)


def _out_size(info: VideoInfo, max_side: int | None) -> tuple[int, int]:
    if not max_side or max(info.width, info.height) <= max_side:
        return info.width, info.height
    s = max_side / max(info.width, info.height)
    return int(round(info.width * s / 2) * 2), int(round(info.height * s / 2) * 2)


def extract_frames(path: str | Path, fps: float = 2.0, max_side: int | None = None,
                   start: float = 0.0, end: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps[N], frames[N,H,W,3] uint8 RGB) sampled at `fps`."""
    info = probe(path)
    w, h = _out_size(info, max_side)
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", f"{max(end - start, 0.01):.3f}"]
    cmd += ["-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    frames = np.frombuffer(raw, np.uint8, count=n * w * h * 3).reshape(n, h, w, 3)
    ts = start + np.arange(n) / fps
    return ts, frames


def frame_at(path: str | Path, t: float, max_side: int | None = None) -> np.ndarray:
    """Single full-quality frame nearest to time t (accurate seek)."""
    info = probe(path)
    w, h = _out_size(info, max_side)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(t, 0):.3f}", "-i", str(path),
           "-frames:v", "1", "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    if len(raw) < w * h * 3:  # t beyond last frame -> take the last one
        cmd[5] = f"{max(info.duration - 0.1, 0):.3f}"
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw[: w * h * 3], np.uint8).reshape(h, w, 3)
