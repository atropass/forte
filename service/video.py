"""Upload validation before a job is queued: container signature + ffprobe limits.

Rejecting a bad file here answers in milliseconds instead of failing minutes later in a worker, and keeps
oversized videos (the pipeline decodes the whole video into memory) away from the workers.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

ALLOWED_EXT = {".mov", ".mp4", ".m4v"}
# ISO-BMFF / QuickTime top-level atoms that can start a file
_ATOMS = {b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot", b"uuid"}


class InvalidVideo(ValueError):
    pass


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    codec: str


def looks_like_mp4(head: bytes) -> bool:
    return len(head) >= 8 and head[4:8] in _ATOMS


def probe(path: Path, timeout: float = 20) -> VideoInfo:
    cmd = ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-select_streams", "v:0",
           "-show_entries", "stream=codec_name,width,height,duration:format=duration,format_name",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=timeout, check=True).stdout
        j = json.loads(out or b"{}")
    except subprocess.TimeoutExpired:
        raise InvalidVideo("video could not be read in time") from None
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        raise InvalidVideo("not a readable video") from None
    streams = j.get("streams") or []
    if not streams:
        raise InvalidVideo("no video stream")
    st, fmt = streams[0], j.get("format") or {}
    if not any(n in (fmt.get("format_name") or "") for n in ("mov", "mp4")):
        raise InvalidVideo("container is not MOV/MP4")

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    dur = num(st.get("duration")) or num(fmt.get("duration"))
    return VideoInfo(duration=dur, width=int(st.get("width") or 0), height=int(st.get("height") or 0),
                     codec=str(st.get("codec_name") or ""))


def check_limits(info: VideoInfo, max_s: float, max_px: int):
    if info.duration <= 0:
        raise InvalidVideo("video has no duration")
    if info.duration > max_s:
        raise InvalidVideo(f"video longer than {int(max_s)} s")
    if max(info.width, info.height) > max_px or min(info.width, info.height) <= 0:
        raise InvalidVideo(f"video resolution must be at most {max_px} px on the longest side")
