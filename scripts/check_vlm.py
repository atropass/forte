"""Smoke test of the configured VLM backend (reads .env via the Makefile): reachability, model name,
one text answer, one decision-mode answer on a real training frame, latency.

make check-vlm
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    backend = os.getenv("VLM_BACKEND", "mlx")
    print(f"backend={backend} model={os.getenv('VLM_MODEL')} base={os.getenv('VLM_API_BASE', '-')}")
    if backend in ("openai", "remote"):
        import httpx
        base = os.getenv("VLM_API_BASE", "").rstrip("/")
        h = {"Authorization": f"Bearer {os.getenv('VLM_API_KEY', '')}"}
        try:
            r = httpx.get(f"{base}/models", headers=h, timeout=20)
        except httpx.HTTPError as e:
            sys.exit(f"FAIL: cannot reach {base}: {e}")
        if r.status_code != 200:
            sys.exit(f"FAIL: GET {base}/models -> HTTP {r.status_code} {r.text[:200]!r}. "
                     "Is the pod running and the port exposed? Is VLM_API_KEY right?")
        served = [m["id"] for m in r.json().get("data", [])]
        print("served models:", served)
        if os.getenv("VLM_MODEL") not in served:
            sys.exit(f"FAIL: VLM_MODEL={os.getenv('VLM_MODEL')} is not served; set VLM_MODEL to one of {served}")
    from pipeline.frames import frame_at
    from pipeline.vlm import VLM
    from pipeline.vlm_backends import make_backend
    vlm = VLM(make_backend(), max_side=848)
    vids = sorted((ROOT / "open" / "videos").glob("*.mov"))
    frame = frame_at(vids[0], 3.0) if vids else __import__("numpy").full((848, 464, 3), 128, "uint8")
    t = time.time()
    txt = vlm.raw([("Describe this frame in 10 words.", [frame])], max_tokens=30)[0]
    t1 = time.time()
    a = vlm.prob([("cashier_at_desk", [frame])])[0]
    t2 = time.time()
    print(f"text ({t1 - t:.1f}s): {txt!r}")
    print(f"decision mode ({t2 - t1:.1f}s): P(person at cashier desk) = {a.p:.2f}")
    print("OK")


if __name__ == "__main__":
    main()
