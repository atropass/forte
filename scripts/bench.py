"""Latency benchmark for the time budget. On the GPU server (backend stopped, see README 1.1):
    docker compose run --rm backend python3 scripts/bench.py

Simulates the per-video workload on the longest training clip (X-14, 142.5 s):
  1. frame extraction at PIPE_FPS (default 2) downscaled to 640 px + person detector on every frame
  2. scene-state pass: one JSON call every 3 s (+ refinement), N_PRESENCE (default 70)
  3. yes/no checks in decision mode (1 forward pass each), N_SINGLE (default 60)
  4. hand-over episode frames, 6-way choice, N_HANDOVER (default 150 = 3 episodes x 25 s x 2 fps)
Each stage is sent as one batch (vLLM batches it; MLX runs it sequentially).
Env: VLM_BACKEND, VLM_MODEL, DET_MODEL, DEVICE, N_PRESENCE, N_SINGLE, N_HANDOVER.
Writes reports/bench_<host>_<backend>.json.
"""
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline.frames import extract_frames, frame_at  # noqa: E402

VIDEO = next(p for p in (ROOT / "open" / "videos").glob("*.mov") if p.stem.endswith("-14"))


def main():
    import torch
    from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

    fps = float(os.getenv("PIPE_FPS", "2"))
    n_single, n_hand = int(os.getenv("N_SINGLE", "60")), int(os.getenv("N_HANDOVER", "150"))
    n_presence = int(os.getenv("N_PRESENCE", "70"))
    dev = os.getenv("DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    det_name = os.getenv("DET_MODEL", "PekingU/rtdetr_v2_r18vd")
    r = {"host": socket.gethostname(), "platform": platform.platform(), "device": dev, "video_sec": 142.5}

    t = time.time()
    ts, frames = extract_frames(VIDEO, fps=fps, max_side=640)
    r["extract_s"] = round(time.time() - t, 2)
    r["n_frames"] = len(frames)

    proc = AutoImageProcessor.from_pretrained(det_name)
    model = RTDetrV2ForObjectDetection.from_pretrained(det_name).to(dev).eval()
    if dev == "cuda":
        model = model.half()
    with torch.no_grad():
        def run(batch):
            inp = proc(images=list(batch), return_tensors="pt").to(dev)
            if dev == "cuda":
                inp["pixel_values"] = inp["pixel_values"].half()
            out = model(**inp)
            return proc.post_process_object_detection(out, target_sizes=[f.shape[:2] for f in batch], threshold=0.4)
        run(frames[:4])
        t = time.time()
        for i in range(0, len(frames), 16):
            run(frames[i:i + 16])
        if dev == "cuda":
            torch.cuda.synchronize()
        r["detector_s"] = round(time.time() - t, 2)
    r["detector_ms_per_frame"] = round(1000 * r["detector_s"] / len(frames), 1)
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    from pipeline.vlm import VLM
    from pipeline.vlm_backends import make_backend
    t = time.time()
    backend = make_backend()
    r["vlm_backend"], r["vlm_model"] = backend.name, backend.model_id
    r["vlm_load_s"] = round(time.time() - t, 1)
    vlm = VLM(backend, max_side=848)  # no cache: we want real latency
    full = [frame_at(VIDEO, 2 + i * 2.2) for i in range(n_single)]
    vlm.ask([("cashier_at_desk", [full[0]])])  # warm-up
    from pipeline import handover
    from pipeline.vlm import QUESTIONS
    pres = [frame_at(VIDEO, 0.5 + 2 * i) for i in range(n_presence)]
    t = time.time()
    vlm.raw([(QUESTIONS["state"]["text"].strip(), [f]) for f in pres], max_tokens=60)
    r["vlm_state_batch_s"] = round(time.time() - t, 2)
    t = time.time()
    vlm.prob([("safe_open" if i % 2 else "money_on_desk", [f]) for i, f in enumerate(full)])
    r["vlm_checks_batch_s"] = round(time.time() - t, 2)
    hand = [frame_at(VIDEO, 2 + 0.5 * i) for i in range(n_hand)]
    t = time.time()
    vlm.choose([(handover.PROMPT, [f], handover.OPTIONS) for f in hand])
    r["vlm_handover_batch_s"] = round(time.time() - t, 2)
    r["vlm_s_per_state"] = round(r["vlm_state_batch_s"] / n_presence, 3)
    r["vlm_s_per_check"] = round(r["vlm_checks_batch_s"] / n_single, 3)
    r["vlm_s_per_handover_frame"] = round(r["vlm_handover_batch_s"] / n_hand, 3)
    r["projected_total_s"] = round(r["extract_s"] + r["detector_s"] + r["vlm_state_batch_s"]
                                   + r["vlm_checks_batch_s"] + r["vlm_handover_batch_s"], 1)
    r["budget_ok_60s"] = r["projected_total_s"] <= 60  # 5 jury videos queued must each finish < 5 min
    r["budget_ok_150s"] = r["projected_total_s"] <= 150
    out = ROOT / "reports" / f"bench_{r['host'].split('.')[0]}_{backend.name}.json"
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
