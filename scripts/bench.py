"""Latency benchmark for the time budget. Re-run on the deployment GPU:  make bench

Simulates the per-video workload on the longest training clip (X-14, 142.5 s):
  1. frame extraction at PIPE_FPS (default 2) downscaled to 640 px
  2. person detector on every sampled frame (batched)
  3. VLM presence pass: one "how many people inside" call every 2 s of video (N_PRESENCE, default 72)
  4. VLM: N_SINGLE single-image questions + N_MULTI 4-frame questions, sent as one batch
     (vLLM batches them; MLX runs them sequentially)
Env: VLM_BACKEND, VLM_MODEL, DET_MODEL, DEVICE, N_SINGLE (default 40), N_MULTI (default 8).
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
    n_single, n_multi = int(os.getenv("N_SINGLE", "40")), int(os.getenv("N_MULTI", "8"))
    n_presence = int(os.getenv("N_PRESENCE", "72"))
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
    full = [frame_at(VIDEO, 3 + i * 3) for i in range(n_single)]
    vlm.ask([("cashier_at_desk", [full[0]])])  # warm-up
    from pipeline.vlm import QUESTIONS
    pres = [frame_at(VIDEO, 0.5 + 2 * i) for i in range(n_presence)]
    t = time.time()
    vlm.raw([(QUESTIONS["presence"]["text"].strip(), [f]) for f in pres], max_tokens=40)
    r["vlm_presence_batch_s"] = round(time.time() - t, 2)
    t = time.time()
    vlm.ask([("safe_open" if i % 2 else "money_on_desk", [f]) for i, f in enumerate(full)])
    r["vlm_single_batch_s"] = round(time.time() - t, 2)
    multi = [[frame_at(VIDEO, 5 + i * 15 + k, max_side=512) for k in range(4)] for i in range(n_multi)]
    t = time.time()
    vlm.ask([("id_shown", m) for m in multi])
    r["vlm_multi_batch_s"] = round(time.time() - t, 2)
    r["vlm_s_per_single"] = round(r["vlm_single_batch_s"] / n_single, 3)
    r["vlm_s_per_multi"] = round(r["vlm_multi_batch_s"] / n_multi, 3)
    r["projected_total_s"] = round(r["extract_s"] + r["detector_s"] + r["vlm_presence_batch_s"]
                                   + r["vlm_single_batch_s"] + r["vlm_multi_batch_s"], 1)
    r["budget_ok_150s"] = r["projected_total_s"] <= 150
    out = ROOT / "reports" / f"bench_{r['host'].split('.')[0]}_{backend.name}.json"
    out.write_text(json.dumps(r, indent=1))
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
