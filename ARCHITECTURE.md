# Architecture

## Pipeline

```
 .mov ──► frames (ffmpeg, PTS-exact grid)                                         ┌──────────────┐
           │  2 fps, 640 px                                                         │ legal/       │
           ├─► RT-DETRv2-r18 ── person boxes ── motion onset (when the cashier ──┐  │ norms.yaml   │
           │                                     stood up / turned away)         │  └──────┬───────┘
           │  1 frame                                                            │         │
           ├─► Qwen3-VL grounding ── safe / desk / door / laptop boxes (per video)│         │
           │  coarse 1.5–3 s + 0.5-s refinement where state changes, 464×848     ▼         │
           ├─► Qwen3-VL scene state ── {people INSIDE, cashier seated, visitor at desk}    │
           │                     │                                                     │
           │                     ▼                                                     │
           │        timeline.py: cashier_left / cashier_entered / visitor_arrived /     │
           │        second_visitor / cashier_absent_with_visitor (debounced)            │
           │                     │                                                     │
           │                     ▼                                                     │
           │        rules.py plan(): which yes/no questions, on which frames & crops     │
           │                     │                                                     │
           ├─► Qwen3-VL checks, decision mode (1 forward pass → P(yes)) on object crops,  │
           │   2-of-3 frames or mean ≥ calibrated threshold                              │
           └─► visitor episodes, 2 fps desk-zone crops → P(card / phone / cash / bag /    │
               examining) per frame → sequence logic (ID before money?)  [+ optional 2nd model]
                                 │                                                     │
                                 ▼                                                     ▼
                     rules.py decide() ──► violations ──► evidence.py (JPEG + box) ──► JSON (spec schema)
```

| Module | Role |
|---|---|
| `pipeline/frames.py` | ffmpeg decoding; frames on a fixed time grid → timestamps correct for VFR sources |
| `pipeline/perception.py` | RT-DETRv2 detector, IoU tracker, scene boxes, in-room heuristics |
| `pipeline/vlm_backends.py` | `VLMBackend` interface: `mlx` (Mac dev) and `vllm` (GPU server), chosen by `VLM_BACKEND` |
| `pipeline/vlm.py` + `questions.yaml` | every prompt in one file; batching, on-disk cache, strict JSON parsing, crops |
| `pipeline/timeline.py` | events from the scene-state pass, refined by detector motion |
| `pipeline/handover.py` | visitor-episode sequence analysis for V05 (client ID) and V13 (collector documents) |
| `pipeline/rules.py` + `rules.yaml` | one entry per rule: trigger, checks, voting, timecode convention, description template, legal ref |
| `pipeline/legal.py` + `legal/norms.yaml` | legal references are looked up, never generated |
| `pipeline/evidence.py` | 1–3 evidence frames with the relevant object outlined |
| `pipeline/analyze.py` | orchestration, progress callback, exact API schema |
| `service/app.py` | stateless FastAPI: jury API (`X-API-Key`), web-UI sessions, upload validation, health/metrics |
| `service/worker.py`, `worker_main.py`, `jobqueue.py`, `reaper.py` | RQ workers (process per job, hard timeout, retry on GPU outage), job recovery |
| `service/state.py`, `auth.py`, `blob.py` | job state, accounts, API keys in Redis; files in S3 / shared disk |
| `service/cv_service.py` | RT-DETRv2 detector microservice on the GPU server |
| `forte-vision-ai-development/` | Next.js UI (static export), served by nginx in the `forte-frontend` image |
| `docker/` | `backend.Dockerfile` (api / worker / admin), `frontend.Dockerfile` + `nginx.conf` |

## Why this approach
* **40 videos are far too few for end-to-end video classification**, and the test set uses a new camera
  angle. Instead, a general VLM answers *narrow, verifiable* questions ("is the safe door open?") at
  moments chosen by deterministic logic. Rules stay readable, auditable and editable in YAML.
* **No fixed pixel zones.** The VLM grounds the safe, desk, door and laptop separately in each video, so
  the same code runs on all four training setups and on unseen angles.
* **Who is in the room comes from the VLM, not from detector counts.** The detector cannot separate people
  behind the glass wall, or the cashier who just walked out, from people inside (Phase 2; VLM 24/25).
  The detector is still used for *when* the cashier started leaving, at 2 fps.
* **Two signals per violation where possible**: an event from the timeline plus ≥2 of 3 frames agreeing on
  the VLM check. Periodic checks need two consecutive positive samples.
* **Decision mode instead of free text.** For yes/no and multiple-choice checks the model is not asked to
  write an answer. We read the probability of the first answer token (`VLMBackend.score`, logprobs in
  vLLM / MLX). One forward pass per frame: faster, no "unclear", and a confidence that thresholds can be
  calibrated on (`scripts/calibrate.py` → `thresholds.yaml`).
* **ID checks look at the sequence, not one frame.** An ID card and a stack of banknotes are both ~10 px,
  but "visitor holds out a card / shows a phone / cashier examines a small item → then cash goes to the
  counting machine" is visible. `pipeline/handover.py` scores every frame of a visitor episode and decides
  on the order of events. A phone screen counts as ID (digital documents, п.19 пп.1).
* **Legal references never come from a model.** They are looked up in `legal/norms.yaml`, and every clause
  was checked against the act text (`legal/catalog_verification.md`).

## Models and licences
Qwen3-VL-4B-Instruct (Apache-2.0), RT-DETRv2-r18 (Apache-2.0). Evaluated and rejected: Qwen3-VL-8B
(no accuracy gain, 1.7× slower), Qwen2.5-VL-7B. Not used: Ultralytics YOLO (AGPL), Qwen2.5-VL-3B
(non-commercial). Details in [MODELS.md](MODELS.md).

## Locality guarantee
* Weights are downloaded once by `scripts/download_weights.sh`. The backend runs with `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1`; any attempt to reach the hub fails instead of downloading.
* No cloud AI API anywhere in the code. vLLM runs in-process (`vllm.LLM`) with usage stats disabled.
* The UI is a static export with no analytics (Vercel Analytics was removed from the v0 template).
  Next.js telemetry is disabled at build time.
* Videos and results stay on the server's disk (`DATA_DIR`).

## Time budget
Per 145 s video on the GPU server, target ≤ 150 s: detector ~290 frames, ~60 scene-state calls,
~30–60 check calls, each stage batched by vLLM. Mac (M1 Pro, MLX 4-bit) is the pessimistic reference,
at ~2 min per 20 s clip. Confirm on the server with `scripts/bench.py`. Levers: `state_step()` in
`analyze.py`, `PERIODIC_STEP` in `rules.py`, `VLM_MAX_PIXELS`.

## Extending
* New violation type: add a question to `questions.yaml` (if needed), an entry to `rules.yaml` (trigger,
  checks, template, `legal` key) and a verified reference to `legal/norms.yaml`.
* New camera/room: nothing to configure; grounding runs per video.
* Bigger GPU: set `VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct`.
