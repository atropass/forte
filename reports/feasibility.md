# Phase 2 — Perception feasibility

Machine: Apple M1 Pro 16 GB (MLX, MPS), shared with other apps (the system was swapping ~5 GB during the runs).
Mac latencies are **pessimistic**; re-run `make bench` on the 24 GB NVIDIA server (vLLM) to confirm the budget.

Reproduce: `make feasibility VLM_MODEL=<id>` (items in `scripts/feasibility_items.yaml`), then
`scripts/prompt_experiments.py`, `scripts/key_experiment.py`. Raw results: `reports/feasibility/*.json`,
failing frames: `reports/feasibility/fail/<model>/`, grounding overlays: `reports/feasibility/grounding/<model>/`.

## 1. Test set
Positives = labelled violation moments; negatives = matching moments in the clean "twin" videos
(X-21/23/25/27/29/31 are the no-violation counterparts of X-20/22/24/26/28/30). 3–9 items per question
(small: treat ±15 pp as noise). "crop" = crop around the object grounded by the VLM itself (§3), upscaled to ≥448 px.

## 2. Accuracy per question (all MLX 4-bit)
| question | mode | Qwen3-VL-4B | Qwen3-VL-8B | Qwen2.5-VL-7B |
|---|---|---|---|---|
| safe_open | full | **1.00** | 0.56 | 0.56 |
| safe_open | crop | **1.00** | **1.00** | 0.56 |
| key_in_safe | crop | 0.00 (always "unclear") | 0.57 (always "no") | 0.57 (always "no") |
| money_on_desk | crop | 0.62 → **0.75** with tight crop (pad 0.05) | 0.75 | 0.50 |
| bag_on_desk | crop | **1.00** | 0.86 | **1.00** |
| laptop_on (work content vs lock screen/off) | crop | **1.00** | 0.86 | 0.57 |
| door_open | full → crop | 0.57 → **0.86** (door crop) | 0.57 | 0.57 |
| personal_items | crop | **1.00** | **1.00** | **1.00** |
| cashier_at_desk | full | **1.00** | 0.75 | 0.62 |
| **presence: people INSIDE the room** (21 frames) | full 848 px | **21/21** (20/21 vs my label; my label was wrong) — 18/21 at 448 px | – | – |
| id_shown | 4 frames | 0.50 | 0.17 | 0.83 (still wrong on X-34) |
| handover first item (multi-choice) | 4–6 frames | 0.43 (biased to "id_document") | – | – |
| bag_handover | 4 frames | 0.75 | 0.75 | 1.00 |
| banknote_manual_check | 4 frames | 0.67 | 0.67 | 0.33 |
| scene grounding (safe/desk/door/laptop/machine) | full | good on all 4 setups; door box often too large | similar | – |

Median uncached latency per call on the Mac: **4B 2.4 s**, 8B 4.1 s, 7B 4.1 s.

**Choice: Qwen3-VL-4B-Instruct (Apache-2.0) as default.** On the narrow crop-based questions it matches or
beats the 8B and Qwen2.5-7B, and it is ~1.7× faster. 8B stays selectable (`VLM_MODEL`) for the GPU server.

## 3. What works, what doesn't, and what we do about it
| Signal | Status | Plan |
|---|---|---|
| Person detection (RT-DETRv2-r18) | Reliable for people at the desk; also fires on people **behind the glass wall** and on reflections; misses some seated cashiers in setup B | Use it for positions/motion only. Filter: score ≥ 0.5, height ≥ 0.05, feet below desk/door line |
| Who is inside the room | Detector alone cannot separate "just walked out, visible through the glass" from "standing inside" (same box height; door boxes unreliable) | **VLM presence count at 0.5–1 fps** (21/21). Occupancy timeline = VLM count, detector refines when the cashier starts leaving |
| Safe open, bag on desk, laptop work screen, personal items | ≥ 0.86–1.00 with crops | VLM on crops, 2-of-3 frames voting |
| Money on desk | 0.75 with tight crop; misses small single bundles (X-38, X-46) | tight desk crop + 3 frames; any-of-3 on the "cashier left" trigger (miss is worse than a false alarm there) |
| Door open | 0.86 with door crop | door crop, 3 frames after someone enters |
| **Key in safe lock** | **Not feasible with a VLM**: the lock is ~3 px wide at 464×848. At 4× zoom a short bar is visible in X-22/X-46 (fainter in X-24) and absent in clean videos, but all 3 VLMs refuse or answer "no", even on 10× lock patches | Lock is located reliably by VLM grounding inside the safe crop → **small trained classifier on lock patches** (HOG + logistic regression, leave-one-video-out) in Phase 3/4. Report as low-confidence in LIMITATIONS |
| **Client showed ID before money** | **Not reliably separable** (card ≈ small cash stack, ~10 px). Per-frame card/cash questions say "card" in the no-ID videos too | Keep the type, run it only with high-confidence VLM answers, document as a known weak spot. A 2× higher-res camera would fix it |
| Collector handed bag without documents | bag handover detected 0.75–1.0; the document part has the same resolution problem as ID | same approach as ID; low confidence |
| Rejected banknotes not checked (X-1/X-44 hypothesis) | manual-check question 0.67; labels unconfirmed | lowest priority until the team confirms the type |

## 4. Time budget (145 s video)
Mac, Qwen3-VL-4B MLX (`make bench`, before the presence pass was added): extract 1.4 s · detector 14.7 s
(52 ms/frame, 285 frames at 2 fps) · 40 single-image checks 94.7 s · 8 multi-frame checks 26.0 s → **137 s**.
The presence pass at 0.5 fps adds ~72 calls ≈ 170 s on the Mac → ~5 min on the Mac.

On the GPU server, vLLM batches all calls of one stage. Expected: an 848×464 image ≈ 500 visual tokens
→ ~110 calls ≈ 60k prefill tokens, a few seconds on a 4090/L4 for a 4B model. Target ≤ 150 s total looks
safe. **To confirm: `VLM_BACKEND=vllm VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct make bench`.** Keep the result
in `reports/bench_<host>_vllm.json`.

Budget levers if the server is slower than expected: presence at 0.33 fps away from detector transitions,
448 px for presence away from exits (18/21), cap the number of periodic checks.
