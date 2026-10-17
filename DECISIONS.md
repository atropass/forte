# Decisions log

| # | Date | Decision | Why |
|---|---|---|---|
| 1 | 2026-10-17 | Use Latin `X-n` as the canonical video id. | Sheet and filenames mix Cyrillic Х / Latin X. |
| 2 | 2026-10-17 | The Excel file has 4 columns: col 3 is the verdict, col 4 the description. Adapted the parser instead of the assumed 3-column layout. | Real file structure. |
| 3 | 2026-10-17 | Merge duplicates X-47→X-4 and X-48→X-5 in eval. Accept either 57 or 68 s for X-4. | Identical MD5. |
| 4 | 2026-10-17 | Timestamps come from PTS (`ffprobe`/decoder), never frame_idx/30. | All files are VFR (avg 29.76–30.0 fps). |
| 5 | 2026-10-17 | Added two types not in the seed list: `client_admitted_workplace_not_ready` (X-30) and `personal_items_on_desk` (X-40, X-46). | Teammate descriptions. |
| 6 | 2026-10-17 | X-1/X-44 tentatively `unchecked_rejected_banknotes`, X-13 `unknown`. Flagged for teammate confirmation; excluded from per-type accuracy until confirmed (still count for verdict recall). | No description in the sheet. |
| 7 | 2026-10-17 | **Hardware differs from the brief:** this machine is an Apple M1 Pro with 16 GB unified memory and no CUDA. Phase 2 will benchmark the VLM via MLX/MPS locally. The production target (Linux + NVIDIA, vLLM) stays the deployment path in Docker; code keeps a backend switch (`mlx` / `transformers` / `vllm`). | The environment section in the brief was left as a template (`<FILL IN>`). |
| 8 | 2026-10-17 | Timecode = start of the triggering behaviour (see data_audit §4). | Matches the labeller's convention on 19/21 videos. |
| 9 | 2026-10-17 | Team answers for X-1/X-13/X-44 types and for the X-4/X-47 timecode were left as placeholders. Kept: count for detection, exclude from type accuracy; accept 57 or 68 s for X-4. Eval uses only X-4 and X-5 (not X-47/X-48). | Team reply, 2026-10-17 |
| 10 | 2026-10-17 | Timecode for client_no_id = moment money is handed over (as in X-2). Eval accepts both moments for X-34 (handover ≈7 s, label 13 s). | Team reply |
| 11 | 2026-10-17 | Default VLM = Qwen3-VL-4B-Instruct (Apache-2.0). On our questions it matches or beats Qwen3-VL-8B and Qwen2.5-VL-7B and is ~1.7× faster (reports/feasibility.md). | Phase 2 |
| 12 | 2026-10-17 | Room occupancy from the VLM ("how many people INSIDE", 21/21), not from detector counts. The detector can't tell people behind the glass wall from people inside. | Phase 2 |
| 13 | 2026-10-17 | Object-state questions use crops around VLM-grounded objects (safe/desk/laptop/door). Grounding happens per video, so nothing is hardcoded to pixels and it carries over to new angles. | crop ≫ full frame in Phase 2 |
| 14 | 2026-10-17 | key_in_safe: VLM infeasible (3 px lock). Plan: lock patch located by VLM grounding + small HOG/logistic classifier with leave-one-video-out validation. | Phase 2 |
| 15 | 2026-10-17 | client_no_id / collector_no_id stay in the rule set but only fire on high-confidence answers. Documented as a resolution-limited weak spot. | Phase 2 |
| 16 | 2026-10-17 | VLM backend interface `VLMBackend` with `mlx` (dev) and `vllm` (deployment) implementations, selected by env VLM_BACKEND / VLM_MODEL; shared prompts (questions.yaml) and parsing. | Team instruction |
| 17 | 2026-10-17 | Adopted the team catalog (V01–V19) as the type taxonomy: API `type` = catalog type_ru; extra fields `type_code`, `subtype`. Our specific rules map onto codes (6 rules → V03). Kept two codes not in the catalog: X01 (laptop unlocked, X-11/X-28) and X02 (client before workplace ready, X-30). | Team request |
| 18 | 2026-10-17 | Every catalog citation was checked against the act text. Not used: No. 231 п.7 (AML due diligence, not visual ID), п.75 (vault entry, not cash room), No. 49 (exchange offices). Used instead/added: п.80 (keys to cash storage devices) for key-in-safe. See legal/catalog_verification.md. | No fabricated or misapplied citations |
| 19 | 2026-10-17 | V10 (counting without a supervisor) deliberately disabled: clean training videos show a cashier counting alone. V06 disabled until the team confirms X-1/X-44. New rule V02 (cashier leaves while a client is at the desk) added, untested. | Avoid false positives |
| 20 | 2026-10-17 | Yes/no and multiple-choice checks use **decision mode**: probability of the first answer token, one forward pass, no generated text. Thresholds per question are calibrated on the deployed model (`scripts/calibrate.py`). | Removes "unclear" (X-11 miss), faster, gives a confidence |
| 21 | 2026-10-17 | V05/V13 decided from the action sequence in the visitor episode (2 fps desk-zone crops, 6-way choice per frame), not from one frame. A phone screen counts as ID (п.19 пп.1: digital documents). Optional ensemble with a second local model (`VLM_AUX_MODEL`) on these frames only. | ID card ≈ banknotes at 464×848 |
| 22 | 2026-10-17 | Scene-state pass is adaptive: coarse 1.5/2.5/3 s, then 0.5-s refinement only between samples whose state differs. | ~2× fewer calls, 0.5-s timecode resolution at events |
| 23 | 2026-10-17 | Rejected: OpenRouter "Jev" (cloud API, disqualifying), OpenJev (CC-BY-NC), JEV-27B-VL (7× larger, slower), Laya/NanoJev (text-only / no licence). | Locality and licence rules |
| 24 | 2026-10-17 | Default model in code corrected to Qwen3-VL-4B (was 8B). Calibration and bench run in a one-off container with the backend stopped (a second vLLM instance does not fit next to the running one). | |
