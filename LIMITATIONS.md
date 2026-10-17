# Limitations and known failure cases

Results are hints for a human reviewer, not a legal or disciplinary conclusion. Measured numbers are in
`reports/eval.md` (training videos) and `reports/robustness.md` (geometric augmentations); the summary
below is updated from them.

## 1. Resolution-limited checks (464×848 video)
| What | Why it fails | Effect |
|---|---|---|
| Key left in the safe lock (V03, X-22/X-24) | The lock is ~3 px wide. At 4× zoom a short bar is visible, but no VLM (4B/8B/Qwen2.5-7B) can read it, and a trained lock-patch classifier learns the camera setup instead of the key (`reports/keyclf.md`) | **Missed.** X-46 is still flagged via V01 |
| Client showed an ID before handing money (V05, X-2/X-34) | An ID card and a small stack of banknotes are both ~10 px. Single-frame "card vs cash" answers are the same in ID and no-ID videos. Now decided on the **sequence** in the episode (card / phone / examining before cash), see `pipeline/handover.py` | Only 2 no-ID episodes in training: reliability unknown until tested on more footage |
| Collector showed documents (V13, X-17) | The bag handover is visible; the documents are not | Low recall |
| Small single bundles of money (V03) | 0.75 accuracy on tight desk crops; misses one-note bundles | Possible misses |
A higher-resolution camera (≥ 1080p) over the desk would fix all four.

## 2. Things the system deliberately does not judge
* Anything that needs data outside the video: ledger entries, ABIS records, document contents,
  signatures and stamps (V08), strap and bundle markings (V12), whether a document was filled in (V11).
* Counting without a supervisor (V10): a cashier counting alone appears in **clean** training videos, so
  the dataset treats it as normal.
* Faces and identities: never. Roles come from position and behaviour (who sits at the cashier's chair,
  who comes in through the door).
* Audio: never.

## 3. Assumptions that can break on new footage
* **Room layout**: one desk, one safe, one door. Grounding runs per video and is not tied to pixels, but a
  room with several desks or safes is out of scope.
* **"Cashier left"** = the room has no people inside it (VLM count). If someone else stays in the room,
  rules that need an unattended workplace will not fire. V02 (a visitor at the desk while the cashier's
  seat is empty) covers part of this, but has no training example.
* **Timecodes** follow the labelling convention (the moment the behaviour *starts*). Scene-state sampling
  is 1 s (videos ≤ 40 s), 2 s (≤ 90 s) or 2.5 s (longer), so expect ±1–2 s.
* **Camera angle**: four setups are in the training data. Mirrored, rotated, zoomed and perspective-warped
  versions are measured in `reports/robustness.md`. Extreme angles (side-on, fisheye) are untested.
* **Unseen violation types**: a generic "valuables left unattended" open check runs whenever the cashier
  leaves. It reports only high-confidence, specific items, mapped to V03.

## 4. Data and label caveats
* 40 unique training videos (2 duplicate pairs). The X-1, X-13 and X-44 types were not confirmed by the
  team: X-1/X-44 are treated as V06 (disabled), X-13 as unknown.
* All thresholds were tuned on these 40 videos; per-video errors are listed so they can be checked.

## 5. Operational
* One GPU worker: jobs queue. A 145 s video must finish in ≤ 5 min including queueing. Confirm with
  `scripts/bench.py` on the target GPU.
* The Mac/MLX path is for development (~6 s of processing per second of video on an M1 Pro).
