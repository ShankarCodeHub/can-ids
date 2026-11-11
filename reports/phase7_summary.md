# Phase 7: Evaluation summary and interpretation (ROAD)

All tables are in `reports/RESULTS.md`, generated from `reports/results/` by
`scripts/make_results.py`. This file interprets them; every number quoted here appears in
`RESULTS.md`.

## Scope actually completed

| Experiment | Status |
|---|---|
| Within-dataset ROAD (Normal / Fuzzy / Spoofing) | Done (Phases 3–4) |
| Within-dataset HCRL (with DoS) | **Not run**: HCRL raw files are not on the evaluation machine |
| Cross-dataset HCRL ↔ ROAD | **Not run** (same reason). The code is ready and tested on synthetic data: `scripts/cross_dataset.py` |
| Ablation (4 variants + full) | Done, ROAD test |
| Runtime benchmark (1M frames) | Done (laptop). Raspberry Pi not attempted |
| Error analysis | Done, ROAD test |

## Main conclusions

1. **The supervised stage makes the IDS usable; Isolation Forest alone does not.**
   - Isolation Forest alone reaches detection 0.122 at FPR 0.0233.
   - The classifier and the tuned hybrid both reach about 0.64–0.67 detection at FPR 0.0011 (alarm precision about 0.70). The baseline rules reach 0.033.
2. **In this configuration, stage 1 adds nothing measurable.**
   - The `no_iforest` ablation equals `full` on every metric apart from 54 Unknown outputs.
   - When stage 1 is allowed to raise Unknown (no overrule), it adds 358 detected attack frames but also 141,951 false alarms.
   - The hybrid's main research question therefore gets a **negative answer on ROAD**: with Isolation Forest's false-alarm rate (2.3% on test, with temporal drift), the anomaly stage cannot be used to flag unseen attacks without flooding the operator.
3. **Timing and payload features detect different attack mechanisms.**
   - Without payload bytes, fabrication attacks are caught at about 99.8% (max-speedometer 0.9984, reverse-light 0.9983) with FPR 0.0001. That rate comes purely from timing and window features: injected frames shift the target ID's share and inter-arrival times.
   - The same model detects **0%** of masquerade attacks, which preserve timing.
   - The full model catches masquerade attacks through payload values, but its fabrication detection is only 0.52–0.72.
   - This argues for **two specialised detectors**, timing-based for fabrication and payload/sequence-based for masquerade, rather than one model over all features. That is future work.
4. **Part of the payload signal is spurious and recording-specific.** Three independent pieces of evidence agree:
   - SHAP (Phase 5) ranks payload byte `b1` first, and the false-alarm explanation has nearly the same reasons as a real attack (`b1` = 0x73).
   - Error analysis: missed spoofing frames have median `b1` of 72–79, while detected ones have 112–114 (same ID, same window share).
   - Error analysis: **97% of false positives (6,580 of 6,769) are on ID 0x0D0 inside attack recordings**, mostly outside the injection window. This includes the coolant captures, whose attack targets a different ID. The FPR in ambient captures stays at about 0.05%.

   `b1` of 0x0D0 most likely encodes vehicle state, such as speed, which differs between recordings. The classifier partly learned the driving context of the attack recordings, not only the attack. This is a threat to validity of the payload-based results, and it is caught only because the split held out whole recordings.
5. **Unseen attacks remain undetected.** The coolant-temperature attack, on an ID never targeted in training, scored 0/43 frames in every variant.
6. **Real-time performance is far above what a CAN bus can carry.**
   - On an AMD Ryzen 5 7520U laptop, the full pipeline processes **16,573 frames/s** (latency p95 16.5 ms), and **10,655 frames/s** end to end through the virtual bus (p95 20.9 ms).
   - By comparison, a 500 kbit/s bus at 100% load carries 3,704–4,505 frames/s of 8-byte frames, or 10,638 frames/s of 0-byte frames.
   - Explaining every alert inline costs 17% of throughput (13,688 frames/s), so SHAP stays asynchronous.
   - Micro-batching is essential: model calls per frame manage only 133 frames/s, against 11,794 frames/s batched on the same 5,000 frames.
   - The models are small: Isolation Forest 2.3 MiB, XGBoost 0.26 MiB. Peak RAM during the 1M-frame benchmark was 944 MiB, most of it the pre-loaded input frames.
7. **Smaller data-quality findings.**
   - 2 of the 6,769 test false positives carry the exact fuzzing payload, so they are probably label noise at injection-window edges (Phase 6).
   - 57 false positives are cold-start frames (the first frame of an ID).
   - The hybrid produces almost no false positives in the drift captures (1), because the classifier overrules Isolation Forest there.

## Threats to validity

- **One dataset, one vehicle.** Without HCRL, nothing can be claimed about generalisation across vehicles.
- **Few recordings per attack:** 3 per type (1 for coolant). Per-attack numbers depend on which recording landed in test.
- **Fuzzy has 114 test frames.** ROAD's Fuzzy labels are defined by the same all-0xFF payload the model learns.
- **Labels come from metadata intervals**, with a small amount of noise at the edges.
- **Ablation hyperparameters** are fixed from the full model (not re-searched per variant). Thresholds were re-tuned on validation.
- **Frame-level metrics only.** No event-level ("was the attack noticed?") evaluation.
