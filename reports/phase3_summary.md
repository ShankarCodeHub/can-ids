# Phase 3: Stage-1 anomaly detection (ROAD)

All numbers come from `reports/results/road_phase3_*.csv|json` (test) and from
`models/road/*/…metadata.json` / `road_if_tuning.csv` / `road_baseline_tuning.csv` (validation).
HCRL has not been evaluated yet because its raw files are not available on this machine.

## Protocol

- **Isolation Forest** is trained on Normal frames only: a seeded 1,000,000-frame sample of the 15.1M Normal training frames, using all 25 features.
- **Tuning uses validation only**: all 23,026 validation attack frames and a seeded sample of 300,000 Normal frames. Settings are chosen for maximum recall at FPR ≤ 1%, with PR-AUC as the tie-break.
- **The rule baseline is tuned on validation under the same FPR ≤ 1% constraint.**
  - The first grid of interval ratios (0.3–0.9) could not reach 1%; the best was 1.77% at 0.3.
  - A validation-only diagnosis showed the interval rule alone caused the false alarms: bus-load 0.000%, unknown-ID 0.0003%.
  - Finer ratios between 0.1 and 0.2 were added. The test set was not used for this.
- **The test set was scored once**, after both models were frozen. Every test access is logged in `reports/results/test_access_log.csv`.

## Selected settings (validation)

| Model | Setting | Val recall | Val precision | Val FPR | Val ROC-AUC | Val PR-AUC |
|---|---|---|---|---|---|---|
| Isolation Forest | 100 trees, max_samples 4096, max_features 1.0, threshold 0.51058 | 0.1886 | 0.5917 | 0.0100 | 0.8394 | 0.4068 |
| Rule baseline | rate quantile 0.999, interval ratio 0.14, unknown-ID on | 0.0285 | 0.2182 | 0.0078 | 0.6254 | 0.1164 |

## Held-out test results (6,121,249 frames, 24,510 attack frames)

| Model | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (rules) | 0.0184 | 0.0325 | 0.0235 | 0.0070 | 0.6758 | 0.0076 | 797 | 42,542 | 23,713 |
| Isolation Forest | 0.0207 | 0.1224 | 0.0354 | 0.0233 | 0.8396 | 0.0205 | 2,999 | 141,960 | 21,511 |

Figure: `reports/figures/road_phase3_roc_pr.png`

### Detection rate by attack type (test)

| Attack type | Frames | Baseline | Isolation Forest |
|---|---|---|---|
| fuzzing | 114 | 0.9649 | 1.0000 |
| correlated_signal | 1,265 | 0.0245 | 1.0000 |
| correlated_signal_masquerade | 1,263 | 0.0000 | 0.1211 |
| max_speedometer | 6,107 | 0.0519 | 0.2335 |
| max_speedometer_masquerade | 6,107 | 0.0000 | 0.0057 |
| reverse_light | 4,784 | 0.0679 | 0.0013 |
| reverse_light_masquerade | 4,784 | 0.0000 | 0.0000 |
| max_engine_coolant_temp (unseen target) | 43 | 0.3023 | 0.0000 |
| max_engine_coolant_temp_masquerade (unseen target) | 43 | 0.0233 | 0.0000 |

### False-positive rate by context (test)

| Context | Normal frames | Baseline FPR | Isolation Forest FPR |
|---|---|---|---|
| ambient captures | 4,891,861 | 0.00687 | 0.02536 |
| normal frames inside attack captures | 1,204,878 | 0.00742 | 0.01487 |

## Findings

1. **Isolation Forest beats the rule baseline on ranking quality.** Its ROC-AUC is 0.840 against 0.676, and its PR-AUC is 0.0205 against 0.0076. At the operating point it reaches about 3.8× the baseline's recall, but at a higher FPR (see finding 3).
2. **Frame-level detection works for attacks that change payload values or traffic composition, and fails for masquerade attacks.**
   - Fuzzing and correlated-signal frames are detected completely.
   - Masquerade versions, where the genuine frames are removed so timing is preserved, are rarely flagged.
   - Reverse-light injection changes a single byte to a value that is also common in normal traffic (see `road_rule_collisions.csv`), so it is essentially invisible to both detectors per frame.
   - Masquerade attacks change the meaning of a signal rather than its statistics. Stage 2 (a supervised classifier) and sequence or event-level context are needed for them.
3. **The threshold tuned on validation did not hold its FPR on test.** The test FPR is 2.33% against the 1.0% target. The excess comes from the last 20% (the test block) of three long ambient drives:

   | Ambient capture | Val FPR | Test FPR |
   |---|---|---|
   | dyno_drive_extended_short | 0.0094 | 0.1903 |
   | dyno_drive_extended_long | 0.0070 | 0.0500 |
   | highway_street_driving_long | 0.0081 | 0.0336 |
   | the other 9 captures | 0.0061–0.0110 | 0.0048–0.0133 |

   This is **temporal drift**: late parts of a drive reach vehicle states not seen earlier. A chronological split exposes this. A random split would have hidden it and reported an optimistic FPR. It motivates periodic threshold recalibration, and it is a documented threat to validity.
4. **At an FPR of 1–2.3%, precision is about 2%.** Because attacks are only 0.40% of test frames, even a 1% FPR produces about 50 false alarms per true alarm. This is why the hybrid design (Phase 4) adds a classifier and an Unknown/Suspicious state, and why alert aggregation (Phase 6) collapses bursts into events.
5. **Neither model detected the coolant-temperature attack** on a target ID never seen in training: 0/43 for the forest, and 13/43 for the baseline via its interval rule. Both counts are very small.
6. **Fuzzy results rest on 114 test frames** and should be presented with that count.

Raw tables: `road_phase3_comparison.csv`, `road_phase3_detection_by_attack.csv`,
`road_phase3_fpr_by_context.csv`, `road_phase3_test_metrics.json`.
