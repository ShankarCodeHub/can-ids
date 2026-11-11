# Phase 4: Attack classifier and hybrid pipeline (ROAD)

All numbers come from `reports/results/road_phase4_*.csv|json` (test),
`road_classifier_tuning.csv`, `road_hybrid_tuning.csv`, and the model metadata JSONs (validation).
HCRL has not been evaluated yet because its raw files are not on this machine. ROAD has no DoS
class, so the learned classes are **Normal, Fuzzy and Spoofing**.

## Protocol

- **Classifier training data:** all 14,977 training attack frames (14,386 Spoofing and 591 Fuzzy), plus a seeded sample of 299,771 Normal frames, with balanced sample weights.
- **Search:** 10 seeded random settings each for RandomForest and XGBoost, on the fixed validation split (all 23,026 attack frames and 300,000 Normal frames). Normal validation frames are weighted ×20.38, so the metrics reflect the true class mix. The best model is selected by validation macro-F1.
- **Hybrid thresholds:** `t_attack` and `t_suppress` are grid-searched on validation to maximise macro-F1, subject to an alarm FPR of at most 1% (Unknown counts as an alarm). `t_normal = 0.9` only sets the low-confidence flag.
- **Single test run:** the test set was scored once, after every model and threshold was frozen. The run is logged in `test_access_log.csv`.
- **The optional LSTM was not attempted.**

## Validation (model selection)

| Item | Result |
|---|---|
| Best RandomForest | macro-F1 0.8306 (max_features 0.5, max_depth 20) |
| **Selected: XGBoost** | 200 trees, max_depth 8, learning rate 0.03, subsample 1.0, colsample 1.0 → **macro-F1 0.8411** (F1: Normal 0.9988, Fuzzy 0.9717, Spoofing 0.5527) |
| Calibration (ECE, top-class) | 0.0003, so no calibration was applied |
| Hybrid (tuned) | t_attack 0.6, t_suppress 0.5 → macro-F1 0.8421, detection 0.4051, alarm FPR 0.0001 |
| Hybrid (no overrule) | t_attack 0.6, rule 3 disabled → macro-F1 0.8405, detection 0.4522, alarm FPR 0.0101 |

## Held-out test results (6,121,249 frames, 24,510 attack frames)

| System | Alarm precision | Detection recall | Binary F1 | Alarm FPR | ROC-AUC | PR-AUC | Macro-F1 | Weighted F1 |
|---|---|---|---|---|---|---|---|---|
| Baseline (rules) | 0.0184 | 0.0325 | 0.0235 | 0.0070 | 0.6758 | 0.0076 | n/a | n/a |
| Isolation Forest only | 0.0207 | 0.1224 | 0.0354 | 0.0233 | 0.8396 | 0.0205 | n/a | n/a |
| Classifier only (XGBoost) | **0.7005** | **0.6677** | **0.6837** | **0.0011** | **0.9038** | **0.6997** | 0.8810 | **0.9975** |
| **Hybrid (tuned)** | 0.6992 | 0.6421 | 0.6695 | **0.0011** | n/a | n/a | **0.8862** | 0.9974 |
| Hybrid (no overrule) | 0.0977 | 0.6567 | 0.1701 | 0.0244 | n/a | n/a | 0.8823 | 0.9857 |

Unknown/Suspicious outputs on test:

| System | Attack frames labelled Unknown | Normal frames labelled Unknown |
|---|---|---|
| Hybrid (tuned) | 0 | 54 |
| Hybrid (no overrule) | 358 | 141,951 |

### Per-class results (test)

| Class | Support | Classifier only (P / R / F1) | Hybrid tuned (P / R / F1) |
|---|---|---|---|
| Normal | 6,096,739 | 0.9987 / 0.9989 / 0.9988 | 0.9986 / 0.9989 / 0.9987 |
| Fuzzy | 114 | 0.9268 / 1.0000 / 0.9620 | 0.9828 / 1.0000 / 0.9913 |
| Spoofing | 24,396 | 0.6993 / 0.6661 / 0.6823 | 0.6995 / 0.6404 / 0.6686 |

Confusion matrices: `reports/figures/road_phase4_cm_clf.png`, `road_phase4_cm_hy.png`, `road_phase4_cm_hy4.png`.

### Detection rate by attack type (test)

| Attack type | Frames | IF only (Phase 3) | Classifier only | Hybrid tuned | Hybrid no overrule |
|---|---|---|---|---|---|
| fuzzing | 114 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| correlated_signal | 1,265 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| correlated_signal_masquerade | 1,263 | 0.1211 | 1.0000 | 1.0000 | 1.0000 |
| max_speedometer | 6,107 | 0.2335 | 0.7200 | 0.7200 | 0.7785 |
| max_speedometer_masquerade | 6,107 | 0.0057 | 0.7193 | 0.7193 | 0.7195 |
| reverse_light | 4,784 | 0.0013 | 0.6242 | 0.5171 | 0.5171 |
| reverse_light_masquerade | 4,784 | 0.0000 | 0.4070 | 0.3829 | 0.3829 |
| max_engine_coolant_temp (unseen target) | 43 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| max_engine_coolant_temp_masquerade (unseen) | 43 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Which rule decided each test frame (hybrid tuned)

| Rule | Attack frames | Normal frames |
|---|---|---|
| 1 normal_confident | 7,299 | 5,939,513 |
| 2 attack_classified | 15,738 | 6,715 |
| 3 anomaly_overruled | 358 | 141,897 |
| 4 unknown_suspicious | 0 | 54 |
| 5 normal_low_confidence | 1,115 | 8,560 |

## Findings

1. **The supervised stage is what makes the system usable.** Against Isolation Forest alone, detection recall rises from 0.122 to 0.668, and the alarm FPR falls from 0.0233 to 0.0011, about 20× fewer false alarms. Alarm precision rises from 0.02 to 0.70.
2. **Masquerade attacks, invisible to stage 1, are largely caught by stage 2.** Correlated-signal masquerade goes from 0.12 to 1.00 and max-speedometer masquerade from 0.006 to 0.72. The classifier learns attack payload/ID patterns that do not look statistically unusual.
3. **The tuned hybrid behaves almost like the classifier alone.**
   - Validation tuning chose `t_suppress = 0.5`: whenever the classifier says P(Normal) ≥ 0.5, it overrules stage 1. Isolation Forest's false alarms (2.3% of test frames) cost far more macro-F1 than its few extra detections gained.
   - Result on test: macro-F1 0.8862 against 0.8810 (better Fuzzy precision), but detection recall 0.642 against 0.668. This is mainly because a lower `t_attack` was chosen on validation and changed some reverse-light frames.
   - **On this data, the hybrid does not beat the classifier alone in detection. It wins slightly on macro-F1.** This is reported as found.
4. **The Unknown/Suspicious path, as designed, costs too much here.** Without rule 3 the hybrid labels 358 attack frames Unknown (all max-speedometer), but also 141,951 Normal frames (alarm FPR 0.0244). That is the Isolation Forest's false-alarm rate passed straight through, including the temporal drift found in Phase 3. The design idea is sound, but its usefulness is limited by stage 1's FPR on this dataset.
5. **The unseen-target test failed for every system.** The coolant-temperature attack targets ID `0x4E7`, which no training attack used. 0/43 frames were detected, by stage 1 or stage 2, in both versions. Validation tuning cannot reward detecting unseen attacks, because validation has none. This is a real limitation of frame-level detection with these features, and it motivates sequence/event-level models (future work).
6. **Spoofing remains the weak class** (test F1 0.68). The missed frames are mostly reverse-light and max-speedometer injections whose payload values also occur in normal traffic (see Phase 1 `road_rule_collisions.csv`).
7. **Validation and test agree on the classifier** (macro-F1 0.841 against 0.881), so model selection did not overfit validation. Test Spoofing F1 (0.682) is in fact higher than validation (0.553), probably because the `_3` recordings differ from the `_2` ones. Per-class results vary by recording, which is a threat to validity when only 3 recordings exist per attack.
8. **Fuzzy results rest on 114 test frames** (and 350 validation frames), so the precision differences between systems (0.93 against 0.98) amount to a handful of frames.

## Artifacts

`models/road/classifier/` (XGBoost + metadata), `models/road/hybrid/hybrid_metadata.json`
(thresholds, rule definitions, validation results for both versions, provenance).
