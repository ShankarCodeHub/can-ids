# Phase 5: SHAP explanations (ROAD)

Sources: `reports/results/road_shap_global_importance.csv`, `road_explained_alerts.json`,
`road_shap_latency.json`, and the figures `reports/figures/road_shap_*.png` and `road_alert_*.png`.
Every explanation is computed on the **validation** split. The test split was not used.

> **SHAP shows model attribution, not causal proof.** A large SHAP value means the model relied
> on that feature for the decision. It does not show that the feature caused, or proves, an attack.

## Method

- **Model explained:** the stage-2 XGBoost classifier (the hybrid's classifier). Values are in **log-odds (margin) space**, per class.
- **Algorithm:** exact TreeSHAP, so SHAP values plus the base value reproduce the model margin (verified in `tests/test_shap_explainer.py`).
- **Backend.** The code prefers `shap.TreeExplainer`. On this laptop, **Windows Application Control blocks the numba DLL that `shap` imports** (`ImportError: An Application Control policy has blocked this file`). The explainer therefore falls back automatically to **XGBoost's built-in TreeSHAP** (`pred_contribs=True`), which implements the same exact algorithm. Additivity was checked on 300 ROAD validation frames, with a maximum error of 3.8×10⁻⁶. The OS security policy was left unchanged.
- **Global importance:** mean |SHAP| over a stratified validation sample of 6,350 frames: 3,000 Normal, 350 Fuzzy (all available) and 3,000 Spoofing.
- **Example alerts:** one random hybrid alert per available case, using seed 42.

## Global importance (top 10, mean |SHAP|)

| Feature | Normal | Fuzzy | Spoofing | Mean |
|---|---|---|---|---|
| b1 | 1.8152 | 0.2150 | 1.9342 | 1.3215 |
| b0 | 0.0496 | 2.9560 | 0.0238 | 1.0098 |
| win_cur_id_share | 0.0422 | 0.0000 | 1.1457 | 0.3960 |
| iat_id | 1.1008 | 0.0000 | 0.0859 | 0.3956 |
| b3 | 0.3388 | 0.0000 | 0.4245 | 0.2544 |
| b5 | 0.3345 | 0.0000 | 0.2141 | 0.1829 |
| b6 | 0.4259 | 0.0000 | 0.1084 | 0.1781 |
| iat_id_z | 0.0019 | 0.0000 | 0.1692 | 0.0570 |
| can_id | 0.1181 | 0.0000 | 0.0296 | 0.0492 |
| b4 | 0.1144 | 0.0133 | 0.0046 | 0.0441 |

Figures: `road_shap_global_bar.png`, `road_shap_beeswarm_fuzzy.png`, `road_shap_beeswarm_spoofing.png`.

## Example explained alerts (validation)

| # | Case | Frame | Hybrid decision | Top contributions (SHAP, log-odds) | Reason produced |
|---|---|---|---|---|---|
| 1 | Fuzzy (true positive) | fuzzing_attack_2, ID 0x49F | Fuzzy, p = 0.998, rule 2 | b0=0xFF (+3.665), b4=0xFF (+0.015), b1=0xFF (+0.003) | "payload byte 0 = 0xFF and payload byte 4 = 0xFF -> Fuzzy-like" |
| 2 | Spoofing, fabrication (true positive) | reverse_light_off_attack_2, ID 0x0D0 | Spoofing, p = 0.998, rule 2 | win_cur_id_share=0.08 (+1.984), b1=0x71 (+0.791), iat_id_z=-0.03 (+0.565), iat_id=0.001 s (+0.277) | "this ID is 0.08 of the last 100 frames and payload byte 1 = 0x71 -> Spoofing-like" |
| 3 | Spoofing, masquerade (true positive) | reverse_light_off_attack_2_masquerade, ID 0x0D0 | Spoofing, p = 0.993, rule 2 | win_cur_id_share=0.04 (+2.268), b1=0x71 (+0.838), iat_id_z (−0.284), b2=0x04 (+0.261) | "this ID is 0.04 of the last 100 frames and payload byte 1 = 0x71 -> Spoofing-like" |
| 4 | False alarm (Normal predicted Spoofing) | ambient_highway_street_driving_long, ID 0x0D0 | Spoofing, p = 0.994, rule 2 | win_cur_id_share=0.05 (+2.207), b1=0x73 (+0.820), b2=0x04 (+0.261), iat_id (−0.201) | "this ID is 0.05 of the last 100 frames and payload byte 1 = 0x73 -> Spoofing-like" |
| — | DoS | — | — | — | Not available: **ROAD has no DoS class**. A DoS example will be produced from HCRL. |

Figures: `road_alert_1_fuzzy.png`, `road_alert_2_spoofing_fabrication.png`,
`road_alert_3_spoofing_masquerade.png`, `road_alert_4_false_alarm.png`.

## Latency (200 single-frame explanations of validation alerts)

| p50 | p95 | p99 | max | Budget | Policy |
|---|---|---|---|---|---|
| 9.97 ms | 11.05 ms | 30.04 ms | 145.27 ms | 50 ms | p95 within budget, so every alert is explained. The rare outliers argue for computing explanations off the critical path (a background thread in Phase 6). |

Each timing covers one TreeSHAP call plus one `predict_proba` call on a single row, on an AMD Ryzen 5 7520U laptop.

## Findings

1. **Fuzzy detection on ROAD is driven almost entirely by `b0 = 0xFF`.** ROAD's fuzzer injects all-`FF` payloads, and the ROAD labels are derived from that same pattern (Phase 1). This is a genuine property of the attack in this dataset, but the model has essentially learned the labelling rule. **It should not be expected to generalise to fuzzers with random payloads**, such as HCRL's. This is a threat to validity for the Fuzzy results, and Phase 7 tests it (cross-dataset, and the no-payload ablation).
2. **Spoofing uses one sensible signal and one suspicious one.**
   - **Sensible:** `win_cur_id_share`. During injection, the target ID's share of recent traffic rises. Note that it still fires for the masquerade example (share 0.04), so the model is not using rate alone.
   - **Suspicious:** `b1` around 0x71–0x73 on ID 0x0D0 adds about +0.8 in both the attack and the false-alarm explanations. Byte 1 of 0x0D0 most likely encodes a vehicle state, which coincides with conditions during the attack recordings. That is a **spurious, recording-specific correlation**.
3. **SHAP explains the false alarm.** The ambient highway frame got almost the same explanation as a real reverse-light attack (same ID, similar share, `b1` = 0x73). This is the kind of error analysis SHAP is meant to enable, and it points to the fix: drop or regularise the vehicle-state payload bytes, or add sequence context. The Phase 7 ablation "without payload bytes" will quantify the effect.
4. **Timing and window features matter mainly for Spoofing.** For Fuzzy, payload content dominates. HCRL's DoS attack is expected to be explained by window features such as `win_id0_share` and `win_rate_fps`; this has not been verified yet, because the HCRL data is not available.
5. **Explanations are fast enough for live use.** The p95 of 11 ms is far below the 50 ms budget, so the dashboard can show an explanation for every alert.
