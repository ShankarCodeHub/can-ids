# 3-minute demo script (final presentation)

**Before the session (5 min early)**
- Run `streamlit run app/dashboard.py` and open http://localhost:8501.
- In **Event History → Danger zone**, delete old events so the counters start at zero.
- Set the browser zoom so all 5 KPI cards fit on one row.
- Keep `reports/RESULTS.md` open in a second tab as a fallback.
- Fallback if the UI fails: `python scripts/run_realtime.py --file data/demo/road_max_speedometer_attack_3.parquet --speed 5`

| Time | What to click | What to say |
|---|---|---|
| 0:00–0:20 | **About** tab | "CAN has no sender authentication, so any node can inject frames. We built a passive, explainable IDS. It never transmits: replay only runs on a virtual bus, and our edge runner refuses any bus that is not in listen-only mode." |
| 0:20–0:40 | **Live Monitor**: choose `demo/road_max_speedometer_attack_3.parquet`, speed **5×**, press **Start** | "This is a real attack recording from the ROAD dataset, from Oak Ridge. It's replayed onto a python-can virtual bus, and the original timestamps are kept, so the features match training at any speed." |
| 0:40–1:10 | Point at the frame-rate chart, the KPI cards, and **Status → ATTACK** | "Each frame goes through the same streaming feature code we trained on, then the Isolation Forest, XGBoost and our hybrid rules. We handle about 16,000 frames per second on this laptop. A 500 kbit/s bus carries at most about 4,500." |
| 1:10–1:35 | **Recent alert events** table and the ground-truth matrix | "Thousands of malicious frames become one Critical event on ID 0x0D0, because we aggregate bursts. On the right, the replayed ground truth shows hits and misses honestly." |
| 1:35–2:05 | **Alert Detail** tab: pick the 0x0D0 event | "Every alert is explained with SHAP. Here, the target ID suddenly makes up 8% of the traffic, which is the injection signature. SHAP shows attribution, not proof." |
| 2:05–2:35 | **Model Performance** tab: comparison table + confusion matrix | "On the held-out test set: Isolation Forest alone detects 12% of attack frames, at 2.3% false alarms. The hybrid detects 64% at 0.11% false alarms, macro-F1 0.886. All numbers come from saved result files." |
| 2:35–3:00 | Back to the table, or the ablation figure in RESULTS.md | "Our most important findings: timing features catch fabrication attacks almost perfectly, payload features are needed for masquerade attacks, and part of the payload signal was a vehicle-state artefact. SHAP and the error analysis revealed that. Limitations: one dataset, and unseen attacks are still missed. Thank you." |

**Likely questions, with one-line answers**
- *Why train the Isolation Forest on normal data only?* So it learns "normal" without being taught any attack, which is what lets it flag unseen ones.
- *Why does stage 1 add little?* At a 1% FPR budget, its false alarms (about 2.3% on test, with drift) outweigh its extra detections, and our ablation shows this.
- *How do you avoid leakage?* Chronological splits with purge gaps, whole held-out attack recordings, statistics fitted on training data only, and a logged single test evaluation.
- *Why HCRL is missing:* its files were not available on the evaluation machine. The cross-dataset code is ready and tested.
