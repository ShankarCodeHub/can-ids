"""canids — ML-based Intrusion Detection System for in-vehicle CAN bus networks.

A hybrid pipeline: an Isolation Forest anomaly detector (stage 1) followed by a
supervised attack classifier (stage 2) that labels traffic as Normal, DoS, Fuzzy,
Spoofing, or Unknown/Suspicious, with SHAP explanations and a Streamlit dashboard.

SAFETY: this is an academic, passive, defensive research prototype. It reads
recorded/replayed CAN traffic only (public datasets, vcan0 or python-can's virtual
bus). It never transmits frames to, or controls, a real vehicle.

Subpackages:
    ingestion      Module 1 - dataset loaders, replay, passive live reader
    preprocessing  Module 2 - cleaning, feature engineering, time-based splits
    models         Modules 3 & 4 - anomaly detector, classifier, hybrid logic, baseline
    explain        Module 5 - SHAP explanations
    alerting       Module 6 (backend) - Alert objects and SQLite event store
    evaluation     Module 7 - metrics, cross-dataset, ablation, runtime benchmarks
    pipeline       Streaming (real-time) inference loop
    utils          Config loading, logging, seeding
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
