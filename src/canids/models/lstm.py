"""OPTIONAL PyTorch LSTM sequence classifier - NOT IMPLEMENTED in this release.

Planned design (Phase 4, "only if time allows"): classify sliding windows of N
frames (per-frame feature vectors) with a small CPU-friendly LSTM, as a sequence
model that could capture masquerade/unseen attacks that frame-level trees miss
(see reports/phase7_summary.md). It was not attempted within the project
timeline. The tree-based classifier is the evaluated stage-2 model.

Requires requirements-dl.txt (torch) if implemented.
"""
