"""Phase 7 error analysis of the FROZEN hybrid on the test split (analysis only, no tuning).

Collects every false negative (attack -> Normal), every false positive
(Normal -> alarm) and every true positive, then looks for patterns:

* FN by attack type x deciding rule, and by position inside the injection window
* TP vs FN feature medians per attack type (what do missed frames look like?)
* FP by capture, by CAN ID, by predicted class, and by context:
  cold start (first frame of an ID), the drift captures, and frames inside the
  injection window of attack captures ("collateral" normal frames)
* possible label noise: Normal-labelled frames with the exact fuzzing payload

Outputs: reports/results/<ds>_error_analysis.json and <ds>_errors_*.csv
Usage:   python scripts/error_analysis.py --dataset road
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from canids.evaluation.metrics import save_json
from canids.ingestion.road_loader import DEFAULT_METADATA_KEYS, load_metadata, parse_injection_spec
from canids.ingestion.schema import BYTE_COLS, LABEL_NORMAL
from canids.models.hybrid import RULE_NAMES, HybridDetector
from canids.preprocessing.datasets import split_dir
from canids.preprocessing.features import FEATURE_COLUMNS
from canids.preprocessing.splits import capture_name
from canids.utils.config import load_config, resolve_path
from canids.utils.provenance import git_commit

DRIFT_CAPTURES = (
    "ambient_dyno_drive_extended_short",
    "ambient_dyno_drive_extended_long",
    "ambient_highway_street_driving_long",
)  # identified in Phase 3


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", choices=["hcrl", "road"], required=True)
    args = ap.parse_args(argv)
    ds = args.dataset
    dcfg = load_config("data")
    art = resolve_path(dcfg["paths"]["artifacts"]) / ds
    results = resolve_path(dcfg["paths"]["results"])
    hybrid = HybridDetector.load(art)
    with (results / "test_access_log.csv").open("a", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerow(
            [
                datetime.now(UTC).isoformat(timespec="seconds"),
                ds,
                "phase7_error_analysis",
                git_commit(),
            ]
        )

    windows: dict[str, tuple[float, float]] = {}
    if ds == "road":
        meta = load_metadata(
            resolve_path(dcfg["paths"]["raw_road"]) / "attacks" / "capture_metadata.json"
        )
        for name, entry in meta.items():
            with contextlib.suppress(ValueError):  # accelerator captures have no interval
                windows[name] = parse_injection_spec(entry, DEFAULT_METADATA_KEYS).intervals[0]

    keep_parts, normal_per_file = [], {}
    for f in sorted(split_dir(ds, "test").glob("*.parquet")):
        df = pq.read_table(
            f, columns=[*FEATURE_COLUMNS, "timestamp", "label", "attack_type", "source_file"]
        ).to_pandas()
        dec = hybrid.predict(df)
        df["pred"] = dec["predicted_class"].astype(str).to_numpy()
        df["rule"] = dec["rule"].map(RULE_NAMES).to_numpy()
        df["class_probability"] = dec["class_probability"].to_numpy()
        df["label"] = df["label"].astype(str)
        cap = capture_name(str(df["source_file"].iloc[0]))
        df["capture"] = cap
        df["elapsed_s"] = df["timestamp"] - df["timestamp"].min()
        a, b = windows.get(cap, (np.nan, np.nan))
        df["in_injection_window"] = (df["elapsed_s"] >= a) & (df["elapsed_s"] <= b)
        df["since_injection_start_s"] = df["elapsed_s"] - a
        attack, alarm = df["label"] != LABEL_NORMAL, df["pred"] != LABEL_NORMAL
        df["outcome"] = np.select(
            [attack & alarm, attack & ~alarm, ~attack & alarm], ["TP", "FN", "FP"], "TN"
        )
        normal_per_file[cap] = int((~attack).sum())
        keep_parts.append(df[df["outcome"] != "TN"])
    e = pd.concat(keep_parts, ignore_index=True)
    fn, fp, tp = e[e.outcome == "FN"], e[e.outcome == "FP"], e[e.outcome == "TP"]
    out: dict = {"dataset": ds, "tp": len(tp), "fn": len(fn), "fp": len(fp)}

    # ---------------- false negatives
    fn_tab = (
        e[e.label != LABEL_NORMAL].groupby(["attack_type", "outcome"]).size().unstack(fill_value=0)
    )
    fn_tab["miss_rate"] = fn_tab.get("FN", 0) / fn_tab.sum(axis=1)
    fn_tab.to_csv(results / f"{ds}_errors_fn_by_attack.csv")
    fn_rule = fn.groupby(["attack_type", "rule"]).size().unstack(fill_value=0)
    fn_rule.to_csv(results / f"{ds}_errors_fn_by_rule.csv")
    att = e[e.label != LABEL_NORMAL].copy()
    att["first_second_of_injection"] = att["since_injection_start_s"] < 1.0
    pos = att.groupby(["first_second_of_injection", "outcome"]).size().unstack(fill_value=0)
    pos["miss_rate"] = pos.get("FN", 0) / pos.sum(axis=1)
    out["fn_by_injection_position"] = pos.reset_index().to_dict(orient="records")

    key = [
        "win_cur_id_share",
        "iat_id",
        "iat_id_z",
        "hamming_prev_id",
        "payload_entropy",
        "win_rate_fps",
        "b1",
        "b2",
        "b5",
    ]
    med = e[e.outcome.isin(["TP", "FN"])].groupby(["attack_type", "outcome"])[key].median().round(4)
    med.to_csv(results / f"{ds}_errors_tp_vs_fn_medians.csv")

    # ---------------- false positives
    fp_file = fp.groupby("capture").size().rename("false_positives").to_frame()
    fp_file["normal_frames"] = fp_file.index.map(normal_per_file)
    fp_file["fpr"] = fp_file["false_positives"] / fp_file["normal_frames"]
    fp_file = fp_file.sort_values("false_positives", ascending=False)
    fp_file.to_csv(results / f"{ds}_errors_fp_by_capture.csv")
    fp_id = (
        fp.assign(can_id_hex=fp["can_id"].map(lambda v: f"0x{int(v):03X}"))
        .groupby("can_id_hex")
        .size()
        .nlargest(10)
    )
    fp_id.to_csv(results / f"{ds}_errors_fp_by_canid.csv")
    is_ff = (fp[BYTE_COLS] == 255).all(axis=1)
    out["fp_detail"] = {
        "by_predicted_class": fp["pred"].value_counts().to_dict(),
        "by_rule": fp["rule"].value_counts().to_dict(),
        "cold_start_first_frame_of_id": int((fp["iat_id"] == -1).sum()),
        "in_drift_captures": int(fp["capture"].isin(DRIFT_CAPTURES).sum()),
        "in_attack_captures": int(fp["capture"].str.contains("attack").sum()),
        "in_attack_captures_inside_injection_window": int(fp["in_injection_window"].sum()),
        "possible_label_noise_all_FF_payload": int(is_ff.sum()),
        "top_can_ids": fp_id.to_dict(),
    }
    out["fn_detail"] = {
        "by_rule": fn["rule"].value_counts().to_dict(),
        "by_predicted": fn["pred"].value_counts().to_dict(),
    }
    save_json(out, results / f"{ds}_error_analysis.json")

    pd.set_option("display.width", 220)
    print(f"\n=== {ds.upper()} error analysis (frozen hybrid, test) ===")
    print(f"TP={len(tp):,}  FN={len(fn):,}  FP={len(fp):,}")
    print("\nMisses by attack type:")
    print(fn_tab.round(4).to_string())
    print("\nMisses by deciding rule:")
    print(fn_rule.to_string())
    print("\nMiss rate by position in the injection window:")
    print(pos.round(4).to_string())
    print("\nTP vs FN medians (key features):")
    print(med.to_string())
    print("\nFalse positives by capture (top 8):")
    print(fp_file.head(8).round(5).to_string())
    print("\nFalse-positive context:")
    for k, v in out["fp_detail"].items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
