"""Sanity-check the derived ROAD frame labels against the capture metadata.

Run after ``build_dataset.py --dataset road``. Writes
``reports/results/road_label_check.csv`` and ``road_rule_collisions.csv``.

Checks
------
1. Every attack frame lies inside its capture's injection interval.
2. Masquerade consistency: the ``_masquerade`` version of a capture has the
   genuine (conflicting) target-ID frames removed, so
   ``rows(original) - rows(masquerade)`` must equal the number of target-ID frames
   inside the window that our rule labelled Normal in the original capture.
3. Rule collisions: how often each injection rule would fire on ambient
   (attack-free) traffic - shows whether a payload pattern alone is distinctive.
"""

from __future__ import annotations

import pandas as pd

from canids.ingestion.road_loader import (
    DEFAULT_METADATA_KEYS,
    load_metadata,
    parse_injection_spec,
    payload_matches,
)
from canids.utils.config import load_config, resolve_path


def main() -> int:
    """Run the checks and print/save the tables."""
    cfg = load_config("data")
    keys = {**DEFAULT_METADATA_KEYS, **cfg["road"].get("metadata_keys", {})}
    interim = resolve_path(cfg["paths"]["interim"]) / "road"
    results = resolve_path(cfg["paths"]["results"])
    meta = load_metadata(
        resolve_path(cfg["paths"]["raw_road"]) / "attacks" / "capture_metadata.json"
    )

    rows = []
    for name, entry in meta.items():
        parquet = interim / f"attacks__{name}.parquet"
        if entry.get(keys["injection_interval"]) is None or not parquet.exists():
            continue
        df = pd.read_parquet(parquet)
        elapsed = df["timestamp"] - df["timestamp"].min()
        spec = parse_injection_spec(entry, keys)
        start, end = spec.intervals[0]
        in_win = (elapsed >= start) & (elapsed <= end)
        is_attack = df["label"] != "Normal"
        target = in_win if spec.can_id is None else in_win & (df["can_id"] == spec.can_id)
        rows.append(
            {
                "capture": name,
                "total_frames": len(df),
                "attack_frames": int(is_attack.sum()),
                "attack_outside_window": int((is_attack & ~in_win).sum()),
                "window_start_s": start,
                "window_end_s": end,
                "target_frames_in_window": int(target.sum()),
                "unmatched_target_in_window": int((target & ~is_attack).sum()),
            }
        )
    check = pd.DataFrame(rows).set_index("capture")
    check["masquerade_rows_removed"] = pd.NA
    for name in check.index:
        masq = f"{name}_masquerade"
        if masq in check.index:
            check.loc[name, "masquerade_rows_removed"] = (
                check.loc[name, "total_frames"] - check.loc[masq, "total_frames"]
            )
    check.to_csv(results / "road_label_check.csv")

    ambient = pd.concat(
        [pd.read_parquet(p) for p in sorted(interim.glob("ambient__*.parquet"))], ignore_index=True
    )
    collisions, seen = [], set()
    for entry in meta.values():
        if entry.get(keys["injection_interval"]) is None:
            continue
        key = (entry[keys["injection_id"]], entry[keys["injection_data"]])
        if key in seen:
            continue
        seen.add(key)
        spec = parse_injection_spec(entry, keys)
        sub = ambient if spec.can_id is None else ambient[ambient["can_id"] == spec.can_id]
        hits = int(payload_matches(sub, spec.data_pattern).sum()) if spec.data_pattern else len(sub)
        collisions.append(
            {
                "injection_id": key[0],
                "injection_data": key[1],
                "ambient_frames_considered": len(sub),
                "ambient_matches": hits,
                "match_pct": round(100 * hits / max(len(sub), 1), 3),
            }
        )
    coll = pd.DataFrame(collisions)
    coll.to_csv(results / "road_rule_collisions.csv", index=False)

    pd.set_option("display.width", 250)
    print(check.to_string())
    print(f"\nAmbient frames: {len(ambient):,}")
    print(coll.to_string(index=False))
    bad = check[check["attack_outside_window"] > 0]
    return 1 if not bad.empty else 0


if __name__ == "__main__":
    raise SystemExit(main())
