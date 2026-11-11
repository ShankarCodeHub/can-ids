"""Create small ROAD demo logs in data/demo/ (for Docker, the Pi and quick demos).

ROAD is licensed CC BY 4.0, which permits redistribution with attribution; the
attribution is written to data/demo/README.md. HCRL data is NOT redistributed.
The samples are cut from the unified interim parquet files, keeping their labels so
that the dashboard can show the replayed ground truth. They are for demonstration
only; all evaluation numbers come from the full splits.

Usage:
    python scripts/make_demo_sample.py
"""

from __future__ import annotations

import pandas as pd

from canids.utils.config import load_config, resolve_path

SAMPLES = {
    # output name: (interim file, max frames)
    "road_fuzzing_attack_3.parquet": ("attacks__fuzzing_attack_3.parquet", None),
    "road_max_speedometer_attack_3.parquet": ("attacks__max_speedometer_attack_3.parquet", None),
    "road_reverse_light_on_attack_3_masquerade.parquet": (
        "attacks__reverse_light_on_attack_3_masquerade.parquet",
        None,
    ),
    "road_ambient_basic_short_60k.parquet": (
        "ambient__ambient_dyno_drive_basic_short.parquet",
        60_000,
    ),
}

README = """# Demo CAN logs (ROAD sample)

Small excerpts of the ROAD dataset in the project's unified frame schema, used by the
dashboard demo, Docker image and edge runner. Labels are included only to display the
replayed ground truth; they are not used for any evaluation.

| File | Frames | Content |
|---|---|---|
{rows}

**Source and licence.** ROAD: Real ORNL Automotive Dynamometer CAN Intrusion Dataset,
Oak Ridge National Laboratory, https://doi.org/10.5281/zenodo.10462796, licensed under
CC BY 4.0. Cite: M. E. Verma et al., "A comprehensive guide to CAN IDS data and
introduction of the ROAD dataset," PLoS ONE 19(1): e0296879, 2024.
Files were converted to the unified schema (scripts/build_dataset.py) and truncated.
"""


def main() -> int:
    """Write the demo files and README."""
    data = load_config("data")
    interim = resolve_path(data["paths"]["interim"]) / "road"
    out = resolve_path("data/demo")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, (src, n) in SAMPLES.items():
        df = pd.read_parquet(interim / src)
        if n:
            df = df.iloc[:n]
        df.to_parquet(out / name, index=False, compression="zstd")
        counts = df["label"].astype(str).value_counts().to_dict()
        rows.append(f"| `{name}` | {len(df):,} | {counts} |")
        print(
            f"{name}: {len(df):,} frames {counts} ({(out / name).stat().st_size / 2**20:.2f} MiB)"
        )
    (out / "README.md").write_text(README.format(rows="\n".join(rows)), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
