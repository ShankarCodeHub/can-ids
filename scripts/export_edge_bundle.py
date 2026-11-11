"""Export a self-contained model bundle for an edge node (e.g. Raspberry Pi 4).

Copies exactly what the runtime needs: Isolation Forest, XGBoost classifier,
hybrid thresholds and per-ID feature statistics. It also writes the XGBoost
model in its native, version-portable UBJSON format, and a manifest with
SHA256 checksums and the library versions the bundle was built with. The Pi
must use the same pinned versions (requirements-pi.txt), because joblib
pickles are version-sensitive.

Usage:
    python scripts/export_edge_bundle.py --dataset road
    # then copy deploy/raspberry_pi/bundle/road/ to the Pi
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost

from canids.utils.config import load_config, resolve_path
from canids.utils.provenance import git_commit

PARTS = {
    "anomaly": ["isolation_forest.joblib", "isolation_forest_metadata.json"],
    "classifier": ["classifier.joblib", "classifier_metadata.json"],
    "hybrid": ["hybrid_metadata.json"],
    "feature_artifacts": ["per_id_stats.json", "feature_metadata.json"],
}


def sha256(path: Path) -> str:
    """SHA256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", default="road")
    ap.add_argument("--out", default="deploy/raspberry_pi/bundle")
    args = ap.parse_args(argv)
    src = resolve_path(load_config("data")["paths"]["artifacts"]) / args.dataset
    dst = resolve_path(args.out) / args.dataset
    if dst.exists():
        shutil.rmtree(dst)
    files = {}
    for sub, names in PARTS.items():
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(src / sub / name, dst / sub / name)
            files[f"{sub}/{name}"] = None
    # portable XGBoost copy (loadable by any xgboost >= 1.6 without unpickling)
    model = joblib.load(src / "classifier" / "classifier.joblib")
    model.get_booster().save_model(str(dst / "classifier" / "classifier.ubj"))
    files["classifier/classifier.ubj"] = None
    for rel in files:
        p = dst / rel
        files[rel] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    manifest = {
        "dataset": args.dataset,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "built_with": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
        "total_bytes": sum(f["bytes"] for f in files.values()),
        "files": files,
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"bundle written to {dst} ({manifest['total_bytes'] / 2**20:.2f} MiB, "
        f"{len(files)} files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
