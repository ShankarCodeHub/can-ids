"""Phase 5: SHAP explanations - global importance, example alerts, latency.

Uses the VALIDATION split (the test split stays reserved for evaluate.py).

Outputs
-------
reports/results/<ds>_shap_global_importance.csv   mean |SHAP| per feature x class
reports/results/<ds>_explained_alerts.json        example alerts: top-5 features + reason
reports/results/<ds>_shap_latency.json            single-explanation latency profile
reports/figures/<ds>_shap_global_bar.png
reports/figures/<ds>_shap_beeswarm_<class>.png
reports/figures/<ds>_alert_<n>_<name>.png

Usage:
    python scripts/explain.py --dataset road
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from canids.evaluation.metrics import save_json
from canids.explain.shap_explainer import (
    CAUSALITY_NOTE,
    AlertExplainer,
    plot_beeswarm,
    plot_explanation,
    plot_global_bar,
)
from canids.ingestion.schema import LABEL_NORMAL
from canids.models.hybrid import RULE_NAMES, HybridDetector
from canids.preprocessing.datasets import load_split
from canids.preprocessing.features import FEATURE_COLUMNS
from canids.utils.config import load_config, resolve_path
from canids.utils.logger import get_logger
from canids.utils.provenance import git_commit
from canids.utils.seed import set_seed

log = get_logger("scripts.explain")


def stratified_sample(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    """Up to ``per_class`` rows of each label (all rows of smaller classes)."""
    parts = [
        g.sample(min(len(g), per_class), random_state=seed)
        for _, g in df.groupby("label", observed=True)
    ]
    return pd.concat(parts).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", choices=["hcrl", "road"], required=True)
    ap.add_argument("--normal-sample", type=int, default=20000)
    args = ap.parse_args(argv)

    mcfg, dcfg = load_config("model"), load_config("data")
    seed = int(mcfg["seed"])
    set_seed(seed)
    ds = args.dataset
    art = resolve_path(dcfg["paths"]["artifacts"]) / ds
    results = resolve_path(dcfg["paths"]["results"])
    figures = resolve_path(dcfg["paths"]["figures"])
    top_k = int(mcfg["shap"]["top_k"])
    budget_ms = float(mcfg["shap"]["max_latency_ms"])

    hybrid = HybridDetector.load(art)
    explainer = AlertExplainer(hybrid.classifier, top_k=top_k)
    log.info("SHAP backend: %s", explainer.backend_name)

    val = load_split(
        ds,
        "val",
        [*FEATURE_COLUMNS, "attack_type", "source_file", "timestamp"],
        max_normal=args.normal_sample,
        seed=seed,
    )
    decisions = hybrid.predict(val)
    val = pd.concat([val.reset_index(drop=True), decisions], axis=1)
    val["label"] = val["label"].astype(str)

    # ---------------- global importance
    sample = stratified_sample(val, 3000, seed)
    table, values = explainer.global_importance(sample)
    table.to_csv(results / f"{ds}_shap_global_importance.csv", index_label="feature")
    plot_global_bar(
        table,
        figures / f"{ds}_shap_global_bar.png",
        f"{ds.upper()}: global feature importance (validation sample, n={len(sample):,})",
    )
    for k, cls in enumerate(hybrid.classifier.classes):
        if cls == LABEL_NORMAL:
            continue
        plot_beeswarm(
            values[:, k, :],
            sample,
            explainer.features,
            figures / f"{ds}_shap_beeswarm_{cls.lower()}.png",
            f"{ds.upper()}: SHAP summary for class '{cls}' (validation sample)",
        )

    # ---------------- example alerts (one per attack type available + one false alarm)
    rng = np.random.default_rng(seed)
    is_masq = val["attack_type"].astype(str).str.endswith("_masquerade")
    pred = val["predicted_class"].astype(str)
    cases = {
        "fuzzy": (val["label"] == "Fuzzy") & (pred == "Fuzzy"),
        "spoofing_fabrication": (val["label"] == "Spoofing") & (pred == "Spoofing") & ~is_masq,
        "spoofing_masquerade": (val["label"] == "Spoofing") & (pred == "Spoofing") & is_masq,
        "dos": (val["label"] == "DoS") & (pred == "DoS"),
        "false_alarm": (val["label"] == LABEL_NORMAL) & pred.isin(["Fuzzy", "Spoofing", "DoS"]),
    }
    examples, n = [], 0
    for name, mask in cases.items():
        idx = np.flatnonzero(mask.to_numpy())
        if len(idx) == 0:
            reason = "dataset has no DoS class" if name == "dos" else "no matching validation alert"
            examples.append({"case": name, "available": False, "reason": reason})
            log.warning("example %s not available: %s", name, reason)
            continue
        i = int(rng.choice(idx))
        row = val.iloc[i]
        exp = explainer.explain(row)
        n += 1
        fig = figures / f"{ds}_alert_{n}_{name}.png"
        plot_explanation(
            exp,
            fig,
            f"{ds.upper()} alert #{n}: {name.replace('_', ' ')} "
            f"(true: {row['label']}, {row['attack_type']})",
        )
        examples.append(
            {
                "case": name,
                "available": True,
                "figure": str(fig.relative_to(resolve_path("."))),
                "frame": {
                    "source_file": str(row["source_file"]),
                    "timestamp": float(row["timestamp"]),
                    "can_id": f"0x{int(row['can_id']):03X}",
                    "true_label": row["label"],
                    "attack_type": str(row["attack_type"]),
                },
                "hybrid": {
                    "predicted_class": row["predicted_class"],
                    "rule": RULE_NAMES[int(row["rule"])],
                    "is_anomaly": bool(row["is_anomaly"]),
                    "anomaly_score": float(row["anomaly_score"]),
                },
                "explanation": exp.to_dict(),
            }
        )
        log.info("alert %d (%s): %s", n, name, exp.reason)

    # ---------------- latency (explanations computed one at a time, as for live alerts)
    alerts = val[pred != LABEL_NORMAL]
    lat = explainer.latency_profile(alerts if len(alerts) else val, n=200)
    lat["budget_ms"] = budget_ms
    lat["within_budget_p95"] = lat["p95_ms"] <= budget_ms
    lat["policy"] = (
        "explain every alert (p95 within budget)"
        if lat["within_budget_p95"]
        else "explain alerts only, asynchronously (p95 over budget)"
    )
    save_json(lat, results / f"{ds}_shap_latency.json")
    save_json(
        {
            "dataset": ds,
            "split": "val",
            "backend": explainer.backend_name,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "note": CAUSALITY_NOTE,
            "examples": examples,
        },
        results / f"{ds}_explained_alerts.json",
    )

    pd.set_option("display.width", 200)
    print(f"\nSHAP backend: {explainer.backend_name}")
    print(f"\nTop 10 features by mean |SHAP| ({len(sample):,} validation frames):")
    print(table.head(10).round(4).to_string())
    print("\nExample alerts:")
    for e in examples:
        if e["available"]:
            x = e["explanation"]
            print(
                f"  [{e['case']}] true={e['frame']['true_label']} pred={x['predicted_class']} "
                f"p={x['probability']:.3f}: {x['reason']}"
            )
        else:
            print(f"  [{e['case']}] not available - {e['reason']}")
    print(
        f"\nLatency per explanation: p50={lat['p50_ms']:.2f} ms  p95={lat['p95_ms']:.2f} ms  "
        f"p99={lat['p99_ms']:.2f} ms  max={lat['max_ms']:.2f} ms  (budget {budget_ms} ms) -> "
        f"{lat['policy']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
