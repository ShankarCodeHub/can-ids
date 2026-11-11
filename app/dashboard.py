"""CAN Bus IDS dashboard (Streamlit + Plotly) - Module 6 UI.

Run:  streamlit run app/dashboard.py

Tabs: Live Monitor | Attack Analytics | Alert Detail | Event History |
      Model Performance | System Health | About

Design: one cached ``MonitorSession`` (models loaded once) runs replay, inference
and SHAP in background threads. The UI only reads snapshots and the SQLite
store, so it never blocks. Live widgets refresh themselves with
``st.fragment(run_every=...)``.
Replay goes to python-can's in-process 'virtual' bus (Windows/Linux) or, when
``CANIDS_REPLAY_INTERFACE=socketcan`` is set (scripts/replay_demo.sh), to vcan0.
"""

from __future__ import annotations

import os
import platform
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psutil
import streamlit as st

from canids import __version__
from canids.alerting.alert import DEFAULT_SEVERITY, SEVERITIES
from canids.alerting.store import AlertStore
from canids.explain.shap_explainer import CAUSALITY_NOTE
from canids.pipeline.realtime import MonitorSession
from canids.utils.config import load_config, resolve_path

APP = load_config("app")
DATA = load_config("data")
DASH = APP["dashboard"]
DATASET = DASH.get("model_dataset", "road")
REFRESH = timedelta(seconds=float(DASH.get("refresh_interval_s", 1.0)))
STATUS_TEXT = {
    "green": ":green[● NORMAL]",
    "amber": ":orange[● SUSPICIOUS]",
    "red": ":red[● ATTACK]",
}
SEV_COLORS = {"Low": "#9e9e9e", "Medium": "#ff9800", "High": "#f44336", "Critical": "#880e4f"}

st.set_page_config(page_title=DASH.get("title", "CAN Bus IDS"), layout="wide")


# =========================================================================== resources
@st.cache_resource
def get_store() -> AlertStore:
    """Shared SQLite store."""
    return AlertStore(resolve_path(APP["alerting"]["db_path"]))


@st.cache_resource
def get_session() -> MonitorSession:
    """Shared monitoring engine (loads models once per server process)."""
    return MonitorSession(
        resolve_path(DATA["paths"]["artifacts"]) / DATASET,
        get_store(),
        window_frames=int(DATA["windows"]["frame_count"]),
        window_ms=float(DATA["windows"]["time_ms"]),
        aggregation_ms=float(APP["alerting"]["aggregation_window_ms"]),
        severity_base=APP["alerting"].get("severity", DEFAULT_SEVERITY),
        batch_size=int(APP["pipeline"]["batch_size"]),
        max_wait_ms=float(APP["pipeline"]["max_wait_ms"]),
        explain=True,
        top_k=int(load_config("model")["shap"]["top_k"]),
    )


def replay_files() -> list[Path]:
    """Files offered for replay (unified parquet / candump logs)."""
    files: list[Path] = []
    for d in DASH.get("replay_dirs", []):
        p = resolve_path(d)
        if p.exists():
            files += sorted([*p.glob("*.parquet"), *p.glob("*.log")])
    return files


def hex_id(v: int) -> str:
    """0x-formatted CAN ID."""
    return f"0x{int(v):03X}"


def events_frame(**filters) -> pd.DataFrame:
    """Alert events with a relative-time column and hex IDs."""
    df = get_store().query(**filters)
    if not df.empty:
        # relative to each session's first alert: capture clocks of different logs are unrelated
        df["t_rel_s"] = df["timestamp"] - df.groupby("session")["timestamp"].transform("min")
        df["CAN ID"] = df["can_id"].map(hex_id)
    return df


# =========================================================================== header
st.title(DASH.get("title", "CAN Bus IDS"))
st.caption(
    f"canids v{__version__} · models: `models/{DATASET}` · passive, defensive research prototype"
)
tabs = st.tabs(
    [
        "Live Monitor",
        "Attack Analytics",
        "Alert Detail",
        "Event History",
        "Model Performance",
        "System Health",
        "About",
    ]
)

# =========================================================================== a) live monitor
with tabs[0]:
    session = get_session()
    files = replay_files()
    c1, c2, c3, c4 = st.columns([4, 1.2, 1.4, 1.4])
    if not files:
        st.warning("No replay files found. Run scripts/build_dataset.py first.")
    else:
        names = [f"{f.parent.name}/{f.name}" for f in files]
        default = next((i for i, n in enumerate(names) if "fuzzing_attack_3" in n), 0)
        choice = c1.selectbox("Log file to replay", names, index=default)
        speed = c2.selectbox(
            "Speed",
            [1.0, 2.0, 5.0, 10.0, 50.0],
            index=2,
            help="Replay speed. Original capture timestamps are preserved, "
            "so features are identical at any speed.",
        )
        max_frames = c3.number_input("Max frames (0 = all)", min_value=0, value=0, step=50_000)
        iface = os.environ.get("CANIDS_REPLAY_INTERFACE", APP["replay"]["interface"])
        c4.write("")
        b1, b2 = c4.columns(2)
        if b1.button("Start", type="primary", disabled=session.running):
            session.start(
                files[names.index(choice)],
                float(speed),
                iface,
                APP["replay"]["channel"],
                APP["replay"]["virtual_channel"],
                int(max_frames) or None,
            )
        if b2.button("Stop", disabled=not session.running):
            session.stop()

    @st.fragment(run_every=REFRESH)
    def live_panel() -> None:
        s = get_session().snapshot()
        if s.get("error"):
            st.error(s["error"])
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Frames / s (now)", f"{s.get('fps_now', 0):,.0f}")
        k2.metric("Total frames", f"{s.get('frames', 0):,}")
        k3.metric("Alert frames", f"{s.get('alert_frames', 0):,}")
        k4.metric("Alert events", f"{s.get('events', 0):,}")
        k5.markdown("**Status**  \n### " + STATUS_TEXT.get(s.get("status", "green"), ""))
        info = s.get("info") or {}
        if info:
            state = "running" if s["running"] else "finished"
            st.caption(
                f"Session `{info['session']}` · {Path(info['file']).name} · "
                f"{info['speed']}× · {info['interface']}:{info['channel']} · {state}"
            )
        ps = pd.DataFrame(s.get("per_second", []), columns=["sec", "frames", "alerts"])
        if not ps.empty:
            ps["time"] = pd.to_datetime(ps["sec"], unit="s")
            fig = go.Figure()
            fig.add_scatter(x=ps["time"], y=ps["frames"], name="frames/s", mode="lines")
            fig.add_bar(
                x=ps["time"],
                y=ps["alerts"],
                name="alert frames/s",
                yaxis="y2",
                marker_color="#f44336",
                opacity=0.6,
            )
            fig.update_layout(
                height=260,
                margin=dict(t=10, b=10, l=10, r=10),
                yaxis=dict(title="frames/s"),
                yaxis2=dict(title="alerts/s", overlaying="y", side="right"),
                legend=dict(orientation="h", y=1.15),
            )
            st.plotly_chart(fig, use_container_width=True)
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Recent alert events**")
            ev = events_frame(session=s.get("session") or None, limit=int(DASH["max_table_rows"]))
            if ev.empty:
                st.info("No alerts yet.")
            else:
                cols = [
                    "t_rel_s",
                    "predicted_class",
                    "severity",
                    "CAN ID",
                    "count",
                    "distinct_ids",
                    "class_probability",
                    "true_label",
                    "explanation_text",
                ]
                st.dataframe(
                    ev[cols].round(3), hide_index=True, use_container_width=True, height=320
                )
        with right:
            if s.get("truth_confusion"):
                st.markdown("**Replayed ground truth vs decision (frames)**")
                cm = pd.Series(s["truth_confusion"]).unstack(fill_value=0)
                cm.index.name, cm.columns.name = "true", "predicted"
                st.dataframe(cm, use_container_width=True)
            if s.get("class_counts"):
                st.markdown("**Decisions so far**")
                st.dataframe(pd.Series(s["class_counts"], name="frames"), use_container_width=True)

    live_panel()

# =========================================================================== b) analytics
with tabs[1]:
    scope = st.radio("Scope", ["Current session", "All sessions"], horizontal=True, key="an_scope")
    sid = get_session().session_id if scope == "Current session" else None
    ev = events_frame(session=sid or None)
    if ev.empty:
        st.info("No alert events stored yet - start a replay in Live Monitor.")
    else:
        c1, c2 = st.columns(2)
        by_class = ev.groupby("predicted_class")["count"].agg(["size", "sum"]).reset_index()
        by_class.columns = ["class", "events", "frames"]
        c1.plotly_chart(
            px.pie(
                by_class, names="class", values="frames", hole=0.4, title="Alert frames by category"
            ),
            use_container_width=True,
        )
        top = ev.groupby("CAN ID")["count"].sum().nlargest(15).reset_index()
        c2.plotly_chart(
            px.bar(
                top,
                x="count",
                y="CAN ID",
                orientation="h",
                title="Top offending CAN IDs (alert frames)",
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            px.scatter(
                ev,
                x="t_rel_s",
                y="predicted_class",
                size="count",
                color="severity",
                color_discrete_map=SEV_COLORS,
                hover_data=["CAN ID", "count", "true_label"],
                title="Alert timeline (capture time, s)",
                height=320,
            ),
            use_container_width=True,
        )
        bins = max(10, min(60, int(ev["t_rel_s"].max() // 2) + 1))
        ev["t_bin"] = pd.cut(ev["t_rel_s"], bins=bins).map(lambda iv: round(iv.left, 1))
        heat = ev.pivot_table(
            index="CAN ID",
            columns="t_bin",
            values="count",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        heat = heat.loc[heat.sum(axis=1).nlargest(20).index]
        st.plotly_chart(
            px.imshow(
                heat,
                aspect="auto",
                color_continuous_scale="Reds",
                labels=dict(x="capture time bin (s)", y="CAN ID", color="frames"),
                title="Alert frames: CAN ID × time",
            ),
            use_container_width=True,
        )

# =========================================================================== c) alert detail
with tabs[2]:
    ev = events_frame(limit=300)
    if ev.empty:
        st.info("No alert events yet.")
    else:
        labels = [
            f"{r.wall_time[11:23]}  {r.predicted_class:<9} {r['CAN ID']}  ×{r['count']}  "
            f"({r.severity})"
            for _, r in ev.iterrows()
        ]
        pick = st.selectbox(
            "Select an alert event (newest first)", range(len(ev)), format_func=lambda i: labels[i]
        )
        r = ev.iloc[pick]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted", r["predicted_class"])
        c2.metric("Severity", r["severity"])
        c3.metric("Class probability", f"{r['class_probability']:.3f}")
        c4.metric("Anomaly score", f"{r['anomaly_score']:.3f}")
        st.markdown(f"**Explanation:** {r['explanation_text'] or '_computing…_'}")
        st.caption(
            f"Decided by hybrid rule `{r['rule']}` · frames in event: {r['count']} · "
            f"distinct IDs: {r['distinct_ids']} · source: {r['source']} · "
            f"ground truth (replay): {r['true_label'] or 'n/a'}"
        )
        shap_df = pd.DataFrame(r["shap_top_features"])
        if not shap_df.empty:
            shap_df["label"] = shap_df.apply(lambda x: f"{x['feature']} = {x['value']:.4g}", axis=1)
            shap_df = shap_df.iloc[::-1]
            fig = px.bar(
                shap_df,
                x="shap",
                y="label",
                orientation="h",
                color=np.where(shap_df["shap"] > 0, "towards class", "against class"),
                color_discrete_map={"towards class": "#d62728", "against class": "#1f77b4"},
                title=f"SHAP contributions to '{r['predicted_class']}' (log-odds)",
            )
            fig.update_layout(height=320, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        st.caption(CAUSALITY_NOTE)
        frame = r["frame"] or {}
        if frame:
            st.markdown("**Raw frame (first frame of the event)**")
            payload = " ".join(
                "--" if frame.get(f"b{i}", -1) < 0 else f"{int(frame[f'b{i}']):02X}"
                for i in range(8)
            )
            st.code(f"ID {hex_id(frame['can_id'])}  DLC {int(frame['dlc'])}  DATA {payload}")
            st.dataframe(pd.Series(frame, name="value").to_frame().T, use_container_width=True)

# =========================================================================== d) history
with tabs[3]:
    allev = events_frame()
    if allev.empty:
        st.info("No alert events stored yet.")
    else:
        f1, f2, f3 = st.columns(3)
        classes = f1.multiselect("Class", sorted(allev["predicted_class"].unique()))
        sev = f2.multiselect("Severity", [s for s in SEVERITIES if s in set(allev["severity"])])
        ids_txt = f3.text_input("CAN IDs (hex, comma separated)", "")
        sessions = sorted(allev["session"].dropna().unique())
        sess = st.multiselect("Session", sessions)
        df = allev
        if classes:
            df = df[df["predicted_class"].isin(classes)]
        if sev:
            df = df[df["severity"].isin(sev)]
        if ids_txt.strip():
            try:
                wanted = {int(x.strip(), 16) for x in ids_txt.split(",") if x.strip()}
                df = df[df["can_id"].isin(wanted)]
            except ValueError:
                st.error("CAN IDs must be hexadecimal, e.g. 0D0, 6E0")
        if sess:
            df = df[df["session"].isin(sess)]
        st.caption(f"{len(df):,} events · {int(df['count'].sum()):,} alert frames")
        show = [
            "wall_time",
            "session",
            "predicted_class",
            "severity",
            "CAN ID",
            "count",
            "distinct_ids",
            "class_probability",
            "anomaly_score",
            "true_label",
            "rule",
            "explanation_text",
            "source",
        ]
        st.dataframe(df[show], hide_index=True, use_container_width=True, height=420)
        export = df.drop(columns=["frame"]).assign(
            shap_top_features=df["shap_top_features"].map(str)
        )
        st.download_button(
            "Export CSV",
            export.to_csv(index=False).encode("utf-8"),
            file_name=f"can_ids_events_{int(time.time())}.csv",
            mime="text/csv",
        )
        with st.expander("Danger zone"):
            if st.button("Delete ALL stored events"):
                get_store().clear()
                st.rerun()

# =========================================================================== e) model performance
with tabs[4]:
    res = resolve_path(DATA["paths"]["results"])
    figs = resolve_path(DATA["paths"]["figures"])
    st.caption(
        f"All numbers are read from saved result files in `{res.name}/` "
        "(held-out test set, evaluated once)."
    )

    def show_csv(name: str, title: str) -> None:
        p = res / name
        if p.exists():
            st.markdown(f"**{title}**")
            st.dataframe(pd.read_csv(p).round(4), hide_index=True, use_container_width=True)
        else:
            st.info(f"{name} not found (run the corresponding phase).")

    def show_img(name: str, caption: str) -> None:
        p = figs / name
        if p.exists():
            st.image(str(p), caption=caption, use_container_width=True)

    show_csv(f"{DATASET}_phase4_comparison.csv", "Baseline vs IF-only vs Classifier-only vs Hybrid")
    c1, c2 = st.columns(2)
    with c1:
        show_img(f"{DATASET}_phase4_cm_hy.png", "Hybrid confusion matrix (test)")
    with c2:
        show_img(f"{DATASET}_phase4_cm_clf.png", "Classifier-only confusion matrix (test)")
    show_csv(f"{DATASET}_phase4_detection_by_attack.csv", "Detection rate by attack type (test)")
    show_img(f"{DATASET}_phase3_roc_pr.png", "Stage-1 ROC / PR curves (test)")
    c1, c2 = st.columns(2)
    with c1:
        show_img(f"{DATASET}_shap_global_bar.png", "Global SHAP importance (validation)")
    with c2:
        show_img(f"{DATASET}_shap_beeswarm_spoofing.png", "SHAP summary - Spoofing")
    cross = res / "cross_dataset_results.csv"
    if cross.exists():
        show_csv("cross_dataset_results.csv", "Cross-dataset results")
    else:
        st.info("Cross-dataset results appear here after Phase 7 (needs HCRL data).")

# =========================================================================== f) health
with tabs[5]:

    @st.fragment(run_every=REFRESH)
    def health_panel() -> None:
        sess = get_session()
        s = sess.snapshot()
        lat = s.get("latency") or {}
        proc = psutil.Process()
        c = st.columns(6)
        c[0].metric("Latency p50", f"{lat.get('p50', float('nan')):.1f} ms")
        c[1].metric("Latency p95", f"{lat.get('p95', float('nan')):.1f} ms")
        c[2].metric("Latency p99", f"{lat.get('p99', float('nan')):.1f} ms")
        c[3].metric("Throughput (avg)", f"{s.get('fps_avg', 0):,.0f} fps")
        c[4].metric("Process CPU", f"{proc.cpu_percent(interval=None):.0f} %")
        c[5].metric("Process RAM", f"{proc.memory_info().rss / 2**20:,.0f} MB")
        st.caption(
            "Latency = frame received -> decision, including the micro-batch wait "
            f"(batch ≤ {APP['pipeline']['batch_size']} frames or "
            f"≤ {APP['pipeline']['max_wait_ms']} ms)."
        )
        if sess.pipeline is not None and sess.pipeline.stats.latencies_ms:
            lats = np.fromiter(sess.pipeline.stats.latencies_ms, dtype=float)
            st.plotly_chart(
                px.histogram(
                    x=lats,
                    nbins=60,
                    labels={"x": "decision latency (ms)"},
                    title="Latency distribution (recent frames)",
                    height=280,
                ),
                use_container_width=True,
            )
        vm = psutil.virtual_memory()
        st.caption(
            f"Host: {platform.processor() or platform.machine()} · "
            f"{psutil.cpu_count()} logical CPUs · system CPU "
            f"{psutil.cpu_percent(interval=None):.0f}% · RAM {vm.used / 2**30:.1f}/"
            f"{vm.total / 2**30:.1f} GB · Python {platform.python_version()}"
        )

    health_panel()

# =========================================================================== g) about
with tabs[6]:
    st.markdown(
        """
### CAN Bus Intrusion Detection System
**ML-Based Security Framework for In-Vehicle (CAN Bus) Networks**
B.Tech CSE Major Project · Dev Bhoomi Uttarakhand University (DBUU), Dehradun ·
School of Engineering and Computing, Dept. of CSE · Batch 2026–2027

| Team member | Roll No. |
|---|---|
| Uma Shankar | 23BTCSE0162 |
| Priyanshu Raj | 23BTCSE0300 |
| Ruby Kumari | 23BTCSE0068 |

**Supervisor:** Mr. Deepak Pathak

**How it works:** streaming features → Isolation Forest (anomaly, trained on normal traffic only)
→ XGBoost attack classifier → hybrid decision (Normal / Fuzzy / Spoofing / DoS / Unknown) →
aggregated alerts with SHAP explanations.

**Datasets:** HCRL Car-Hacking Dataset (Korea University); ROAD (Oak Ridge National Laboratory,
CC BY 4.0). ROAD contains no DoS attacks.
        """
    )
    st.warning(
        "**Safety & ethics.** Academic, passive, defensive research prototype. It uses public "
        "datasets and isolated virtual CAN only (python-can 'virtual' bus or vcan0). The replayer "
        "refuses any physical interface, and the live reader is listen-only with no send path. It "
        "is not a certified automotive safety system and is never connected to, or used to "
        "control, a road-going vehicle."
    )
