
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mission Control | A-8",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DATA
# ============================================================

RESULTS_PATH = (
    "/content/drive/MyDrive/"
    "Spacecraft-Anomaly-Detection/results"
)

REALTIME_FILE = RESULTS_PATH + "/realtime_detector_verified_A8.csv"
SUMMARY_FILE = RESULTS_PATH + "/serial29_realtime_summary_A8.csv"

realtime_df = pd.read_csv(REALTIME_FILE)
summary_df = pd.read_csv(SUMMARY_FILE)

threshold = float(summary_df["threshold"].iloc[0])

ANOMALY_START = 4569
ANOMALY_END = 8374

realtime_df["ground_truth"] = (
    (realtime_df["telemetry_index"] >= ANOMALY_START)
    & (realtime_df["telemetry_index"] <= ANOMALY_END)
).astype(int)

realtime_df["prediction"] = (
    realtime_df["decision"] == "ANOMALY"
).astype(int)

y_true = realtime_df["ground_truth"]
y_pred = realtime_df["prediction"]

tn, fp, fn, tp = confusion_matrix(
    y_true,
    y_pred
).ravel()

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

telemetry_count = len(realtime_df)
ground_truth_count = int(y_true.sum())
detected_count = int(y_pred.sum())

# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family: "Arial", sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 80% 0%,
            rgba(0, 140, 255, 0.08),
            transparent 32%
        ),
        #050b14;
    color: #e7edf5;
}

/* Remove Streamlit top padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1500px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #07111d;
    border-right: 1px solid #1c3147;
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.2rem;
}

.mission-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 5px 18px 5px;
    border-bottom: 1px solid #1d3044;
}

.mission-symbol {
    width: 42px;
    height: 42px;
    border: 1px solid #27d3ff;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #27d3ff;
    font-size: 23px;
    box-shadow: 0 0 18px rgba(39,211,255,.15);
}

.mission-name {
    font-size: 15px;
    font-weight: 800;
    letter-spacing: 2px;
}

.mission-sub {
    font-size: 9px;
    color: #708399;
    letter-spacing: 1.5px;
    margin-top: 3px;
}

.side-label {
    color: #60758c;
    font-size: 9px;
    letter-spacing: 2px;
    font-weight: 700;
    margin: 22px 5px 8px 5px;
}

.side-status {
    margin-top: 22px;
    padding: 12px;
    border: 1px solid #1c3549;
    border-radius: 7px;
    background: #091624;
}

.side-status-title {
    color: #6f849a;
    font-size: 9px;
    letter-spacing: 1.5px;
}

.side-status-value {
    color: #35e89a;
    font-size: 12px;
    font-weight: 700;
    margin-top: 5px;
}

/* Main command bar */
.command-bar {
    border: 1px solid #1c3449;
    background: linear-gradient(
        90deg,
        #091522,
        #0a1827
    );
    min-height: 74px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 22px;
    margin-bottom: 20px;
}

.command-left {
    display: flex;
    align-items: center;
    gap: 13px;
}

.command-mark {
    color: #27d3ff;
    font-size: 20px;
}

.command-title {
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 2px;
}

.command-sub {
    color: #71859b;
    font-size: 9px;
    letter-spacing: 1.4px;
    margin-top: 4px;
}

.command-right {
    display: flex;
    gap: 24px;
    align-items: center;
}

.command-item {
    font-size: 10px;
    color: #8ca0b5;
    letter-spacing: 1px;
}

.command-value {
    color: #dbe8f4;
    font-weight: 700;
}

.online {
    color: #35e89a;
}

/* Page title */
.kicker {
    color: #27d3ff;
    font-size: 9px;
    letter-spacing: 3px;
    font-weight: 800;
    margin-top: 8px;
}

.page-title {
    font-size: 31px;
    font-weight: 800;
    letter-spacing: -.5px;
    margin-top: 4px;
}

.page-description {
    color: #71859b;
    font-size: 12px;
    margin-top: 5px;
    margin-bottom: 20px;
}

/* Mission alert */
.mission-alert {
    border-left: 3px solid #35e89a;
    border-top: 1px solid #173c35;
    border-right: 1px solid #173c35;
    border-bottom: 1px solid #173c35;
    background: rgba(18, 59, 48, .20);
    padding: 13px 16px;
    border-radius: 5px;
    margin-bottom: 18px;
}

.alert-main {
    color: #35e89a;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.4px;
}

.alert-detail {
    color: #8095aa;
    font-size: 10px;
    margin-top: 4px;
}

/* KPI cards */
.kpi {
    background: #091522;
    border: 1px solid #1a3044;
    border-radius: 7px;
    padding: 16px;
    min-height: 94px;
}

.kpi-label {
    color: #708399;
    font-size: 9px;
    letter-spacing: 1.5px;
    font-weight: 700;
}

.kpi-value {
    color: #edf5fb;
    font-size: 25px;
    font-weight: 800;
    margin-top: 7px;
}

.kpi-note {
    color: #4d667e;
    font-size: 9px;
    margin-top: 3px;
}

/* Panels */
.panel {
    background: #08131f;
    border: 1px solid #1b3044;
    border-radius: 7px;
    padding: 17px;
    margin-top: 18px;
}

.panel-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 12px;
    border-bottom: 1px solid #172a3d;
    margin-bottom: 12px;
}

.panel-title {
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.7px;
}

.panel-tag {
    color: #27d3ff;
    font-size: 8px;
    letter-spacing: 1.3px;
}

/* Telemetry cards */
.telemetry-card {
    background: #0b1927;
    border: 1px solid #1b344b;
    border-radius: 6px;
    padding: 13px;
    margin-bottom: 9px;
}

.telemetry-label {
    color: #6e849a;
    font-size: 8px;
    letter-spacing: 1.4px;
}

.telemetry-value {
    color: #dce9f3;
    font-size: 17px;
    font-weight: 800;
    margin-top: 5px;
}

.telemetry-green {
    color: #35e89a;
}

.telemetry-cyan {
    color: #27d3ff;
}

.telemetry-amber {
    color: #f5b942;
}

/* Event console */
.event {
    display: flex;
    gap: 11px;
    padding: 10px 0;
    border-bottom: 1px solid #142536;
}

.event-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #f5b942;
    margin-top: 5px;
    flex-shrink: 0;
}

.event-time {
    color: #27d3ff;
    font-size: 9px;
    font-family: monospace;
}

.event-title {
    color: #d6e2ed;
    font-size: 10px;
    font-weight: 700;
}

.event-detail {
    color: #647b91;
    font-size: 9px;
    margin-top: 2px;
}

/* Mission footer */
.mission-footer {
    border-top: 1px solid #172b3e;
    margin-top: 28px;
    padding-top: 14px;
    color: #52687d;
    font-size: 9px;
    letter-spacing: 1px;
    display: flex;
    justify-content: space-between;
}

div[data-testid="stMetric"] {
    background: #091522;
    border: 1px solid #1a3044;
    border-radius: 6px;
    padding: 12px;
}

</style>
""",
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="mission-logo">
            <div class="mission-symbol">◉</div>
            <div>
                <div class="mission-name">MISSION CONTROL</div>
                <div class="mission-sub">AUTONOMOUS SPACE SYSTEMS</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="side-label">OPERATIONS</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Mission Overview",
            "Telemetry Monitor",
            "Anomaly Events",
            "Model Performance",
            "Research Configuration"
        ],
        label_visibility="collapsed"
    )

    st.markdown(
        """
        <div class="side-status">
            <div class="side-status-title">SYSTEM LINK</div>
            <div class="side-status-value">● TELEMETRY LINK ACTIVE</div>
            <div style="color:#62788e;font-size:9px;margin-top:5px;">
                CHANNEL A-8
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# TOP COMMAND BAR
# ============================================================

st.markdown(
    """
    <div class="command-bar">
        <div class="command-left">
            <div class="command-mark">◉</div>
            <div>
                <div class="command-title">SPACECRAFT A-8</div>
                <div class="command-sub">
                    AUTONOMOUS TELEMETRY & ANOMALY MONITORING
                </div>
            </div>
        </div>

        <div class="command-right">
            <div class="command-item">
                CHANNEL
                <span class="command-value">A-8</span>
            </div>

            <div class="command-item">
                MODEL
                <span class="command-value">LSTM-AE</span>
            </div>

            <div class="command-item online">
                ● LINK NOMINAL
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# MISSION OVERVIEW
# ============================================================

if page == "Mission Overview":

    st.markdown(
        '<div class="kicker">MISSION STATUS / REAL-TIME ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">Spacecraft Mission Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="page-description">
            Machine-learning telemetry monitoring for autonomous spacecraft
            anomaly detection. Current research baseline: telemetry channel A-8.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="mission-alert">
            <div class="alert-main">● TELEMETRY SYSTEM NOMINAL</div>
            <div class="alert-detail">
                Data stream available · anomaly detector operational ·
                reference event window identified
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # KPI ROW
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">PRECISION</div>
                <div class="kpi-value">{precision:.3f}</div>
                <div class="kpi-note">DETECTION ACCURACY</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">RECALL</div>
                <div class="kpi-value">{recall:.3f}</div>
                <div class="kpi-note">ANOMALY COVERAGE</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">F1 SCORE</div>
                <div class="kpi-value">{f1:.3f}</div>
                <div class="kpi-note">BALANCED METRIC</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="kpi">
                <div class="kpi-label">ALERT THRESHOLD</div>
                <div class="kpi-value">{threshold:.4f}</div>
                <div class="kpi-note">RECONSTRUCTION ERROR</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # TELEMETRY + STATUS
    left, right = st.columns([2.7, 1])

    with left:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-head">
                    <div class="panel-title">
                        A-8 TELEMETRY / ANOMALY SCORE
                    </div>
                    <div class="panel-tag">LIVE ANALYSIS VIEW</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        fig, ax = plt.subplots(figsize=(12, 4.2))

        ax.plot(
            realtime_df["telemetry_index"],
            realtime_df["anomaly_score"],
            linewidth=0.8
        )

        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.2
        )

        ax.axvspan(
            ANOMALY_START,
            ANOMALY_END,
            alpha=0.10
        )

        ax.set_facecolor("#08131f")
        fig.patch.set_facecolor("#08131f")

        ax.tick_params(
            colors="#8195a9",
            labelsize=8
        )

        for spine in ax.spines.values():
            spine.set_color("#20374d")

        ax.grid(
            alpha=0.14,
            linewidth=0.6
        )

        ax.set_xlabel(
            "Telemetry Index",
            color="#71859b",
            fontsize=8
        )

        ax.set_ylabel(
            "Anomaly Score",
            color="#71859b",
            fontsize=8
        )

        ax.set_title(
            "LSTM Autoencoder Reconstruction Error",
            color="#dce8f2",
            fontsize=10,
            loc="left",
            pad=10
        )

        st.pyplot(fig, use_container_width=True)

    with right:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-head">
                    <div class="panel-title">SPACECRAFT STATUS</div>
                    <div class="panel-tag">A-8</div>
                </div>

                <div class="telemetry-card">
                    <div class="telemetry-label">TELEMETRY WINDOWS</div>
                    <div class="telemetry-value telemetry-cyan">
                        8,326
                    </div>
                </div>

                <div class="telemetry-card">
                    <div class="telemetry-label">REFERENCE EVENT</div>
                    <div class="telemetry-value">
                        4,569 — 8,374
                    </div>
                </div>

                <div class="telemetry-card">
                    <div class="telemetry-label">DETECTED WINDOWS</div>
                    <div class="telemetry-value telemetry-amber">
                        6,144
                    </div>
                </div>

                <div class="telemetry-card">
                    <div class="telemetry-label">MODEL</div>
                    <div class="telemetry-value">
                        LSTM AUTOENCODER
                    </div>
                </div>

                <div class="telemetry-card">
                    <div class="telemetry-label">LINK STATE</div>
                    <div class="telemetry-value telemetry-green">
                        ● NOMINAL
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # EVENT CONSOLE
    left2, right2 = st.columns([1.4, 1])

    with left2:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-head">
                    <div class="panel-title">MISSION EVENT CONSOLE</div>
                    <div class="panel-tag">RECENT ACTIVITY</div>
                </div>

                <div class="event">
                    <div class="event-dot"></div>
                    <div>
                        <div class="event-time">IDX 4569</div>
                        <div class="event-title">
                            REFERENCE ANOMALY WINDOW OPEN
                        </div>
                        <div class="event-detail">
                            Ground-truth anomaly interval begins.
                        </div>
                    </div>
                </div>

                <div class="event">
                    <div class="event-dot"></div>
                    <div>
                        <div class="event-time">IDX 6144</div>
                        <div class="event-title">
                            DETECTION STREAM ACTIVE
                        </div>
                        <div class="event-detail">
                            Reconstruction-error detector generating alerts.
                        </div>
                    </div>
                </div>

                <div class="event">
                    <div class="event-dot"></div>
                    <div>
                        <div class="event-time">IDX 8374</div>
                        <div class="event-title">
                            REFERENCE ANOMALY WINDOW CLOSED
                        </div>
                        <div class="event-detail">
                            Ground-truth anomaly interval ends.
                        </div>
                    </div>
                </div>

                <div class="event">
                    <div class="event-dot"
                         style="background:#35e89a;"></div>
                    <div>
                        <div class="event-time">SYSTEM</div>
                        <div class="event-title">
                            TELEMETRY LINK NOMINAL
                        </div>
                        <div class="event-detail">
                            Mission dashboard operating normally.
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right2:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-head">
                    <div class="panel-title">DETECTION COUNTS</div>
                    <div class="panel-tag">A-8 BASELINE</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.metric("True Positives", f"{tp:,}")
        st.metric("False Positives", f"{fp:,}")
        st.metric("False Negatives", f"{fn:,}")
        st.metric("True Negatives", f"{tn:,}")

# ============================================================
# TELEMETRY MONITOR
# ============================================================

elif page == "Telemetry Monitor":

    st.markdown(
        '<div class="kicker">TELEMETRY OPERATIONS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">Telemetry Monitor</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="page-description">
            Detailed anomaly-score monitoring for spacecraft telemetry
            channel A-8.
        </div>
        """,
        unsafe_allow_html=True
    )

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        realtime_df["telemetry_index"],
        realtime_df["anomaly_score"],
        linewidth=0.9
    )

    ax.axhline(
        threshold,
        linestyle="--",
        linewidth=1.3
    )

    ax.axvspan(
        ANOMALY_START,
        ANOMALY_END,
        alpha=0.12
    )

    ax.set_facecolor("#08131f")
    fig.patch.set_facecolor("#08131f")

    ax.tick_params(colors="#8195a9")

    for spine in ax.spines.values():
        spine.set_color("#20374d")

    ax.grid(alpha=0.14)

    ax.set_title(
        "A-8 Reconstruction Error / Anomaly Detection",
        color="#dce8f2",
        fontsize=12,
        loc="left"
    )

    ax.set_xlabel(
        "Telemetry Index",
        color="#71859b"
    )

    ax.set_ylabel(
        "Anomaly Score",
        color="#71859b"
    )

    st.pyplot(fig, use_container_width=True)

    a, b, c = st.columns(3)

    a.metric("Telemetry Samples", f"{telemetry_count:,}")
    b.metric("Detected Anomalies", f"{detected_count:,}")
    c.metric("Reference Anomaly Windows", f"{ground_truth_count:,}")

# ============================================================
# ANOMALY EVENTS
# ============================================================

elif page == "Anomaly Events":

    st.markdown(
        '<div class="kicker">EVENT ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">Anomaly Event Console</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="page-description">
            Ground-truth reference interval and model-generated anomaly
            classifications.
        </div>
        """,
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    a.metric("Reference Windows", f"{ground_truth_count:,}")
    b.metric("Detected Windows", f"{detected_count:,}")
    c.metric("False Positives", f"{fp:,}")

    st.markdown(
        """
        <div class="panel">
            <div class="panel-head">
                <div class="panel-title">EVENT REFERENCE</div>
                <div class="panel-tag">GROUND TRUTH / MODEL</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    event_table = pd.DataFrame(
        {
            "Parameter": [
                "Reference anomaly start",
                "Reference anomaly end",
                "Reference anomaly windows",
                "Detected anomaly windows",
                "True positives",
                "False positives",
                "False negatives",
                "True negatives"
            ],
            "Value": [
                f"{ANOMALY_START:,}",
                f"{ANOMALY_END:,}",
                f"{ground_truth_count:,}",
                f"{detected_count:,}",
                f"{tp:,}",
                f"{fp:,}",
                f"{fn:,}",
                f"{tn:,}"
            ]
        }
    )

    st.dataframe(
        event_table,
        use_container_width=True,
        hide_index=True
    )

# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="kicker">MODEL EVALUATION</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">Detection Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="page-description">
            Evaluation of the current LSTM Autoencoder A-8 baseline against
            the available reference anomaly interval.
        </div>
        """,
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    a.metric("Precision", f"{precision:.3f}")
    b.metric("Recall", f"{recall:.3f}")
    c.metric("F1 Score", f"{f1:.3f}")

    st.markdown("### Confusion Matrix")

    cm_df = pd.DataFrame(
        [
            ["True Negative", tn],
            ["False Positive", fp],
            ["False Negative", fn],
            ["True Positive", tp]
        ],
        columns=["Classification", "Count"]
    )

    st.dataframe(
        cm_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Metric Definitions")

    st.write(
        "Precision = proportion of predicted anomalies that were "
        "correctly classified."
    )

    st.write(
        "Recall = proportion of reference anomaly windows detected "
        "by the model."
    )

    st.write(
        "F1 = harmonic mean of precision and recall."
    )

# ============================================================
# RESEARCH CONFIGURATION
# ============================================================

else:

    st.markdown(
        '<div class="kicker">RESEARCH CONFIGURATION</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="page-title">Mission Research Configuration</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="page-description">
            Current experimental configuration for the spacecraft anomaly
            detection research project.
        </div>
        """,
        unsafe_allow_html=True
    )

    config = pd.DataFrame(
        {
            "Configuration": [
                "Project",
                "Dataset",
                "Spacecraft channel",
                "Model",
                "Detection method",
                "Threshold",
                "Telemetry samples",
                "Reference anomaly start",
                "Reference anomaly end"
            ],
            "Current Value": [
                "Machine Learning-Based Anomaly Detection for Autonomous Space Systems",
                "Spacecraft telemetry",
                "A-8",
                "LSTM Autoencoder",
                "Reconstruction Error",
                f"{threshold:.6f}",
                f"{telemetry_count:,}",
                f"{ANOMALY_START:,}",
                f"{ANOMALY_END:,}"
            ]
        }
    )

    st.dataframe(
        config,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        """
        <div class="panel">
            <div class="panel-head">
                <div class="panel-title">RESEARCH ROADMAP</div>
                <div class="panel-tag">NEXT EXPERIMENTS</div>
            </div>

            <p style="color:#8195a9;font-size:11px;line-height:1.7;">
                A-8 is maintained as the current single-channel baseline.
                Future research can extend this baseline to multivariate
                telemetry, additional spacecraft channels, alternative
                anomaly-detection algorithms, comparative experiments,
                explainability, and lightweight deployment.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="mission-footer">
        <span>MISSION CONTROL / AUTONOMOUS SPACE SYSTEMS</span>
        <span>A-8 · RESEARCH PROTOTYPE · TELEMETRY LINK NOMINAL</span>
    </div>
    """,
    unsafe_allow_html=True
)
