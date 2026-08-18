"""
dashboard/app.py — Streamlit dashboard for MDT AI Copilot.

Panels:
  1. Overview: row counts, anomaly rate, DB freshness
  2. RSRP Time Series: per-UE RSRP chart with anomaly markers
  3. Anomaly Feed: live table of flagged samples
  4. Cell Summary: per-cell avg RSRP and health indicator
  5. Copilot Chat: send questions to the LangGraph agent

Usage:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MDT AI Copilot",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for premium dark theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global dark theme */
    .main { background: #0a0e1a; font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0d1529 50%, #0a1a2e 100%); }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1529 0%, #081121 100%);
        border-right: 1px solid rgba(56, 139, 253, 0.2);
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(13, 21, 41, 0.8);
        border: 1px solid rgba(56, 139, 253, 0.2);
        border-radius: 12px;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }

    /* Headers */
    h1, h2, h3 { color: #e2e8f0; font-family: 'Inter', sans-serif; }

    /* Chat input */
    .stChatInput textarea { background: rgba(13, 21, 41, 0.9) !important; color: #e2e8f0 !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(13, 21, 41, 0.6);
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        border-radius: 6px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(56, 139, 253, 0.2) !important;
        color: #60a5fa !important;
    }

    /* DataFrames */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }

    /* Status indicators */
    .status-good { color: #34d399; font-weight: 600; }
    .status-warn { color: #fbbf24; font-weight: 600; }
    .status-bad  { color: #f87171; font-weight: 600; }

    /* Section divider */
    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin: 1rem 0 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

DB_PATH = os.getenv("MDT_DB_PATH", "data/mdt.db")


@st.cache_data(ttl=30)
def load_samples() -> pd.DataFrame:
    """Load all ue_samples from the database."""
    import sqlite3
    if not Path(DB_PATH).exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM ue_samples ORDER BY ue_rrc_id, logged_at", conn
    )
    conn.close()
    return df


@st.cache_data(ttl=30)
def get_db_counts() -> dict:
    """Return row counts for both tables."""
    import sqlite3
    if not Path(DB_PATH).exists():
        return {"ue_samples": 0, "mdt_reports": 0}
    conn = sqlite3.connect(DB_PATH)
    ue = conn.execute("SELECT COUNT(*) FROM ue_samples").fetchone()[0]
    mdt = conn.execute("SELECT COUNT(*) FROM mdt_reports").fetchone()[0]
    conn.close()
    return {"ue_samples": ue, "mdt_reports": mdt}


@st.cache_data(ttl=60)
def run_anomaly_detection(window: int = 20, threshold: float = 3.0) -> pd.DataFrame:
    """Run z-score anomaly detection and cache the result."""
    from ml.anomaly import detect_anomalies_zscore
    df = load_samples()
    if df.empty:
        return pd.DataFrame()
    return detect_anomalies_zscore(df, window=window, threshold=threshold)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 📡 MDT AI Copilot")
    st.markdown("*5G Coverage Analyst*")
    st.divider()

    st.markdown('<p class="section-header">Database</p>', unsafe_allow_html=True)
    counts = get_db_counts()
    st.metric("UE Samples", f"{counts['ue_samples']:,}")
    st.metric("gNB Reports", f"{counts['mdt_reports']:,}")
    db_exists = Path(DB_PATH).exists()
    status = "🟢 Connected" if db_exists else "🔴 Not found"
    st.markdown(f"**DB Status:** {status}")
    st.caption(f"`{DB_PATH}`")

    st.divider()
    st.markdown('<p class="section-header">Anomaly Detection</p>', unsafe_allow_html=True)
    zscore_window = st.slider("Z-score window", 5, 50, 20, key="anom_window")
    zscore_threshold = st.slider("Z-score threshold", 1.0, 5.0, 3.0, 0.5, key="anom_thresh")

    st.divider()
    if st.button("🔄 Refresh data", width='stretch'):
        st.cache_data.clear()
        st.rerun()

    if st.button("⚙️ Init / Regenerate DB", width='stretch'):
        with st.spinner("Generating synthetic data..."):
            from pipeline.db import init_db
            from pipeline.synthetic_gen import generate_synthetic_dataset
            init_db(DB_PATH)
            result = generate_synthetic_dataset(DB_PATH, num_ues=10, duration_s=3600, seed=42)
            st.cache_data.clear()
        st.success(f"✅ Generated {result['rows_inserted']:,} rows")
        st.rerun()

# ---------------------------------------------------------------------------
# Main content — Tabs
# ---------------------------------------------------------------------------

st.markdown("# 📡 MDT AI Copilot Dashboard")
st.markdown("*5G Minimization of Drive Tests — Coverage Analytics & Agentic Copilot*")

tab_overview, tab_rsrp, tab_anomalies, tab_cells, tab_copilot = st.tabs([
    "📊 Overview",
    "📈 RSRP Trends",
    "⚠️ Anomaly Feed",
    "🗼 Cell Health",
    "🤖 AI Copilot",
])

# ── Tab 1: Overview ─────────────────────────────────────────────────────────
with tab_overview:
    if counts["ue_samples"] == 0:
        st.warning(
            "No data found. Click **⚙️ Init / Regenerate DB** in the sidebar to generate synthetic data."
        )
    else:
        df = load_samples()
        anom_df = run_anomaly_detection(zscore_window, zscore_threshold)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total UE Samples", f"{len(df):,}")
        with col2:
            n_ues = df["ue_rrc_id"].nunique() if not df.empty else 0
            st.metric("Active UEs", n_ues)
        with col3:
            anom_rate = float(anom_df["is_anomaly"].mean()) if not anom_df.empty else 0.0
            st.metric("Anomaly Rate", f"{anom_rate:.1%}")
        with col4:
            avg_rsrp = float(df["rsrp_dbm"].mean()) if not df.empty else 0.0
            st.metric("Avg RSRP", f"{avg_rsrp:.1f} dBm")

        st.divider()

        # RSRP distribution histogram
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("RSRP Distribution")
            fig = px.histogram(
                df, x="rsrp_dbm", nbins=40, color_discrete_sequence=["#60a5fa"],
                title="RSRP Distribution Across All UEs",
                labels={"rsrp_dbm": "RSRP (dBm)", "count": "Samples"},
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", title_font_color="#e2e8f0",
            )
            fig.add_vline(x=-100, line_dash="dash", line_color="#f87171",
                         annotation_text="low_rsrp threshold", annotation_font_color="#f87171")
            st.plotly_chart(fig, width='stretch')

        with col_right:
            st.subheader("Trigger Reason Distribution")
            reason_counts = df["reason"].value_counts().reset_index()
            reason_counts.columns = ["reason", "count"]
            fig2 = px.pie(
                reason_counts, names="reason", values="count",
                color_discrete_sequence=px.colors.qualitative.Set3,
                title="MDT Trigger Reasons",
            )
            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0", title_font_color="#e2e8f0",
            )
            st.plotly_chart(fig2, width='stretch')

        # Per-UE summary table
        st.subheader("Per-UE Summary")
        if not anom_df.empty:
            ue_summary = anom_df.groupby("ue_rrc_id").agg(
                samples=("rsrp_dbm", "count"),
                avg_rsrp=("rsrp_dbm", "mean"),
                min_rsrp=("rsrp_dbm", "min"),
                anomalies=("is_anomaly", "sum"),
                injected=("is_injected_anomaly", "sum"),
            ).reset_index()
            ue_summary["avg_rsrp"] = ue_summary["avg_rsrp"].round(1)
            ue_summary["anomaly_rate"] = (ue_summary["anomalies"] / ue_summary["samples"]).map(
                lambda x: f"{x:.1%}"
            )
            ue_summary["status"] = ue_summary["avg_rsrp"].apply(
                lambda r: "🟢 Good" if r > -90 else ("🟡 Fair" if r > -100 else "🔴 Poor")
            )
            st.dataframe(ue_summary, width='stretch')


# ── Tab 2: RSRP Trends ───────────────────────────────────────────────────────
with tab_rsrp:
    df = load_samples()
    if df.empty:
        st.warning("No data. Use the sidebar to generate synthetic data.")
    else:
        anom_df = run_anomaly_detection(zscore_window, zscore_threshold)

        ue_options = sorted(df["ue_rrc_id"].unique())
        selected_ues = st.multiselect(
            "Select UEs to plot", ue_options, default=ue_options[:3], key="rsrp_ues"
        )

        if selected_ues:
            plot_df = anom_df[anom_df["ue_rrc_id"].isin(selected_ues)].copy()
            plot_df["logged_at"] = pd.to_datetime(plot_df["logged_at"])
            plot_df["ue_label"] = "UE " + plot_df["ue_rrc_id"].astype(str)

            fig = go.Figure()
            colors = px.colors.qualitative.Plotly

            for i, ue_id in enumerate(selected_ues):
                ue_data = plot_df[plot_df["ue_rrc_id"] == ue_id]
                color = colors[i % len(colors)]

                # Normal points
                normal = ue_data[~ue_data["is_anomaly"]]
                fig.add_trace(go.Scatter(
                    x=normal["logged_at"], y=normal["rsrp_dbm"],
                    name=f"UE {ue_id} (normal)",
                    mode="lines", line=dict(color=color, width=1.5),
                    opacity=0.8,
                ))
                # Anomaly points
                anomalies = ue_data[ue_data["is_anomaly"]]
                if not anomalies.empty:
                    fig.add_trace(go.Scatter(
                        x=anomalies["logged_at"], y=anomalies["rsrp_dbm"],
                        name=f"UE {ue_id} (anomaly)",
                        mode="markers",
                        marker=dict(color="#f87171", size=8, symbol="x",
                                   line=dict(width=2, color="#f87171")),
                    ))

            fig.add_hline(y=-100, line_dash="dash", line_color="#fbbf24",
                         annotation_text="low_rsrp threshold (−100 dBm)",
                         annotation_font_color="#fbbf24")
            fig.update_layout(
                title="RSRP Over Time per UE (anomalies marked ×)",
                xaxis_title="Time",
                yaxis_title="RSRP (dBm)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#e2e8f0",
                title_font_color="#e2e8f0",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
                height=500,
            )
            st.plotly_chart(fig, width='stretch')

            # Rolling z-score
            if st.checkbox("Show rolling z-score", key="show_zscore"):
                plot_z = plot_df[plot_df["ue_rrc_id"] == selected_ues[0]]
                fig_z = px.line(
                    plot_z, x="logged_at", y="z_score",
                    title=f"Rolling Z-score — UE {selected_ues[0]}",
                    color_discrete_sequence=["#a78bfa"],
                    labels={"logged_at": "Time", "z_score": "Z-score"},
                )
                fig_z.add_hline(y=zscore_threshold, line_dash="dash", line_color="#f87171",
                               annotation_text=f"threshold={zscore_threshold}")
                fig_z.add_hline(y=-zscore_threshold, line_dash="dash", line_color="#f87171")
                fig_z.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#e2e8f0", title_font_color="#e2e8f0",
                )
                st.plotly_chart(fig_z, width='stretch')


# ── Tab 3: Anomaly Feed ──────────────────────────────────────────────────────
with tab_anomalies:
    anom_df = run_anomaly_detection(zscore_window, zscore_threshold)
    if anom_df.empty:
        st.warning("No data available.")
    else:
        anomalies = anom_df[anom_df["is_anomaly"]].copy()
        anomalies["logged_at"] = pd.to_datetime(anomalies["logged_at"])

        st.subheader(f"⚠️ Anomalous Samples — {len(anomalies):,} flagged")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Anomalies Detected", len(anomalies))
        with col2:
            injected_hit = int((anomalies["is_injected_anomaly"] == 1).sum())
            st.metric("True Positives (vs ground truth)", injected_hit)
        with col3:
            fp = int((anomalies["is_injected_anomaly"] == 0).sum())
            st.metric("False Positives", fp)

        # Anomaly timeline
        fig = px.scatter(
            anomalies, x="logged_at", y="rsrp_dbm",
            color="ue_rrc_id", symbol="reason",
            title="Anomalous Samples Timeline",
            labels={"ue_rrc_id": "UE ID", "logged_at": "Time", "rsrp_dbm": "RSRP (dBm)"},
            color_continuous_scale="Viridis",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_color="#e2e8f0",
        )
        st.plotly_chart(fig, width='stretch')

        # Table
        display_cols = ["ue_rrc_id", "rsrp_dbm", "delta_rsrp_db", "z_score",
                       "reason", "is_injected_anomaly", "logged_at"]
        st.dataframe(
            anomalies[display_cols].sort_values("logged_at", ascending=False),
            width='stretch',
        )


# ── Tab 4: Cell Health ───────────────────────────────────────────────────────
with tab_cells:
    df = load_samples()
    if df.empty:
        st.warning("No data available.")
    else:
        st.subheader("🗼 Cell Health Overview")

        cell_stats = df.groupby("nid_cell").agg(
            sample_count=("rsrp_dbm", "count"),
            avg_rsrp=("rsrp_dbm", "mean"),
            min_rsrp=("rsrp_dbm", "min"),
            std_rsrp=("rsrp_dbm", "std"),
            unique_ues=("ue_rrc_id", "nunique"),
        ).reset_index()
        cell_stats["avg_rsrp"] = cell_stats["avg_rsrp"].round(2)
        cell_stats["std_rsrp"] = cell_stats["std_rsrp"].round(2)
        cell_stats["health"] = cell_stats["avg_rsrp"].apply(
            lambda r: "🟢 Good" if r > -90 else ("🟡 Fair" if r > -100 else "🔴 Poor")
        )

        st.dataframe(cell_stats, width='stretch')

        # Bar chart of avg RSRP per cell
        fig = px.bar(
            cell_stats, x="nid_cell", y="avg_rsrp",
            title="Average RSRP per Cell",
            labels={"nid_cell": "Cell ID", "avg_rsrp": "Avg RSRP (dBm)"},
            color="avg_rsrp",
            color_continuous_scale="RdYlGn",
            range_color=[-115, -70],
        )
        fig.add_hline(y=-100, line_dash="dash", line_color="#f87171",
                     annotation_text="−100 dBm threshold")
        fig.add_hline(y=-90, line_dash="dash", line_color="#fbbf24",
                     annotation_text="−90 dBm warning")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_color="#e2e8f0",
        )
        st.plotly_chart(fig, width='stretch')


# ── Tab 5: AI Copilot ────────────────────────────────────────────────────────
with tab_copilot:
    st.subheader("🤖 MDT AI Copilot Chat")
    st.caption("Powered by LangGraph | Cohere Command-R | Agentic RAG with Cohere embed-v4.0")

    # Check API key
    if not os.getenv("COHERE_API_KEY") or os.getenv("COHERE_API_KEY") == "your-cohere-api-key-here":
        st.warning("⚠️ COHERE_API_KEY not set. The copilot will not work until you configure it in .env and restart Streamlit.")
    else:
        # Example questions
        with st.expander("💡 Example questions", expanded=False):
            examples = [
                "How many MDT samples do we have for cell 1?",
                "Which UEs show anomalous RSRP patterns?",
                "Why does the gNB use a ring buffer instead of a plain array?",
                "What RSRP threshold triggers the rsrp_drop reason?",
                "Cell 1 has anomalies — is that expected given MDT trigger logic?",
                "What does the suggest_threshold tool recommend for cell 1?",
            ]
            for q in examples:
                if st.button(f"💬 {q}", key=f"ex_{q[:20]}", width='stretch'):
                    st.session_state["pending_question"] = q

        # Chat history
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Handle pre-filled example question
        pending = st.session_state.pop("pending_question", None)

        # Chat input
        user_input = st.chat_input("Ask the MDT Copilot anything...", key="chat_input")
        question = user_input or pending

        if question:
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        from copilot.agent import ask
                        answer = ask(question, verbose=False)
                        st.markdown(answer)
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": answer}
                        )
                    except Exception as e:
                        error_msg = f"⚠️ Error: {e}"
                        st.error(error_msg)
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": error_msg}
                        )

        if st.button("🗑️ Clear conversation", key="clear_chat"):
            st.session_state["messages"] = []
            st.rerun()
