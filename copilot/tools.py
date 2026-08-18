"""
copilot/tools.py — Structured tools and RAG retrieval tool for the MDT AI Copilot.

All tools are plain Python functions decorated with @tool (LangChain).
The agent calls them by name; LangChain/LangGraph handles serialisation.

Structured tools (Section 7.2):
  - query_reports       — query mdt_reports and ue_samples with filters
  - run_anomaly_scan    — run z-score anomaly detection on recent samples
  - get_cell_summary    — summary stats for one cell
  - suggest_threshold   — suggest adaptive RSRP thresholds for a cell

Retrieval tool (Section 7.3):
  - retrieve_docs       — semantic search over the Chroma vector store

All tool outputs are truncated to MAX_TOOL_OUTPUT_CHARS to prevent
context window overflow (guardrail per Section 7.4).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_PATH = os.getenv("MDT_DB_PATH", "data/mdt.db")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "data/chroma")
MAX_TOOL_OUTPUT_CHARS = int(os.getenv("TOOL_OUTPUT_MAX_CHARS", "4000"))


def _truncate(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Truncate tool output to prevent context window overflow."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated — {len(text) - max_chars} chars omitted]"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Structured Tool 1: query_reports
# ---------------------------------------------------------------------------

@tool
def query_reports(
    cell_id: Optional[int] = None,
    ue_id: Optional[int] = None,
    since_minutes: Optional[int] = None,
    limit: int = 50,
) -> str:
    """Query MDT UE samples from the database with optional filters.

    Returns recent RSRP samples filtered by cell, UE, or time window.
    Use this for data questions like 'how many reports from cell 3?' or
    'show me recent RSRP readings for UE 5'.

    Args:
        cell_id:       Filter to samples from this cell (nid_cell).
        ue_id:         Filter to samples from this UE (ue_rrc_id).
        since_minutes: Only return samples logged in the last N minutes.
        limit:         Maximum number of rows to return (default 50, max 200).

    Returns:
        JSON string of matching rows, or a message if no rows found.
    """
    conn = _get_conn()
    conditions = []
    params: list = []

    if cell_id is not None:
        conditions.append("nid_cell = ?")
        params.append(cell_id)
    if ue_id is not None:
        conditions.append("ue_rrc_id = ?")
        params.append(ue_id)
    if since_minutes is not None:
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=since_minutes)).isoformat()
        conditions.append("logged_at >= ?")
        params.append(cutoff)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    limit_val = min(limit, 200)
    query = f"""
        SELECT ue_rrc_id, nid_cell, rsrp_dbm, delta_rsrp_db, reason,
               is_injected_anomaly, logged_at
        FROM ue_samples
        {where}
        ORDER BY logged_at DESC
        LIMIT {limit_val}
    """

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        return "No samples found matching the given filters."

    result = {
        "count": len(rows),
        "filters": {
            "cell_id": cell_id,
            "ue_id": ue_id,
            "since_minutes": since_minutes,
        },
        "data_source": "synthetic (is_injected_anomaly field available)",
        "rows": [dict(r) for r in rows],
    }
    return _truncate(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Structured Tool 2: run_anomaly_scan
# ---------------------------------------------------------------------------

@tool
def run_anomaly_scan(
    cell_id: Optional[int] = None,
    window: int = 20,
    threshold: float = 3.0,
) -> str:
    """Run the z-score anomaly detector over recent RSRP samples.

    Uses a rolling z-score per UE to identify samples where RSRP deviates
    significantly from the UE's own recent baseline. Flags |z| > threshold.

    Use this when asked about coverage anomalies, coverage holes, or
    which UEs or cells show unusual signal patterns.

    Args:
        cell_id:   Optionally restrict scan to one cell.
        window:    Rolling window size in samples (default 20).
        threshold: Z-score magnitude to flag as anomaly (default 3.0).

    Returns:
        JSON summary: total samples scanned, anomaly count per UE, anomalous UEs.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ml.anomaly import detect_anomalies_zscore

    conn = _get_conn()
    q = "SELECT * FROM ue_samples"
    params: list = []
    if cell_id is not None:
        q += " WHERE nid_cell = ?"
        params.append(cell_id)
    q += " ORDER BY ue_rrc_id, logged_at"

    df = pd.read_sql_query(q, conn, params=params)
    conn.close()

    if df.empty:
        return "No samples found to scan."

    result_df = detect_anomalies_zscore(df, window=window, threshold=threshold)
    anomalous = result_df[result_df["is_anomaly"]]

    per_ue = (
        anomalous.groupby("ue_rrc_id")
        .agg(anomaly_count=("is_anomaly", "sum"))
        .reset_index()
        .to_dict(orient="records")
    )

    # Compute precision/recall if ground truth available
    evaluation = {}
    if "is_injected_anomaly" in result_df.columns:
        tp = int(((result_df["is_anomaly"]) & (result_df["is_injected_anomaly"] == 1)).sum())
        fp = int(((result_df["is_anomaly"]) & (result_df["is_injected_anomaly"] == 0)).sum())
        fn = int(((~result_df["is_anomaly"]) & (result_df["is_injected_anomaly"] == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        evaluation = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "note": "Evaluation against is_injected_anomaly ground truth (synthetic data only)",
        }

    result = {
        "data_source": "ue_samples",
        "cell_filter": cell_id,
        "detector": f"z-score (window={window}, threshold={threshold})",
        "total_samples": len(result_df),
        "anomalous_samples": int(result_df["is_anomaly"].sum()),
        "anomaly_rate": round(float(result_df["is_anomaly"].mean()), 4),
        "anomalous_ues": per_ue,
        "evaluation": evaluation,
    }
    return _truncate(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Structured Tool 3: get_cell_summary
# ---------------------------------------------------------------------------

@tool
def get_cell_summary(cell_id: int) -> str:
    """Get summary statistics for one cell: avg RSRP, sample count, anomaly rate.

    Use this when asked about the health or status of a specific cell.

    Args:
        cell_id: The cell ID (nid_cell) to summarise.

    Returns:
        JSON with avg/min/max RSRP, sample count, anomaly fraction, and
        most common trigger reasons for this cell.
    """
    conn = _get_conn()
    stats = conn.execute(
        """
        SELECT
            COUNT(*)                      AS sample_count,
            ROUND(AVG(rsrp_dbm), 2)       AS avg_rsrp_dbm,
            MIN(rsrp_dbm)                 AS min_rsrp_dbm,
            MAX(rsrp_dbm)                 AS max_rsrp_dbm,
            ROUND(AVG(delta_rsrp_db), 2)  AS avg_delta_db,
            SUM(is_injected_anomaly)      AS injected_anomaly_count
        FROM ue_samples
        WHERE nid_cell = ?
        """,
        (cell_id,),
    ).fetchone()

    if not stats or stats["sample_count"] == 0:
        conn.close()
        return f"No data found for cell_id={cell_id}."

    reasons = conn.execute(
        """
        SELECT reason, COUNT(*) as count
        FROM ue_samples
        WHERE nid_cell = ?
        GROUP BY reason
        ORDER BY count DESC
        """,
        (cell_id,),
    ).fetchall()

    ue_count = conn.execute(
        "SELECT COUNT(DISTINCT ue_rrc_id) FROM ue_samples WHERE nid_cell = ?",
        (cell_id,),
    ).fetchone()[0]
    conn.close()

    result = {
        "cell_id": cell_id,
        "sample_count": stats["sample_count"],
        "unique_ues": ue_count,
        "avg_rsrp_dbm": stats["avg_rsrp_dbm"],
        "min_rsrp_dbm": stats["min_rsrp_dbm"],
        "max_rsrp_dbm": stats["max_rsrp_dbm"],
        "avg_delta_db": stats["avg_delta_db"],
        "injected_anomaly_count": stats["injected_anomaly_count"],
        "trigger_reasons": [dict(r) for r in reasons],
        "data_source": "synthetic (ue_samples)",
    }
    return _truncate(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Structured Tool 4: suggest_threshold
# ---------------------------------------------------------------------------

@tool
def suggest_threshold(cell_id: int) -> str:
    """Suggest adaptive drop/low-RSRP thresholds for a cell based on its noise profile.

    The existing gNB uses fixed thresholds (3 dB drop, -100 dBm absolute).
    This tool computes per-cell suggestions based on the actual RSRP distribution,
    adapting to cells with higher natural variance or different baseline levels.

    Args:
        cell_id: The cell to compute threshold suggestions for.

    Returns:
        JSON with suggested drop_threshold_db and low_rsrp_threshold_dbm,
        plus the statistical basis for the suggestion.
    """
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT rsrp_dbm, delta_rsrp_db FROM ue_samples WHERE nid_cell = ?",
        conn,
        params=(cell_id,),
    )
    conn.close()

    if df.empty:
        return f"No data found for cell_id={cell_id}."

    rsrp_mean = float(df["rsrp_dbm"].mean())
    rsrp_std = float(df["rsrp_dbm"].std())
    rsrp_p5 = float(df["rsrp_dbm"].quantile(0.05))   # 5th percentile
    rsrp_p95 = float(df["rsrp_dbm"].quantile(0.95))  # 95th percentile

    delta_std = float(df["delta_rsrp_db"].dropna().std())

    # Suggest drop threshold as 2× the natural delta std (adaptive to local noise)
    # but at least 2 dB and at most 6 dB
    suggested_drop = round(max(2.0, min(6.0, 2 * delta_std)), 1)

    # Suggest low-RSRP threshold as 5th percentile - 1×std, bounded to [-115, -85]
    suggested_low = round(max(-115.0, min(-85.0, rsrp_p5 - rsrp_std)), 1)

    result = {
        "cell_id": cell_id,
        "current_gNB_thresholds": {
            "drop_threshold_db": 3,
            "low_rsrp_threshold_dbm": -100,
        },
        "suggested_thresholds": {
            "drop_threshold_db": suggested_drop,
            "low_rsrp_threshold_dbm": suggested_low,
        },
        "statistical_basis": {
            "rsrp_mean_dbm": round(rsrp_mean, 2),
            "rsrp_std_db": round(rsrp_std, 2),
            "rsrp_p5_dbm": round(rsrp_p5, 2),
            "rsrp_p95_dbm": round(rsrp_p95, 2),
            "delta_rsrp_std_db": round(delta_std, 2),
            "sample_count": len(df),
        },
        "rationale": (
            f"Drop threshold set to 2×delta_std ({suggested_drop} dB) to adapt to this "
            f"cell's natural {delta_std:.1f} dB per-sample variation. "
            f"Low-RSRP threshold set at 5th-pct ({rsrp_p5:.1f} dBm) - 1σ, "
            f"reflecting this cell's actual signal floor."
        ),
    }
    return _truncate(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Retrieval Tool: retrieve_docs
# ---------------------------------------------------------------------------

@tool
def retrieve_docs(query: str, k: int = 4) -> str:
    """Search the MDT project documentation and C source for conceptual answers.

    Use this for questions about how the system works, why design decisions
    were made, 3GPP standards, C data structures, ring buffer design,
    RSRP conversion formulas, or MDT trigger reasons.

    Do NOT use this for data questions (use query_reports, get_cell_summary instead).

    Args:
        query: A natural-language description of what you want to know.
        k:     Number of document chunks to retrieve (default 4, max 8).

    Returns:
        The top-k relevant document excerpts with source filenames.
    """
    from rag.build_index import load_vectorstore

    try:
        vs = load_vectorstore(CHROMA_PERSIST_DIR)
    except Exception as e:
        return (
            f"Vector store not found at '{CHROMA_PERSIST_DIR}'. "
            f"Run 'python -m rag.build_index' first to build the index.\n"
            f"Error: {e}"
        )

    k_clamped = min(k, 8)
    docs = vs.similarity_search(query, k=k_clamped)

    if not docs:
        return "No relevant documents found for this query."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        source_name = Path(source).name if source != "unknown" else "unknown"
        parts.append(
            f"[{i}] Source: {source_name}\n"
            f"{doc.page_content.strip()}"
        )

    combined = "\n\n---\n\n".join(parts)
    return _truncate(combined)


# ---------------------------------------------------------------------------
# Export list for agent
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    query_reports,
    run_anomaly_scan,
    get_cell_summary,
    suggest_threshold,
    retrieve_docs,
]
