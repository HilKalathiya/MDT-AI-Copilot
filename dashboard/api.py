"""
dashboard/api.py — FastAPI backend for the Next.js web dashboard.

Endpoints:
  GET  /api/stats           — DB row counts and overview metrics
  GET  /api/samples         — UE samples with optional filters
  GET  /api/anomalies       — Run anomaly detection, return results
  GET  /api/cells           — Per-cell summary stats
  GET  /api/forecast/{ue_id} — RSRP forecast for a specific UE
  POST /api/chat            — Send a question to the LangGraph agent

Usage:
    pip install fastapi uvicorn
    python -m dashboard.api
    # or: uvicorn dashboard.api:app --reload --port 8000
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI is required for the web dashboard backend.\n"
        "Install with: pip install fastapi uvicorn"
    )

import sqlite3

DB_PATH = os.getenv("MDT_DB_PATH", "data/mdt.db")

app = FastAPI(
    title="MDT AI Copilot API",
    description="REST API for the 5G MDT Analytics and Agentic Copilot",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_conn() -> sqlite3.Connection:
    if not Path(DB_PATH).exists():
        raise HTTPException(status_code=503, detail=f"Database not found at {DB_PATH}. Run demo_phase1.py first.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# GET /api/stats
# ---------------------------------------------------------------------------

@app.get("/api/stats")
def get_stats():
    """Overview: row counts, avg RSRP, anomaly rate."""
    conn = _get_conn()
    ue_count = conn.execute("SELECT COUNT(*) FROM ue_samples").fetchone()[0]
    report_count = conn.execute("SELECT COUNT(*) FROM mdt_reports").fetchone()[0]
    avg_rsrp = conn.execute("SELECT ROUND(AVG(rsrp_dbm), 2) FROM ue_samples").fetchone()[0]
    n_ues = conn.execute("SELECT COUNT(DISTINCT ue_rrc_id) FROM ue_samples").fetchone()[0]
    conn.close()

    return {
        "ue_samples": ue_count,
        "mdt_reports": report_count,
        "avg_rsrp_dbm": avg_rsrp,
        "unique_ues": n_ues,
        "data_source": "synthetic",
    }


# ---------------------------------------------------------------------------
# GET /api/samples
# ---------------------------------------------------------------------------

@app.get("/api/samples")
def get_samples(
    ue_id: Optional[int] = Query(None),
    cell_id: Optional[int] = Query(None),
    limit: int = Query(200, le=1000),
):
    """Return UE samples with optional filters."""
    conn = _get_conn()
    conditions, params = [], []
    if ue_id is not None:
        conditions.append("ue_rrc_id = ?")
        params.append(ue_id)
    if cell_id is not None:
        conditions.append("nid_cell = ?")
        params.append(cell_id)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    rows = conn.execute(
        f"SELECT * FROM ue_samples {where} ORDER BY logged_at DESC LIMIT ?",
        params + [limit],
    ).fetchall()
    conn.close()
    return {"count": len(rows), "rows": [dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# GET /api/anomalies
# ---------------------------------------------------------------------------

@app.get("/api/anomalies")
def get_anomalies(
    window: int = Query(20, ge=5, le=100),
    threshold: float = Query(3.0, ge=1.0, le=6.0),
    cell_id: Optional[int] = Query(None),
):
    """Run z-score anomaly detection and return anomalous samples."""
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
        return {"anomalies": [], "total_samples": 0, "anomaly_count": 0}

    result_df = detect_anomalies_zscore(df, window=window, threshold=threshold)
    anom_df = result_df[result_df["is_anomaly"]].copy()

    # Precision/recall stats
    tp = int(((anom_df["is_injected_anomaly"] == 1)).sum())
    fp = int(((anom_df["is_injected_anomaly"] == 0)).sum())

    return {
        "total_samples": len(result_df),
        "anomaly_count": len(anom_df),
        "true_positives": tp,
        "false_positives": fp,
        "anomaly_rate": round(float(result_df["is_anomaly"].mean()), 4),
        "anomalies": anom_df[["ue_rrc_id", "rsrp_dbm", "z_score", "reason",
                              "is_injected_anomaly", "logged_at"]].to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# GET /api/cells
# ---------------------------------------------------------------------------

@app.get("/api/cells")
def get_cell_summary():
    """Per-cell summary statistics."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT
            nid_cell AS cell_id,
            COUNT(*) AS sample_count,
            COUNT(DISTINCT ue_rrc_id) AS unique_ues,
            ROUND(AVG(rsrp_dbm), 2) AS avg_rsrp_dbm,
            MIN(rsrp_dbm) AS min_rsrp_dbm,
            MAX(rsrp_dbm) AS max_rsrp_dbm,
            ROUND(AVG(delta_rsrp_db), 2) AS avg_delta_db
        FROM ue_samples
        GROUP BY nid_cell
        ORDER BY nid_cell
    """).fetchall()
    conn.close()
    return {"cells": [dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# GET /api/forecast/{ue_id}
# ---------------------------------------------------------------------------

@app.get("/api/forecast/{ue_id}")
def get_forecast(ue_id: int, n_steps: int = Query(10, ge=1, le=50)):
    """Forecast the next N RSRP values for a specific UE."""
    from ml.forecast import train_forecaster, predict_next

    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM ue_samples WHERE ue_rrc_id = ? ORDER BY logged_at",
        conn, params=(ue_id,)
    )
    conn.close()

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data for UE {ue_id}")
    if len(df) < 20:
        raise HTTPException(status_code=422, detail=f"Need ≥20 samples to forecast, got {len(df)}")

    all_df = pd.read_sql_query(
        "SELECT * FROM ue_samples ORDER BY ue_rrc_id, logged_at",
        sqlite3.connect(DB_PATH)
    )
    model, _ = train_forecaster(all_df)

    recent = df["rsrp_dbm"].tail(10).tolist()
    forecasts = []
    current = list(recent)
    for _ in range(n_steps):
        pred = predict_next(model, current)
        forecasts.append(round(pred, 2))
        current = current[1:] + [pred]

    last_logged = df["logged_at"].iloc[-1]
    return {
        "ue_id": ue_id,
        "last_actual_rsrp": float(df["rsrp_dbm"].iloc[-1]),
        "last_logged_at": last_logged,
        "forecast": forecasts,
        "n_steps": n_steps,
    }


# ---------------------------------------------------------------------------
# POST /api/chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Send a question to the LangGraph MDT copilot agent."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    from copilot.agent import ask
    try:
        answer = ask(req.question, verbose=False)
        return {"question": req.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# POST /api/init
# ---------------------------------------------------------------------------

@app.post("/api/init")
def init_database():
    """Initialise database and generate synthetic data."""
    from pipeline.db import init_db
    from pipeline.synthetic_gen import generate_synthetic_dataset

    init_db(DB_PATH)
    result = generate_synthetic_dataset(DB_PATH, num_ues=10, duration_s=3600, seed=42)
    return {"status": "ok", "rows_inserted": result["rows_inserted"],
            "anomalous_ues": result["anomalous_ues"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
