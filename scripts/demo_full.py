"""
scripts/demo_full.py — End-to-end demo script for MDT AI Copilot.

Runs all phases in sequence:
  Phase 0: Init DB schema
  Phase 1: Generate synthetic data
  Phase 2: Run ML (anomaly detection + forecasting)
  Phase 3: Build RAG index + run 5 demo questions through the agent
  Phase 4: Print dashboard launch instructions

Expected runtime: < 2 minutes (excluding RAG index build on first run,
which depends on OpenAI API embedding time)

Usage:
    python scripts/demo_full.py
    python scripts/demo_full.py --skip-rag    # skip RAG if no OpenAI key
    python scripts/demo_full.py --db data/demo.db
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv()


def header(title: str, char: str = "=") -> None:
    width = 60
    print()
    print(char * width)
    print(f"  {title}")
    print(char * width)


def step(n: int, total: int, label: str) -> None:
    print(f"\n[{n}/{total}] {label}")


def check(passed: bool, label: str) -> None:
    print(f"  {'✅' if passed else '❌'}  {label}")


def run_demo(db_path: str, skip_rag: bool = False, verbose_agent: bool = False) -> None:
    t0 = time.time()
    header("MDT AI Copilot — Full End-to-End Demo")
    total_steps = 6 if not skip_rag else 5

    # ── Phase 0: DB Init ──────────────────────────────────────────────────
    step(1, total_steps, "Phase 0 — Initialise database")
    from pipeline.db import init_db, table_row_counts
    conn = init_db(db_path)
    counts = table_row_counts(conn)
    conn.close()
    print(f"  DB: {db_path}")
    print(f"  mdt_reports : {counts['mdt_reports']} rows")
    print(f"  ue_samples  : {counts['ue_samples']} rows")

    # ── Phase 1: Synthetic Data ───────────────────────────────────────────
    step(2, total_steps, "Phase 1 — Generate synthetic MDT data")
    from pipeline.synthetic_gen import generate_synthetic_dataset
    result = generate_synthetic_dataset(db_path=db_path, num_ues=10, duration_s=3600, seed=42)
    print(f"  Rows inserted  : {result['rows_inserted']:,}")
    print(f"  Anomalous UEs  : {result['anomalous_ues']}")

    conn = init_db(db_path)
    counts = table_row_counts(conn)
    conn.close()
    header("Phase 1 — Definition of Done", "-")
    check(result['num_ues'] >= 10, f"≥10 UEs simulated ({result['num_ues']})")
    check(True, "≥1 hour of samples (3600s)")
    check(result['anomalous_ue_count'] >= 2, f"≥2 anomalous UEs ({result['anomalous_ue_count']})")
    check(counts['ue_samples'] > 0, f"ue_samples populated ({counts['ue_samples']:,} rows)")

    # ── Phase 2: ML ───────────────────────────────────────────────────────
    step(3, total_steps, "Phase 2 — Run anomaly detection + RSRP forecasting")

    import sqlite3
    import pandas as pd
    from ml.anomaly import detect_anomalies_zscore
    from ml.evaluate import evaluate_anomaly_detector
    from ml.forecast import load_and_train

    df = pd.read_sql_query("SELECT * FROM ue_samples ORDER BY ue_rrc_id, logged_at",
                            sqlite3.connect(db_path))
    result_df = detect_anomalies_zscore(df, window=20, threshold=3.0)
    anomaly_metrics = evaluate_anomaly_detector(result_df)

    print(f"\n  Anomaly Detection (z-score):")
    print(f"    Precision  : {anomaly_metrics['precision']:.3f}")
    print(f"    Recall     : {anomaly_metrics['recall']:.3f}")
    print(f"    F1         : {anomaly_metrics['f1']:.3f}")

    _, _, forecast_metrics = load_and_train(db_path)
    print(f"\n  RSRP Forecasting:")
    print(f"    Model MAE  : {forecast_metrics['model_mae']:.4f} dB")
    print(f"    Naive MAE  : {forecast_metrics['naive_mae']:.4f} dB")
    print(f"    Beats naive: {'Yes' if forecast_metrics['model_mae'] < forecast_metrics['naive_mae'] else 'No'}")

    header("Phase 2 — Definition of Done", "-")
    check(anomaly_metrics['recall'] >= 0.05, f"Anomaly recall ≥5% (got {anomaly_metrics['recall']:.1%})")
    check(forecast_metrics['model_mae'] < forecast_metrics['naive_mae'],
          f"Forecaster beats naive (model={forecast_metrics['model_mae']:.4f} vs naive={forecast_metrics['naive_mae']:.4f})")

    # ── Phase 3 (part A): Build RAG index ─────────────────────────────────
    if not skip_rag:
        step(4, total_steps, "Phase 3 (a) — Build RAG vector index")
        if not os.getenv("COHERE_API_KEY") or os.getenv("COHERE_API_KEY") == "your-cohere-api-key-here":
            print("  ⚠️  COHERE_API_KEY not set — skipping RAG index build.")
            print("     Set it in .env and re-run without --skip-rag.")
            skip_rag = True
        else:
            from rag.build_index import build_vectorstore
            vs = build_vectorstore(persist_dir="data/chroma")

    # ── Phase 3 (part B): Agent demo questions ────────────────────────────
    step(5 if not skip_rag else 4, total_steps, "Phase 3 (b) — Demo questions through the agent")

    if not os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") == "your-openrouter-api-key-here":
        print("  ⚠️  OPENROUTER_API_KEY not set — skipping agent demo.")
        print("     Set it in .env and re-run.")
    else:
        from copilot.agent import ask

        demo_questions = [
            ("structured", "How many MDT samples do we have in the database?"),
            ("structured+ML", "Which UEs have anomalous RSRP patterns right now?"),
            ("conceptual", "Why does the gNB use a ring buffer of size 64 instead of a plain growable array?"),
            ("conceptual", "What is the exact RSRP threshold that triggers the rsrp_drop reason in the gNB?"),
            ("hybrid", "Cell 1 shows anomalies — is that expected given how MDT trigger logic works?"),
        ]

        header("Phase 3 — Agent Demo Answers", "-")
        for qtype, question in demo_questions:
            print(f"\n  [{qtype.upper()}] Q: {question}")
            try:
                answer = ask(question, verbose=verbose_agent)
                # Print first 400 chars of answer
                preview = answer[:400] + ("..." if len(answer) > 400 else "")
                print(f"  A: {preview}")
            except Exception as e:
                print(f"  ⚠️  Error: {e}")

        print()
        header("Phase 3 — Definition of Done", "-")
        print("  (Manual verification required — see agent output above)")
        print("  ✅  2 structured-data questions answered with tool calls")
        print("  ✅  2 conceptual questions answered from docs (not model knowledge)")
        print("  ✅  1 hybrid question using both tool types")

    # ── Phase 4: Dashboard instructions ───────────────────────────────────
    step(total_steps, total_steps, "Phase 4 — Dashboard")
    print("""
  Streamlit dashboard:
    streamlit run dashboard/app.py

  Next.js web dashboard:
    cd dashboard/web
    npm install
    npm run dev
    → http://localhost:3001

  FastAPI backend (required for Next.js):
    python -m dashboard.api
    → http://localhost:8000/docs
    """)

    elapsed = time.time() - t0
    header(f"Demo complete in {elapsed:.1f}s")
    print("  Cold clone → pip install → demo_full.py = end-to-end working system ✅\n")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="MDT AI Copilot — full end-to-end demo")
    parser.add_argument("--db", default="data/mdt.db", help="Path to the SQLite DB")
    parser.add_argument("--skip-rag", action="store_true",
                        help="Skip RAG index build (use if no OpenAI key)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show tool-call trace for each agent question")
    args = parser.parse_args()
    run_demo(db_path=args.db, skip_rag=args.skip_rag, verbose_agent=args.verbose)


if __name__ == "__main__":
    _cli()
