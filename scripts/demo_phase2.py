"""
scripts/demo_phase2.py — Phase 2 ML evaluation demo.

Runs:
  1. Loads (or generates) synthetic data
  2. Runs z-score anomaly detection
  3. Evaluates vs. is_injected_anomaly ground truth
  4. Trains GradientBoosting forecaster
  5. Evaluates MAE vs. naive baseline

Phase 2 definition of done:
  - Recall ≥ 80% on injected anomalies
  - Model MAE < naive baseline MAE

Usage:
    python scripts/demo_phase2.py
    python scripts/demo_phase2.py --db data/mdt.db --regenerate
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd

from pipeline.db import init_db, table_row_counts
from pipeline.synthetic_gen import generate_synthetic_dataset
from ml.anomaly import detect_anomalies_zscore, load_samples_from_db
from ml.evaluate import evaluate_anomaly_detector, print_anomaly_report, print_forecast_report
from ml.forecast import load_and_train


def run_demo(db_path: str, regenerate: bool = False) -> None:
    print("=" * 60)
    print("  MDT AI Copilot — Phase 2 ML Demo")
    print("=" * 60)

    # Ensure DB exists and has data
    conn = init_db(db_path)
    counts = table_row_counts(conn)
    conn.close()

    if counts["ue_samples"] == 0 or regenerate:
        print(f"\n[1/4] Generating synthetic data into {db_path}...")
        result = generate_synthetic_dataset(db_path=db_path, num_ues=10, duration_s=3600, seed=42)
        print(f"      {result['rows_inserted']} rows, anomalous UEs: {result['anomalous_ues']}")
    else:
        print(f"\n[1/4] Using existing data: {counts['ue_samples']} rows in ue_samples")

    # Anomaly detection
    print("\n[2/4] Running z-score anomaly detection...")
    df = load_samples_from_db(db_path)
    result_df = detect_anomalies_zscore(df, window=20, threshold=3.0)

    total_anomalous = result_df["is_anomaly"].sum()
    total_injected = result_df["is_injected_anomaly"].sum()
    print(f"      Samples flagged as anomaly: {total_anomalous}")
    print(f"      Ground-truth anomaly samples: {total_injected}")

    # Evaluation
    print("\n[3/4] Evaluating anomaly detector...")
    anomaly_metrics = evaluate_anomaly_detector(result_df)
    print_anomaly_report(anomaly_metrics)

    # Forecasting
    print("\n[4/4] Training and evaluating RSRP forecaster...")
    _, _, forecast_metrics = load_and_train(db_path, n_lags=5)
    print_forecast_report(forecast_metrics)

    # Phase 2 checks
    print("\n" + "=" * 60)
    print("  Phase 2 — Definition of Done Checks")
    print("=" * 60)
    checks = [
        ("Anomaly recall ≥ 80%",    anomaly_metrics["recall"] >= 0.80),
        ("Forecaster beats naive",  forecast_metrics["model_mae"] < forecast_metrics["naive_mae"]),
    ]
    all_pass = True
    for label, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status}  {label}")
        if not passed:
            all_pass = False
    print()
    if all_pass:
        print("  🎉  All Phase 2 checks passed!")
    else:
        print("  ⚠️   Some checks failed — see details above.")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="MDT AI Copilot — Phase 2 demo")
    parser.add_argument("--db", default="data/mdt.db", help="Path to SQLite DB")
    parser.add_argument("--regenerate", action="store_true",
                        help="Re-generate synthetic data even if DB already has rows")
    args = parser.parse_args()
    run_demo(db_path=args.db, regenerate=args.regenerate)


if __name__ == "__main__":
    _cli()
