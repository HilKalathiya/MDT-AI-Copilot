"""
ml/evaluate.py — Evaluation utilities for Module 2 (Section 6.3).

Evaluation:
  - Anomaly detection: precision/recall against is_injected_anomaly ground truth
  - Forecasting: MAE vs. naive baseline, documented in a printable report

Usage:
    python -m ml.evaluate --db data/mdt.db
"""

from __future__ import annotations

import argparse
import sqlite3

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_anomaly_detector(
    df_with_predictions: pd.DataFrame,
) -> dict:
    """Compute precision/recall against is_injected_anomaly ground truth.

    Args:
        df_with_predictions: Output of detect_anomalies_zscore() or
                             detect_anomalies_isolation_forest() — must
                             have both is_injected_anomaly and is_anomaly columns.

    Returns:
        Dict with precision, recall, f1, confusion_matrix, and a
        classification_report string.
    """
    # Only evaluate rows where ground truth is available
    valid = df_with_predictions.dropna(subset=["is_injected_anomaly", "is_anomaly"])
    y_true = valid["is_injected_anomaly"].astype(int)
    y_pred = valid["is_anomaly"].astype(int)

    report_str = classification_report(
        y_true, y_pred, target_names=["normal", "anomaly"]
    )
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "classification_report": report_str,
    }


def print_anomaly_report(metrics: dict, method: str = "z-score") -> None:
    """Pretty-print anomaly evaluation results."""
    print(f"\n── Anomaly Detection ({method}) ──────────────────────")
    print(f"  Precision : {metrics['precision']:.3f}")
    print(f"  Recall    : {metrics['recall']:.3f}")
    print(f"  F1        : {metrics['f1']:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted Normal  Predicted Anomaly")
    print(f"  True Normal  {metrics['true_negatives']:>16d}  {metrics['false_positives']:>17d}")
    print(f"  True Anomaly {metrics['false_negatives']:>16d}  {metrics['true_positives']:>17d}")
    print(f"\n  Full Classification Report:\n")
    print(metrics["classification_report"])


def print_forecast_report(metrics: dict) -> None:
    """Pretty-print forecasting evaluation results."""
    beats = metrics["model_mae"] < metrics["naive_mae"]
    improvement = (
        (metrics["naive_mae"] - metrics["model_mae"]) / metrics["naive_mae"] * 100
        if metrics["naive_mae"] > 0 else 0.0
    )
    print(f"\n── RSRP Forecasting (GradientBoosting) ─────────────")
    print(f"  Model MAE  : {metrics['model_mae']:.4f} dB")
    print(f"  Naive MAE  : {metrics['naive_mae']:.4f} dB  (predict last value)")
    print(f"  Beats naive: {'✅  Yes' if beats else '❌  No'}")
    if beats:
        print(f"  Improvement: {improvement:.1f}% better than naive baseline")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="MDT AI Copilot — ML evaluation")
    parser.add_argument("--db", default="data/mdt.db", help="Path to the SQLite DB")
    parser.add_argument("--window", type=int, default=20, help="Anomaly rolling window")
    parser.add_argument("--threshold", type=float, default=3.0, help="Z-score threshold")
    parser.add_argument("--n-lags", type=int, default=5, help="Lag features for forecaster")
    args = parser.parse_args()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from ml.anomaly import detect_anomalies_zscore, load_samples_from_db
    from ml.forecast import load_and_train

    print("=" * 60)
    print("  MDT AI Copilot — Module 2 Evaluation")
    print("=" * 60)

    df = load_samples_from_db(args.db)
    print(f"\nLoaded {len(df)} samples from {args.db}")

    # Anomaly detection
    result_df = detect_anomalies_zscore(df, window=args.window, threshold=args.threshold)
    anomaly_metrics = evaluate_anomaly_detector(result_df)
    print_anomaly_report(anomaly_metrics, method=f"z-score (w={args.window}, t={args.threshold})")

    # Forecasting
    _, _, forecast_metrics = load_and_train(args.db, n_lags=args.n_lags)
    print_forecast_report(forecast_metrics)

    # Phase 2 definition of done
    print("\n" + "=" * 60)
    print("  Phase 2 — Definition of Done Checks")
    print("=" * 60)
    checks = [
        ("Anomaly recall ≥5%",      anomaly_metrics["recall"] >= 0.05),
        ("Forecaster beats naive",  forecast_metrics["model_mae"] < forecast_metrics["naive_mae"]),
    ]
    for label, passed in checks:
        print(f"  {'✅' if passed else '❌'}  {label}")


if __name__ == "__main__":
    _cli()
