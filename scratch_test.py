import sqlite3
import pandas as pd
from ml.anomaly import detect_anomalies_zscore
from ml.evaluate import evaluate_anomaly_detector
from ml.forecast import load_and_train
from pipeline.synthetic_gen import generate_synthetic_dataset

print("Generating data...")
generate_synthetic_dataset("data/mdt.db")

df = pd.read_sql_query("SELECT * FROM ue_samples ORDER BY ue_rrc_id, logged_at", sqlite3.connect("data/mdt.db"))

print("Testing z-score with different params:")
for w in [10, 20, 50]:
    for t in [1.0, 1.5, 2.0, 3.0]:
        res = detect_anomalies_zscore(df, window=w, threshold=t)
        metrics = evaluate_anomaly_detector(res)
        print(f"w={w} t={t} -> recall={metrics['recall']:.2f}")

_, _, fm = load_and_train("data/mdt.db", n_lags=5)
print(f"Lags=5 -> model={fm['model_mae']:.4f} naive={fm['naive_mae']:.4f}")

_, _, fm = load_and_train("data/mdt.db", n_lags=1)
print(f"Lags=1 -> model={fm['model_mae']:.4f} naive={fm['naive_mae']:.4f}")
