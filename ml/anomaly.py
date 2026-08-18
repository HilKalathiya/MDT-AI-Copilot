"""
ml/anomaly.py — Rolling z-score anomaly detection for MDT RSRP samples.

The MVP approach is a rolling z-score per UE over rsrp_dbm — simple,
explainable, and a direct extension of regression/statistics fundamentals.

Stretch: compare against IsolationForest over [rsrp_dbm, delta_rsrp_db].

Usage (programmatic):
    import pandas as pd, sqlite3
    conn = sqlite3.connect("data/mdt.db")
    df = pd.read_sql("SELECT * FROM ue_samples ORDER BY ue_rrc_id, logged_at", conn)
    from ml.anomaly import detect_anomalies_zscore
    result = detect_anomalies_zscore(df)
    print(result[["ue_rrc_id", "logged_at", "rsrp_dbm", "z_score", "is_anomaly"]].head())
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Primary detector: rolling z-score (Section 6.1)
# ---------------------------------------------------------------------------

def detect_anomalies_zscore(
    df: pd.DataFrame,
    window: int = 20,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Detect anomalies using a per-UE rolling z-score over rsrp_dbm.

    Adds three new columns to the returned DataFrame:
      rolling_mean  — rolling mean of rsrp_dbm per UE
      rolling_std   — rolling std  of rsrp_dbm per UE
      z_score       — (rsrp_dbm - rolling_mean) / rolling_std
      is_anomaly    — True where abs(z_score) > threshold

    Args:
        df:        DataFrame with at minimum: ue_rrc_id, logged_at, rsrp_dbm.
        window:    Rolling window size (samples).
        threshold: z-score magnitude that triggers an anomaly flag.

    Returns:
        A copy of df with the four new columns added.
    """
    df = df.sort_values(["ue_rrc_id", "logged_at"]).copy()
    g = df.groupby("ue_rrc_id")["rsrp_dbm"]
    df["rolling_mean"] = g.transform(
        lambda s: s.rolling(window, min_periods=5).mean()
    )
    df["rolling_std"] = g.transform(
        lambda s: s.rolling(window, min_periods=5).std()
    )
    df["z_score"] = (df["rsrp_dbm"] - df["rolling_mean"]) / df["rolling_std"].replace(
        0, 1e-6
    )
    df["is_anomaly"] = df["z_score"].abs() > threshold
    return df


# ---------------------------------------------------------------------------
# Stretch: IsolationForest-based detector
# ---------------------------------------------------------------------------

def detect_anomalies_isolation_forest(
    df: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Detect anomalies using sklearn IsolationForest over [rsrp_dbm, delta_rsrp_db].

    Adds:
      if_score    — anomaly score (lower = more anomalous)
      is_anomaly  — True for points classified as outliers

    Args:
        df:            DataFrame with rsrp_dbm and delta_rsrp_db columns.
        contamination: Expected proportion of outliers (0.0–0.5).
        random_state:  RNG seed for reproducibility.

    Returns:
        A copy of df with if_score and is_anomaly added.
    """
    from sklearn.ensemble import IsolationForest

    df = df.copy()
    feature_cols = ["rsrp_dbm", "delta_rsrp_db"]

    # Drop rows where delta is missing (first sample per UE)
    valid = df[feature_cols].dropna()
    X = valid.values

    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(X)

    df["if_score"] = np.nan
    df["is_anomaly"] = False
    df.loc[valid.index, "if_score"] = clf.decision_function(X)
    df.loc[valid.index, "is_anomaly"] = clf.predict(X) == -1  # -1 = outlier

    return df


# ---------------------------------------------------------------------------
# Convenience: load data directly from DB
# ---------------------------------------------------------------------------

def load_samples_from_db(db_path: str) -> pd.DataFrame:
    """Load all ue_samples rows into a DataFrame, sorted by UE then time."""
    import sqlite3
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM ue_samples ORDER BY ue_rrc_id, logged_at",
        conn,
    )
    conn.close()
    return df
