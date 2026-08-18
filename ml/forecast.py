"""
ml/forecast.py — Lag-feature + GradientBoostingRegressor RSRP forecaster.

Approach (Section 6.2 of PROJECT_GUIDE.md):
  - Build lag features: lag_1 ... lag_N  (previous N rsrp_dbm values per UE)
  - Target: next rsrp_dbm value (shift(-1))
  - Model: GradientBoostingRegressor from sklearn

This beats a naive LSTM here because:
  - Simpler, faster to train on this data volume (~7000 rows)
  - Easier to explain in an interview (feature importances are interpretable)
  - Still clearly outperforms the naive "predict last value" baseline

Usage:
    from ml.forecast import train_forecaster, predict_next, evaluate_mae
    import pandas as pd, sqlite3
    conn = sqlite3.connect("data/mdt.db")
    df = pd.read_sql("SELECT * FROM ue_samples ORDER BY ue_rrc_id, logged_at", conn)
    model, features_df = train_forecaster(df)
    mae, naive_mae = evaluate_mae(model, features_df)
    print(f"Model MAE: {mae:.2f}  Naive baseline MAE: {naive_mae:.2f}")
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Feature engineering (Section 6.2)
# ---------------------------------------------------------------------------

def build_lag_features(df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
    """Create lag features and a next-sample target per UE.

    For each UE's time series, this creates columns:
      lag_1 ... lag_N  — previous N rsrp_dbm readings
      target           — the next rsrp_dbm reading (what we want to predict)

    Rows where any lag or the target is NaN (first N samples per UE, and
    the last sample per UE) are dropped.

    Args:
        df:     DataFrame with ue_rrc_id, logged_at, rsrp_dbm (sorted).
        n_lags: Number of lag features to create.

    Returns:
        DataFrame with lag columns and target, NaN rows removed.
    """
    df = df.sort_values(["ue_rrc_id", "logged_at"]).copy()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df.groupby("ue_rrc_id")["rsrp_dbm"].shift(lag)
    df["target"] = df.groupby("ue_rrc_id")["rsrp_dbm"].shift(-1)
    return df.dropna(subset=[f"lag_{i}" for i in range(1, n_lags + 1)] + ["target"])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_forecaster(
    df: pd.DataFrame,
    n_lags: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[GradientBoostingRegressor, pd.DataFrame]:
    """Fit a GradientBoostingRegressor on lag features.

    The last 20% of data (by time) is held out as a test set — this mirrors
    how forecasting models are evaluated in practice (no look-ahead).

    Args:
        df:          DataFrame with ue_rrc_id, logged_at, rsrp_dbm.
        n_lags:      Number of lag features.
        test_size:   Fraction of data to hold out for evaluation.
        random_state: RNG seed for reproducibility.

    Returns:
        (fitted_model, features_df)  — features_df has lag cols + target + split column.
    """
    features = build_lag_features(df, n_lags)
    lag_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]

    # Time-aware split: hold out the last test_size fraction by sorted index
    split_idx = int(len(features) * (1 - test_size))
    train_df = features.iloc[:split_idx]
    features["_split"] = "test"
    features.iloc[:split_idx, features.columns.get_loc("_split")] = "train"

    X_train = train_df[lag_cols]
    y_train = train_df["target"]

    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train)

    return model, features


# ---------------------------------------------------------------------------
# Evaluation (MAE vs. naive baseline)
# ---------------------------------------------------------------------------

def evaluate_mae(
    model: GradientBoostingRegressor,
    features_df: pd.DataFrame,
    n_lags: int = 5,
) -> dict[str, float]:
    """Evaluate model MAE against a naive 'predict last value' baseline.

    Uses only the held-out test split (rows with _split == 'test').

    Args:
        model:       Fitted GradientBoostingRegressor.
        features_df: DataFrame returned by train_forecaster (has _split column).
        n_lags:      Number of lag features (must match what model was trained on).

    Returns:
        Dict with 'model_mae' and 'naive_mae'.
    """
    lag_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    test_df = features_df[features_df["_split"] == "test"].copy()

    if test_df.empty:
        return {"model_mae": float("nan"), "naive_mae": float("nan")}

    X_test = test_df[lag_cols]
    y_test = test_df["target"]

    preds = model.predict(X_test)
    naive_preds = test_df["lag_1"]   # naive: last observed value

    model_mae = float(np.mean(np.abs(preds - y_test)))
    naive_mae = float(np.mean(np.abs(naive_preds - y_test)))

    return {"model_mae": model_mae, "naive_mae": naive_mae}


def predict_next(
    model: GradientBoostingRegressor,
    recent_rsrp_values: list[float],
    n_lags: int = 5,
) -> float:
    """Predict the next RSRP given the most recent N readings.

    Args:
        model:              Fitted GradientBoostingRegressor.
        recent_rsrp_values: List of the N most recent rsrp_dbm values
                            (oldest first, newest last).
        n_lags:             Must match model's training n_lags.

    Returns:
        Predicted next rsrp_dbm (float).
    """
    if len(recent_rsrp_values) < n_lags:
        raise ValueError(
            f"Need at least {n_lags} values, got {len(recent_rsrp_values)}"
        )
    # lag_1 = most recent, lag_n = oldest
    lags = list(reversed(recent_rsrp_values[-n_lags:]))
    X = pd.DataFrame([lags], columns=[f"lag_{i}" for i in range(1, n_lags + 1)])
    return float(model.predict(X)[0])


# ---------------------------------------------------------------------------
# Convenience: load + train from DB
# ---------------------------------------------------------------------------

def load_and_train(db_path: str, n_lags: int = 5) -> tuple[GradientBoostingRegressor, pd.DataFrame, dict]:
    """Load ue_samples from DB, train forecaster, evaluate, return everything."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT * FROM ue_samples ORDER BY ue_rrc_id, logged_at",
        conn,
    )
    conn.close()

    model, features_df = train_forecaster(df, n_lags=n_lags)
    metrics = evaluate_mae(model, features_df, n_lags=n_lags)
    return model, features_df, metrics
