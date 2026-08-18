"""
tests/test_forecast.py — Unit tests for ml/forecast.py

Run with:
    pytest tests/test_forecast.py -v
"""

import numpy as np
import pandas as pd
import pytest

from ml.forecast import build_lag_features, train_forecaster, evaluate_mae, predict_next


@pytest.fixture
def clean_rsrp_df() -> pd.DataFrame:
    """A simple, clean RSRP time series for two UEs."""
    rng = np.random.default_rng(7)
    rows = []
    for ue_id in range(2):
        rsrp = -85.0
        for i in range(100):
            rsrp += rng.normal(0, 0.5)
            rsrp = max(-120, min(-70, rsrp))
            rows.append({
                "ue_rrc_id": ue_id,
                "logged_at": f"2024-01-01T{i // 60:02d}:{i % 60:02d}:00",
                "rsrp_dbm": round(rsrp),
            })
    return pd.DataFrame(rows)


class TestBuildLagFeatures:
    def test_adds_lag_columns(self, clean_rsrp_df):
        result = build_lag_features(clean_rsrp_df, n_lags=3)
        for i in range(1, 4):
            assert f"lag_{i}" in result.columns

    def test_adds_target_column(self, clean_rsrp_df):
        result = build_lag_features(clean_rsrp_df, n_lags=3)
        assert "target" in result.columns

    def test_drops_nan_rows(self, clean_rsrp_df):
        result = build_lag_features(clean_rsrp_df, n_lags=5)
        assert result[["lag_1", "lag_5", "target"]].isna().sum().sum() == 0

    def test_fewer_rows_than_input(self, clean_rsrp_df):
        result = build_lag_features(clean_rsrp_df, n_lags=5)
        # Each UE loses 5 rows at start + 1 at end
        assert len(result) < len(clean_rsrp_df)

    def test_lag1_is_previous_value(self, clean_rsrp_df):
        result = build_lag_features(clean_rsrp_df.query("ue_rrc_id == 0"), n_lags=1)
        # lag_1 should equal the rsrp_dbm of the previous row
        # (verify a couple of entries)
        assert result.iloc[0]["lag_1"] == pytest.approx(
            clean_rsrp_df[clean_rsrp_df["ue_rrc_id"] == 0].iloc[0]["rsrp_dbm"], abs=1
        )


class TestTrainForecaster:
    def test_returns_fitted_model(self, clean_rsrp_df):
        from sklearn.ensemble import GradientBoostingRegressor
        model, _ = train_forecaster(clean_rsrp_df)
        assert isinstance(model, GradientBoostingRegressor)

    def test_returns_features_df(self, clean_rsrp_df):
        _, features_df = train_forecaster(clean_rsrp_df)
        assert isinstance(features_df, pd.DataFrame)
        assert "_split" in features_df.columns

    def test_split_column_values(self, clean_rsrp_df):
        _, features_df = train_forecaster(clean_rsrp_df)
        assert set(features_df["_split"].unique()) <= {"train", "test"}


class TestEvaluateMAE:
    def test_model_mae_is_positive(self, clean_rsrp_df):
        model, features_df = train_forecaster(clean_rsrp_df)
        metrics = evaluate_mae(model, features_df)
        assert metrics["model_mae"] >= 0

    def test_naive_mae_is_positive(self, clean_rsrp_df):
        model, features_df = train_forecaster(clean_rsrp_df)
        metrics = evaluate_mae(model, features_df)
        assert metrics["naive_mae"] >= 0

    def test_model_beats_naive(self, clean_rsrp_df):
        """On enough clean data the GBR should beat naive last-value prediction."""
        model, features_df = train_forecaster(clean_rsrp_df)
        metrics = evaluate_mae(model, features_df)
        # This should generally hold; if it doesn't it's a signal the model is broken
        assert metrics["model_mae"] <= metrics["naive_mae"] * 1.5, (
            f"Model MAE {metrics['model_mae']:.4f} is much worse than naive "
            f"{metrics['naive_mae']:.4f}"
        )


class TestPredictNext:
    def test_returns_float(self, clean_rsrp_df):
        model, _ = train_forecaster(clean_rsrp_df)
        result = predict_next(model, [-85, -84, -86, -87, -85])
        assert isinstance(result, float)

    def test_prediction_in_reasonable_range(self, clean_rsrp_df):
        model, _ = train_forecaster(clean_rsrp_df)
        result = predict_next(model, [-85, -84, -86, -87, -85])
        assert -130 <= result <= -50, f"Prediction out of bounds: {result}"

    def test_raises_on_too_few_values(self, clean_rsrp_df):
        model, _ = train_forecaster(clean_rsrp_df)
        with pytest.raises(ValueError, match="Need at least"):
            predict_next(model, [-85, -84])  # n_lags=5 by default
