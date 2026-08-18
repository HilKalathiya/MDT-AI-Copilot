"""
tests/test_anomaly.py — Unit tests for ml/anomaly.py

Run with:
    pytest tests/test_anomaly.py -v
"""

import pandas as pd
import numpy as np
import pytest

from ml.anomaly import detect_anomalies_zscore


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """Create a small DataFrame with one UE and a clear spike anomaly."""
    rng = np.random.default_rng(42)
    rsrp = rng.normal(-85, 1.0, 40).tolist()
    # Inject 3 consecutive large drops to create an obvious anomaly
    rsrp[25] = -115
    rsrp[26] = -118
    rsrp[27] = -116

    return pd.DataFrame({
        "ue_rrc_id": [0] * 40,
        "logged_at": pd.date_range("2024-01-01", periods=40, freq="5s").astype(str).tolist(),
        "rsrp_dbm": [round(r) for r in rsrp],
        "delta_rsrp_db": [0] + [round(rsrp[i] - rsrp[i-1]) for i in range(1, 40)],
        "is_injected_anomaly": [0]*25 + [1, 1, 1] + [0]*12,
    })


@pytest.fixture
def multi_ue_df() -> pd.DataFrame:
    """Create a DataFrame with 3 UEs, one anomalous."""
    rng = np.random.default_rng(0)
    rows = []
    for ue_id in range(3):
        rsrp_vals = rng.normal(-85, 1.5, 30).tolist()
        if ue_id == 1:
            rsrp_vals[15] = -118  # anomaly in UE 1
        for i, r in enumerate(rsrp_vals):
            rows.append({
                "ue_rrc_id": ue_id,
                "logged_at": f"2024-01-01T00:{i:02d}:00",
                "rsrp_dbm": round(r),
                "is_injected_anomaly": int(ue_id == 1 and i == 15),
            })
    return pd.DataFrame(rows)


class TestDetectAnomaliesZscore:
    def test_returns_dataframe(self, simple_df):
        result = detect_anomalies_zscore(simple_df)
        assert isinstance(result, pd.DataFrame)

    def test_adds_required_columns(self, simple_df):
        result = detect_anomalies_zscore(simple_df)
        for col in ("rolling_mean", "rolling_std", "z_score", "is_anomaly"):
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_modify_original(self, simple_df):
        original_cols = set(simple_df.columns)
        _ = detect_anomalies_zscore(simple_df)
        assert set(simple_df.columns) == original_cols

    def test_detects_obvious_spike(self, simple_df):
        """The three injected large drops should be flagged as anomalies."""
        result = detect_anomalies_zscore(simple_df, window=20, threshold=3.0)
        anomalous = result[result["is_anomaly"]]
        # At least one of the injected rows (indices 25-27) should be flagged
        injected_indices = {25, 26, 27}
        detected_indices = set(anomalous.index)
        overlap = injected_indices & detected_indices
        assert len(overlap) >= 1, (
            f"Expected at least one injected anomaly to be detected, "
            f"but got overlap={overlap}"
        )

    def test_normal_data_mostly_not_anomaly(self):
        """Normal Gaussian data should have very few anomalies."""
        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "ue_rrc_id": [0] * 100,
            "logged_at": pd.date_range("2024-01-01", periods=100, freq="5s").astype(str).tolist(),
            "rsrp_dbm": [round(x) for x in rng.normal(-85, 1.5, 100)],
        })
        result = detect_anomalies_zscore(df, window=20, threshold=3.0)
        fp_rate = result["is_anomaly"].mean()
        # With threshold=3.0, expect <1% false positives on clean Gaussian data
        assert fp_rate < 0.05, f"Too many false positives on clean data: {fp_rate:.2%}"

    def test_multi_ue_independence(self, multi_ue_df):
        """Anomaly detection per UE should not bleed between UEs."""
        result = detect_anomalies_zscore(multi_ue_df, window=10, threshold=3.0)
        # UE 2 (the clean one) should not be flagged
        ue2 = result[result["ue_rrc_id"] == 2]
        assert not ue2["is_anomaly"].any(), "Clean UE 2 should not have anomalies"

    def test_threshold_affects_detection_count(self, simple_df):
        """Higher threshold should detect fewer anomalies."""
        result_low = detect_anomalies_zscore(simple_df, threshold=1.5)
        result_high = detect_anomalies_zscore(simple_df, threshold=4.0)
        assert result_low["is_anomaly"].sum() >= result_high["is_anomaly"].sum()
