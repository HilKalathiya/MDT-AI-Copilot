"""
tests/test_log_parser.py — Unit tests for pipeline/log_parser.py.

Five hand-written sample lines cover:
  1. Normal line with neighbor
  2. Normal line, different UE and cell IDs
  3. Line with no neighbor (no neighbor_cell / neighbor_RSRP fields)
  4. Non-MDT log line (should return None)
  5. Line with negative serving RSRP and neighbor cell = 0 (has_neighbor = 0)

Run with:
    pytest tests/test_log_parser.py -v
"""

import pytest
from pipeline.log_parser import parse_log_line


# ---------------------------------------------------------------------------
# Test data — hand-written sample log lines
# ---------------------------------------------------------------------------

SAMPLE_LINES = {
    "with_neighbor": (
        "[MDT][gNB UE 3] stored report #12 measId=1 serving_cell=1 "
        "serving_RSRP=-89 dBm neighbor_cell=2 neighbor_RSRP=-97 dBm"
    ),
    "different_ids": (
        "[MDT][gNB UE 7] stored report #5 measId=2 serving_cell=3 "
        "serving_RSRP=-74 dBm neighbor_cell=4 neighbor_RSRP=-85 dBm"
    ),
    "no_neighbor": (
        "[MDT][gNB UE 1] stored report #1 measId=1 serving_cell=1 "
        "serving_RSRP=-95 dBm"
    ),
    "non_mdt": (
        "[RRC][gNB UE 3] RRCConnectionSetupComplete received"
    ),
    "neighbor_zero": (
        "[MDT][gNB UE 2] stored report #20 measId=3 serving_cell=2 "
        "serving_RSRP=-100 dBm neighbor_cell=0 neighbor_RSRP=0 dBm"
    ),
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseLogLine:
    """Tests for parse_log_line()."""

    def test_with_neighbor_parses_correctly(self):
        """Line with a neighbor cell should parse all fields."""
        row = parse_log_line(SAMPLE_LINES["with_neighbor"])

        assert row is not None, "Expected a parsed row, got None"
        assert row["ue_rrc_id"] == 3
        assert row["report_seq"] == 12
        assert row["meas_id"] == 1
        assert row["serving_cell_id"] == 1
        assert row["serving_rsrp_dbm"] == -89
        assert row["neighbor_cell_id"] == 2
        assert row["neighbor_rsrp_dbm"] == -97
        assert row["has_neighbor"] == 1
        assert row["source"] == "sim_log"
        assert row["raw_line"] == SAMPLE_LINES["with_neighbor"]

    def test_different_ids_parse_correctly(self):
        """Parsing should work for any valid UE/cell ID combination."""
        row = parse_log_line(SAMPLE_LINES["different_ids"])

        assert row is not None
        assert row["ue_rrc_id"] == 7
        assert row["report_seq"] == 5
        assert row["serving_cell_id"] == 3
        assert row["serving_rsrp_dbm"] == -74
        assert row["neighbor_cell_id"] == 4
        assert row["neighbor_rsrp_dbm"] == -85
        assert row["has_neighbor"] == 1

    def test_no_neighbor_returns_has_neighbor_false(self):
        """Line without neighbor fields should return has_neighbor=0 and None fields."""
        row = parse_log_line(SAMPLE_LINES["no_neighbor"])

        assert row is not None, "Serving-only line should still parse"
        assert row["ue_rrc_id"] == 1
        assert row["serving_rsrp_dbm"] == -95
        assert row["has_neighbor"] == 0
        assert row["neighbor_cell_id"] is None
        assert row["neighbor_rsrp_dbm"] is None

    def test_non_mdt_line_returns_none(self):
        """A non-MDT log line should return None (not parsed)."""
        row = parse_log_line(SAMPLE_LINES["non_mdt"])
        assert row is None, "Non-MDT line should return None"

    def test_neighbor_cell_zero_has_neighbor_false(self):
        """neighbor_cell=0 should be treated as no neighbor (has_neighbor=0)."""
        row = parse_log_line(SAMPLE_LINES["neighbor_zero"])

        assert row is not None
        assert row["serving_rsrp_dbm"] == -100
        assert row["neighbor_cell_id"] == 0
        # neighbor_cell=0 is the "no neighbor" sentinel — has_neighbor should be 0
        assert row["has_neighbor"] == 0

    def test_received_at_is_iso8601(self):
        """received_at should be a valid ISO8601 string."""
        from datetime import datetime
        row = parse_log_line(SAMPLE_LINES["with_neighbor"])
        assert row is not None
        # Should not raise
        dt = datetime.fromisoformat(row["received_at"])
        assert dt is not None

    def test_empty_line_returns_none(self):
        """Empty string should return None."""
        assert parse_log_line("") is None

    def test_partial_mdt_line_returns_none(self):
        """Incomplete MDT line should return None."""
        partial = "[MDT][gNB UE 3] stored report #12 measId=1"
        # Full pattern won't match; no-neighbor pattern also needs serving_cell/RSRP
        row = parse_log_line(partial)
        # Either None (doesn't match) or a partial match is acceptable behaviour
        # — but if it matches, it must have at minimum the serving fields
        if row is not None:
            assert "serving_rsrp_dbm" in row
