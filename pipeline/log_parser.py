"""
pipeline/log_parser.py — Parse gNB stdout MDT log lines into the mdt_reports table.

Log format (from Section 2.3 of PROJECT_GUIDE.md):
  [MDT][gNB UE <id>] stored report #<seq> measId=<id> serving_cell=<id>
      serving_RSRP=<dBm> dBm neighbor_cell=<id> neighbor_RSRP=<dBm> dBm

NOTE: The regex is built from the *documented* log format, not verified source.
      Re-validate against real captured output before relying on it at scale
      (see Section 2.3 and 5.2 of PROJECT_GUIDE.md for the caveat).

Usage:
    # One-shot parse of a file
    python -m pipeline.log_parser path/to/gnb.log --db data/mdt.db

    # Follow a growing log file (like tail -f)
    python -m pipeline.log_parser path/to/gnb.log --db data/mdt.db --follow
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import time
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Regex pattern (Section 5.2 of PROJECT_GUIDE.md)
# ---------------------------------------------------------------------------

LOG_PATTERN = re.compile(
    r"\[MDT\]\[gNB UE (?P<ue_id>\d+)\] stored report #(?P<report_seq>\d+) "
    r"measId=(?P<meas_id>\d+) serving_cell=(?P<serving_cell>\d+) "
    r"serving_RSRP=(?P<serving_rsrp>-?\d+) dBm "
    r"neighbor_cell=(?P<neighbor_cell>\d+) neighbor_RSRP=(?P<neighbor_rsrp>-?\d+) dBm"
)

# Fallback pattern for lines WITHOUT a neighbor (has_neighbor = false)
# The exact format when has_neighbor is false is not confirmed — this captures
# the serving-cell-only variant and leaves neighbor fields as None.
LOG_PATTERN_NO_NEIGHBOR = re.compile(
    r"\[MDT\]\[gNB UE (?P<ue_id>\d+)\] stored report #(?P<report_seq>\d+) "
    r"measId=(?P<meas_id>\d+) serving_cell=(?P<serving_cell>\d+) "
    r"serving_RSRP=(?P<serving_rsrp>-?\d+) dBm"
)


# ---------------------------------------------------------------------------
# Core parsing functions
# ---------------------------------------------------------------------------

def parse_log_line(line: str) -> dict | None:
    """Parse one gNB stdout line.

    Returns a row dict suitable for INSERT into mdt_reports, or None if
    the line is not an MDT stored-report line.

    The function tries the full pattern (with neighbor) first, then the
    serving-only variant.
    """
    # Try full pattern (neighbor present)
    m = LOG_PATTERN.search(line)
    if m:
        d = m.groupdict()
        neighbor_cell = int(d["neighbor_cell"])
        return {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "source": "sim_log",
            "ue_rrc_id": int(d["ue_id"]),
            "report_seq": int(d["report_seq"]),
            "meas_id": int(d["meas_id"]),
            "serving_cell_id": int(d["serving_cell"]),
            "serving_rsrp_dbm": int(d["serving_rsrp"]),
            "neighbor_cell_id": neighbor_cell,
            "neighbor_rsrp_dbm": int(d["neighbor_rsrp"]),
            # neighbor_cell == 0 is treated as "no neighbor" (assumption — verify
            # against real logs per PROJECT_GUIDE.md Section 5.2 caveat)
            "has_neighbor": 1 if neighbor_cell != 0 else 0,
            "raw_line": line.strip(),
        }

    # Try serving-only pattern (no neighbor)
    m2 = LOG_PATTERN_NO_NEIGHBOR.search(line)
    if m2:
        d = m2.groupdict()
        return {
            "received_at": datetime.now(timezone.utc).isoformat(),
            "source": "sim_log",
            "ue_rrc_id": int(d["ue_id"]),
            "report_seq": int(d["report_seq"]),
            "meas_id": int(d["meas_id"]),
            "serving_cell_id": int(d["serving_cell"]),
            "serving_rsrp_dbm": int(d["serving_rsrp"]),
            "neighbor_cell_id": None,
            "neighbor_rsrp_dbm": None,
            "has_neighbor": 0,
            "raw_line": line.strip(),
        }

    return None


def insert_report(conn: sqlite3.Connection, row: dict) -> None:
    """Insert one parsed row into mdt_reports."""
    conn.execute(
        """INSERT INTO mdt_reports
           (received_at, source, ue_rrc_id, report_seq, meas_id, serving_cell_id,
            serving_rsrp_dbm, neighbor_cell_id, neighbor_rsrp_dbm, has_neighbor, raw_line)
           VALUES (:received_at, :source, :ue_rrc_id, :report_seq, :meas_id, :serving_cell_id,
                   :serving_rsrp_dbm, :neighbor_cell_id, :neighbor_rsrp_dbm, :has_neighbor, :raw_line)""",
        row,
    )
    conn.commit()


def parse_file(log_path: str, conn: sqlite3.Connection) -> int:
    """Parse an entire log file from the beginning. Returns count of rows inserted."""
    inserted = 0
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            row = parse_log_line(line)
            if row:
                insert_report(conn, row)
                inserted += 1
    return inserted


def tail_and_ingest(log_path: str, db_path: str, poll_interval: float = 1.0) -> None:
    """Follow a growing log file (like `tail -f`) and insert parsed rows as they appear.

    This function blocks indefinitely. Interrupt with Ctrl-C.
    """
    conn = sqlite3.connect(db_path)
    print(f"📡  Tailing {log_path}  →  {db_path}")
    print("    Press Ctrl-C to stop.\n")

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)  # start at end of file — only ingest NEW lines
        try:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(poll_interval)
                    continue
                row = parse_log_line(line)
                if row:
                    insert_report(conn, row)
                    print(f"  ↳  UE {row['ue_rrc_id']}  "
                          f"cell {row['serving_cell_id']}  "
                          f"{row['serving_rsrp_dbm']} dBm")
        except KeyboardInterrupt:
            print("\n⏹   Stopped.")
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="MDT AI Copilot — log parser")
    parser.add_argument("log_path", help="Path to the gNB log file")
    parser.add_argument("--db", default="data/mdt.db", help="Path to the SQLite DB")
    parser.add_argument("--follow", action="store_true",
                        help="Follow the file like tail -f (blocks)")
    args = parser.parse_args()

    if args.follow:
        tail_and_ingest(args.log_path, args.db)
    else:
        from pipeline.db import init_db
        conn = init_db(args.db)
        count = parse_file(args.log_path, conn)
        print(f"✅  Inserted {count} MDT report rows into {args.db}")
        conn.close()


if __name__ == "__main__":
    _cli()
