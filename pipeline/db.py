"""
pipeline/db.py — SQLite database initialisation for MDT AI Copilot.

Schema matches Section 5.1 of PROJECT_GUIDE.md exactly.

Usage:
    python -m pipeline.db --init             # creates DB at default path
    python -m pipeline.db --init --path data/custom.db
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


# ---------------------------------------------------------------------------
# Schema (matches Section 5.1 of PROJECT_GUIDE.md exactly)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Reports as received/decoded at the gNB (from real log capture or synthetic)
CREATE TABLE IF NOT EXISTS mdt_reports (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    received_at       TEXT NOT NULL,             -- ISO8601, when the parser ingested this row
    source            TEXT NOT NULL DEFAULT 'sim_log',  -- 'sim_log' | 'synthetic'
    ue_rrc_id         INTEGER NOT NULL,
    report_seq        INTEGER,                   -- the #N from "stored report #N"
    meas_id           INTEGER,
    serving_cell_id   INTEGER,
    serving_rsrp_dbm  INTEGER,
    neighbor_cell_id  INTEGER,
    neighbor_rsrp_dbm INTEGER,
    has_neighbor      INTEGER NOT NULL DEFAULT 0, -- 0/1
    raw_line          TEXT                        -- original log line, for debugging/audit
);
CREATE INDEX IF NOT EXISTS idx_reports_ue   ON mdt_reports(ue_rrc_id);
CREATE INDEX IF NOT EXISTS idx_reports_time ON mdt_reports(received_at);
CREATE INDEX IF NOT EXISTS idx_reports_cell ON mdt_reports(serving_cell_id);

-- UE-side samples (richer — mirrors nr_mdt_sample_t; used by the synthetic generator
-- and by any future real UE-side log capture)
CREATE TABLE IF NOT EXISTS ue_samples (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ue_rrc_id           INTEGER NOT NULL,
    frame               INTEGER,
    hfn                 INTEGER,
    gnb_index           INTEGER,
    nid_cell            INTEGER,
    is_csi_meas         INTEGER NOT NULL DEFAULT 0,
    is_neighboring_cell INTEGER NOT NULL DEFAULT 0,
    rsrp_dbm            INTEGER NOT NULL,
    delta_rsrp_db       INTEGER,
    reason              TEXT,   -- 'enable' | 'rsrp_drop' | 'low_rsrp' | 'meas_update' | 'periodic' | 'none'
    is_injected_anomaly INTEGER NOT NULL DEFAULT 0,  -- ground truth, synthetic data only
    logged_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_ue   ON ue_samples(ue_rrc_id);
CREATE INDEX IF NOT EXISTS idx_samples_time ON ue_samples(logged_at);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Create the MDT database and apply the schema.

    If the database already exists, the schema is applied idempotently
    (all statements use CREATE TABLE/INDEX IF NOT EXISTS).

    Args:
        db_path: Path to the SQLite file (will be created if absent).

    Returns:
        An open sqlite3.Connection to the initialised database.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


def get_connection(db_path: str) -> sqlite3.Connection:
    """Return a connection to an existing database.

    Does NOT run the schema — call init_db() first.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # enables dict-style row access
    return conn


def table_row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Return {table_name: row_count} for the two MDT tables."""
    counts: dict[str, int] = {}
    for table in ("mdt_reports", "ue_samples"):
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = row[0]
    return counts


# ---------------------------------------------------------------------------
# CLI entry point: python -m pipeline.db --init
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="MDT AI Copilot — DB initialisation")
    parser.add_argument("--init", action="store_true", help="Initialise the database schema")
    parser.add_argument("--path", default="data/mdt.db", help="Path to the SQLite file")
    args = parser.parse_args()

    if args.init:
        conn = init_db(args.path)
        counts = table_row_counts(conn)
        print(f"✅  Database initialised at: {args.path}")
        print(f"    mdt_reports : {counts['mdt_reports']} rows")
        print(f"    ue_samples  : {counts['ue_samples']} rows")
        conn.close()
    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
