"""
scripts/demo_phase1.py — Phase 1 sanity-check demo.

Runs:
  1. init_db()           — creates the SQLite schema
  2. generate_synthetic_dataset() — populates ue_samples
  3. Prints row counts from both tables

Expected output (default 10 UEs, 1h at 5s intervals):
  ue_samples  : 7200  rows   (10 UEs × 720 samples each)
  mdt_reports : 0     rows   (log parser not run — this is synthetic-only)

Usage:
    python scripts/demo_phase1.py
    python scripts/demo_phase1.py --db data/my_test.db --ues 5
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure project root is on the path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.db import init_db, table_row_counts
from pipeline.synthetic_gen import generate_synthetic_dataset


def run_demo(db_path: str, num_ues: int = 10, duration_s: int = 3600) -> None:
    print("=" * 60)
    print("  MDT AI Copilot — Phase 1 Demo")
    print("=" * 60)

    # Step 1: Init DB
    print(f"\n[1/3] Initialising database at: {db_path}")
    conn = init_db(db_path)
    counts_before = table_row_counts(conn)
    conn.close()
    print(f"      Schema created. Initial counts: {counts_before}")

    # Step 2: Generate synthetic data
    print(f"\n[2/3] Generating synthetic dataset  ({num_ues} UEs, {duration_s}s)")
    result = generate_synthetic_dataset(
        db_path=db_path,
        num_ues=num_ues,
        duration_s=duration_s,
        sample_interval_s=5,
        seed=42,
    )
    print(f"      Rows inserted  : {result['rows_inserted']}")
    print(f"      Anomalous UEs  : {result['anomalous_ues']}")

    # Step 3: Print row counts
    print(f"\n[3/3] Row counts in {db_path}:")
    conn = init_db(db_path)
    counts = table_row_counts(conn)
    conn.close()
    for table, count in counts.items():
        print(f"      {table:<20s}: {count:>6d} rows")

    # Phase 1 definition of done checks
    print("\n" + "=" * 60)
    print("  Phase 1 — Definition of Done Checks")
    print("=" * 60)
    samples = counts["ue_samples"]
    expected_min = num_ues * (duration_s // 5)  # 5s interval
    checks = [
        ("≥10 UEs simulated",        num_ues >= 10),
        ("≥1 hour of samples",        duration_s >= 3600),
        ("≥2 anomalous UEs injected", result["anomalous_ue_count"] >= 2),
        ("ue_samples rows match expected", abs(samples - expected_min) <= num_ues),
    ]
    all_pass = True
    for label, passed in checks:
        status = "✅" if passed else "❌"
        print(f"  {status}  {label}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  🎉  All Phase 1 checks passed!")
    else:
        print("  ⚠️   Some checks failed — review output above.")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="MDT AI Copilot — Phase 1 demo")
    parser.add_argument("--db", default="data/mdt.db", help="Path to the SQLite DB")
    parser.add_argument("--ues", type=int, default=10, help="Number of UEs to simulate")
    parser.add_argument("--duration", type=int, default=3600, help="Simulation duration in seconds")
    args = parser.parse_args()
    run_demo(db_path=args.db, num_ues=args.ues, duration_s=args.duration)


if __name__ == "__main__":
    _cli()
