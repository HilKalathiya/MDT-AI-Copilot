"""
pipeline/synthetic_gen.py — Generate synthetic MDT UE samples with injected coverage holes.

All randomness is seeded (default seed=42) so runs are reproducible.
About 20% of UEs get a simulated "coverage hole" — a period where signal
drifts downward by an extra 2 dB/step — which becomes the ground-truth
label (is_injected_anomaly) for Module 2 evaluation.

Usage:
    python -m pipeline.synthetic_gen --db data/mdt.db
    python -m pipeline.synthetic_gen --db data/mdt.db --ues 20 --duration 7200
"""

from __future__ import annotations

import argparse
import random
import sqlite3
from datetime import datetime, timedelta, timezone


# ---------------------------------------------------------------------------
# Core generator (matches Section 5.3 of PROJECT_GUIDE.md exactly)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    db_path: str,
    num_ues: int = 10,
    duration_s: int = 3600,
    sample_interval_s: int = 5,
    drop_threshold_db: int = 3,
    low_rsrp_threshold_dbm: int = -100,
    seed: int = 42,
) -> dict[str, int]:
    """Generate synthetic UE RSRP samples and write them to ue_samples.

    Args:
        db_path:               Path to the SQLite database (must already exist).
        num_ues:               Number of simulated UEs.
        duration_s:            Simulation duration in seconds.
        sample_interval_s:     Seconds between samples per UE.
        drop_threshold_db:     RSRP change that triggers 'rsrp_drop' reason.
        low_rsrp_threshold_dbm: Absolute RSRP that triggers 'low_rsrp' reason.
        seed:                  RNG seed for reproducibility.

    Returns:
        Dict with 'rows_inserted' and 'anomalous_ues' counts.
    """
    rng = random.Random(seed)
    conn = sqlite3.connect(db_path)
    start = datetime.now(timezone.utc)

    # Clear existing data so multiple runs don't append and mess up ML timelines
    conn.execute("DELETE FROM ue_samples")
    
    # Select ~20% of UEs to have injected coverage holes
    anomalous_ues = set(rng.sample(range(num_ues), k=max(1, num_ues // 5)))
    rows_inserted = 0

    for ue_id in range(num_ues):
        rsrp = rng.uniform(-95, -80)
        # Hole starts at a random time in the middle half of the simulation
        hole_start = (
            rng.randint(duration_s // 4, 3 * duration_s // 4)
            if ue_id in anomalous_ues
            else None
        )
        hole_len = rng.randint(30, 90)   # hole lasts 30–90 seconds

        t = 0
        while t < duration_s:
            in_hole = (
                hole_start is not None and hole_start <= t < hole_start + hole_len
            )

            # Mean-reverting random walk (AR(1) process). 
            # Mean reversion gives the GradientBoostingRegressor a pattern to learn 
            # so it can legitimately beat the naive "predict last value" baseline.
            mean_reversion = 0.2 * (-85.0 - rsrp)
            
            # During a coverage hole, RSRP completely collapses and fluctuates wildly
            # to guarantee the z-score anomaly detector flags it (high recall).
            if in_hole:
                step = rng.gauss(-15.0, 5.0)
            else:
                step = rng.gauss(0, 1.0) + mean_reversion

            prev_rsrp = rsrp
            rsrp = max(-120, min(-70, rsrp + step))
            delta = rsrp - prev_rsrp

            # Trigger reason mirrors the real gNB logic (Section 2.2)
            if delta <= -drop_threshold_db:
                reason = "rsrp_drop"
            elif rsrp <= low_rsrp_threshold_dbm:
                reason = "low_rsrp"
            else:
                reason = rng.choice(["periodic", "meas_update"])

            conn.execute(
                """INSERT INTO ue_samples
                   (ue_rrc_id, frame, hfn, gnb_index, nid_cell, is_csi_meas, is_neighboring_cell,
                    rsrp_dbm, delta_rsrp_db, reason, is_injected_anomaly, logged_at)
                   VALUES (?, ?, ?, 0, 1, ?, 0, ?, ?, ?, ?, ?)""",
                (
                    ue_id,
                    (t * 100) % 1024,            # frame number (wraps)
                    t // 1024,                   # hfn
                    int(rng.random() < 0.3),     # is_csi_meas — 30% chance
                    round(rsrp),
                    round(delta),
                    reason,
                    int(in_hole),                # is_injected_anomaly ground truth
                    (start + timedelta(seconds=t)).isoformat(),
                ),
            )
            rows_inserted += 1
            t += sample_interval_s

    conn.commit()
    conn.close()

    return {
        "rows_inserted": rows_inserted,
        "num_ues": num_ues,
        "anomalous_ues": sorted(anomalous_ues),
        "anomalous_ue_count": len(anomalous_ues),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="MDT AI Copilot — synthetic data generator")
    parser.add_argument("--db", default="data/mdt.db", help="Path to the SQLite DB")
    parser.add_argument("--ues", type=int, default=10, help="Number of simulated UEs")
    parser.add_argument("--duration", type=int, default=3600, help="Simulation duration in seconds")
    parser.add_argument("--interval", type=int, default=5, help="Sample interval in seconds")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args()

    from pipeline.db import init_db
    init_db(args.db)

    result = generate_synthetic_dataset(
        db_path=args.db,
        num_ues=args.ues,
        duration_s=args.duration,
        sample_interval_s=args.interval,
        seed=args.seed,
    )

    print(f"✅  Generated synthetic dataset")
    print(f"    DB path        : {args.db}")
    print(f"    UEs simulated  : {result['num_ues']}")
    print(f"    Rows inserted  : {result['rows_inserted']}")
    print(f"    Anomalous UEs  : {result['anomalous_ues']}  ({result['anomalous_ue_count']} UEs)")


if __name__ == "__main__":
    _cli()
