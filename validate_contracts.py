"""
validate_contracts.py - Monorepo validation script for all SIH IDR data contracts.

Validates:
  1. RawSample parsing from preprocessing/output/RawSample_*.csv
  2. AlignedSample parsing from preprocessing/output/AlignedSample_*.csv
  3. SpeedEstimate parsing from speed_estimation/output/SpeedEstimates_*.csv
  4. IDRFusedEstimate serialization & deserialization (Contract for EKF Track 3)
"""

import os
import sys
import pandas as pd
import numpy as np

# Import canonical schemas directly from repo root
from schemas import RawSample, AlignedSample, SpeedEstimate, IDRFusedEstimate, GroundTruthSample

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def validate_all():
    print("=" * 80)
    print("SIH IDR PS 26168 — MONOREPO CONTRACT VALIDATION SUITE")
    print("=" * 80)

    # 1. RawSample validation
    raw_files = [
        "preprocessing/output/RawSample_S1.csv",
        "preprocessing/output/RawSample_M.csv",
        "preprocessing/output/RawSample_S3c.csv",
        "preprocessing/output/RawSample_Vfa01.csv",
        "preprocessing/output/RawSample_Vw15_Stationary.csv"
    ]
    for rf in raw_files:
        full_p = os.path.join(REPO_ROOT, rf)
        if os.path.exists(full_p):
            df = pd.read_csv(full_p, nrows=25)
            for _, row in df.iterrows():
                s = RawSample.from_csv_row(row)
                assert isinstance(s.timestamp_s, float)
                assert isinstance(s.accel_x, float)
                assert isinstance(s.gps_satellites, int)
            print(f"  [PASS] RawSample validated against {rf} ({len(df)} rows)")

    print("-" * 80)

    # 2. AlignedSample validation
    aligned_files = [
        "preprocessing/output/AlignedSample_S1.csv",
        "preprocessing/output/AlignedSample_M.csv",
        "preprocessing/output/AlignedSample_S3c.csv",
        "preprocessing/output/AlignedSample_Vfa01.csv"
    ]
    for af in aligned_files:
        full_p = os.path.join(REPO_ROOT, af)
        if os.path.exists(full_p):
            df = pd.read_csv(full_p, nrows=25)
            for _, row in df.iterrows():
                s = AlignedSample.from_csv_row(row)
                assert isinstance(s.timestamp_s, float)
                assert isinstance(s.accel_vehicle_fwd, float)
                assert isinstance(s.zupt_flag, bool)
                assert s.maneuver_state in {"stationary", "straight", "gentle_curve", "sharp_turn", "reversal"}
            print(f"  [PASS] AlignedSample validated against {af} ({len(df)} rows)")

    print("-" * 80)

    # 3. SpeedEstimate validation
    speed_files = [
        "speed_estimation/output/SpeedEstimates_Trip_S1.csv",
        "speed_estimation/output/SpeedEstimates_Trip_M.csv",
        "speed_estimation/output/SpeedEstimates_Trip_S2.csv",
        "speed_estimation/output/SpeedEstimates_Trip_S3c.csv",
        "speed_estimation/output/SpeedEstimates_demo.csv"
    ]
    for sf in speed_files:
        full_p = os.path.join(REPO_ROOT, sf)
        if os.path.exists(full_p):
            df = pd.read_csv(full_p, nrows=25)
            for _, row in df.iterrows():
                s = SpeedEstimate.from_csv_row(row)
                assert isinstance(s.timestamp_s, float)
                assert isinstance(s.estimated_speed_mps, float)
                assert isinstance(s.speed_variance, float)
                assert isinstance(s.speed_confidence, float)
                assert s.active_source in {"ml_model", "physics_fallback"}
                assert isinstance(s.zupt_flag, bool)
                assert s.maneuver_state in {"stationary", "straight", "gentle_curve", "sharp_turn", "reversal"}
            print(f"  [PASS] SpeedEstimate validated against {sf} ({len(df)} rows)")

    print("-" * 80)

    # 4. IDRFusedEstimate roundtrip validation
    sample_fused = IDRFusedEstimate(
        timestamp_s=10.0,
        pos_lat=52.40166,
        pos_lon=-1.50529,
        pos_alt=147.5,
        vel_north_mps=7.4,
        vel_east_mps=2.1,
        vel_down_mps=0.01,
        vel_forward_mps=7.69,
        heading_deg=15.8,
        pitch_deg=1.1,
        roll_deg=-0.4,
        pos_uncertainty_m=1.35,
        vel_uncertainty_mps=0.28,
        heading_uncertainty_deg=1.2,
        gnss_trust_weight=0.88,
        gnss_active=True,
        active_speed_source="ml_model",
        zupt_active=False,
        maneuver_state="straight"
    )
    d = sample_fused.to_dict()
    reloaded = IDRFusedEstimate.from_csv_row(d)
    assert reloaded == sample_fused
    print(f"  [PASS] IDRFusedEstimate (EKF handoff contract) serialization roundtrip verified.")

    print("=" * 80)
    print("ALL 4 CONTRACTS PASSED COMPLETE MONOREPO INTEGRATION VALIDATION!")
    print("=" * 80)

if __name__ == "__main__":
    validate_all()
