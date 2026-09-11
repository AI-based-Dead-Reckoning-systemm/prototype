"""
pipeline.py - End-to-end Sensing & Preprocessing Pipeline for SIH IDR.
"""

import os
import pandas as pd
import numpy as np
from src.loader import load_raw_smartphone, load_ground_truth_can
from src.zupt import compute_zupt, evaluate_zupt
from src.alignment import get_fixed_rotation_matrix, apply_alignment, evaluate_alignment_correlation
from src.maneuver import classify_maneuvers, evaluate_maneuvers

def process_trip(
    s_csv_path: str,
    v_csv_path: str = None,
    output_raw_path: str = None,
    output_aligned_path: str = None
) -> dict:
    """
    Runs the complete preprocessing pipeline on a given trip.
    """
    # 1. Ingest Raw Data
    raw_df = load_raw_smartphone(s_csv_path)
    if output_raw_path:
        os.makedirs(os.path.dirname(output_raw_path), exist_ok=True)
        raw_df.to_csv(output_raw_path, index=False)

    # 2. ZUPT Stationary Detection
    zupt_flag = compute_zupt(raw_df)

    # 3. Fixed Phone-to-Vehicle Alignment
    R_p_to_v = get_fixed_rotation_matrix()
    aligned_sensor_df = apply_alignment(raw_df, R_p_to_v)

    # 4. Maneuver Classification
    maneuver_state = classify_maneuvers(
        aligned_sensor_df,
        zupt_flag,
        speed_mps=raw_df["gps_speed_mps"]
    )

    # 5. Assemble AlignedSample Contract
    aligned_sample = pd.DataFrame({
        "timestamp_s": raw_df["timestamp_s"],
        "accel_vehicle_fwd": aligned_sensor_df["accel_vehicle_fwd"],
        "accel_vehicle_lat": aligned_sensor_df["accel_vehicle_lat"],
        "accel_vehicle_up": aligned_sensor_df["accel_vehicle_up"],
        "gyro_vehicle_yaw": aligned_sensor_df["gyro_vehicle_yaw"],
        "gyro_vehicle_pitch": aligned_sensor_df["gyro_vehicle_pitch"],
        "gyro_vehicle_roll": aligned_sensor_df["gyro_vehicle_roll"],
        "zupt_flag": zupt_flag.astype(bool),
        "maneuver_state": maneuver_state,
        "gps_lat": raw_df["gps_lat"],
        "gps_lon": raw_df["gps_lon"],
        "gps_alt": raw_df["gps_alt"],
        "gps_speed_mps": raw_df["gps_speed_mps"],
        "gps_accuracy_m": raw_df["gps_accuracy_m"],
        "gps_heading_deg": raw_df["gps_heading_deg"],
        "gps_satellites": raw_df["gps_satellites"]
    })

    if output_aligned_path:
        os.makedirs(os.path.dirname(output_aligned_path), exist_ok=True)
        aligned_sample.to_csv(output_aligned_path, index=False)

    # 6. Optional Ground-Truth Validation
    validation_results = {}
    if v_csv_path and os.path.exists(v_csv_path):
        gt_df = load_ground_truth_can(v_csv_path)
        
        # Ensure equal length
        min_len = min(len(aligned_sample), len(gt_df))
        aligned_sub = aligned_sample.iloc[:min_len]
        gt_sub = gt_df.iloc[:min_len]

        zupt_metrics = evaluate_zupt(aligned_sub["zupt_flag"], gt_sub["gt_is_stationary"])
        alignment_metrics = evaluate_alignment_correlation(aligned_sensor_df.iloc[:min_len], gt_sub)
        maneuver_metrics = evaluate_maneuvers(aligned_sub["maneuver_state"], gt_sub)

        validation_results = {
            "zupt_validation": zupt_metrics,
            "alignment_validation": alignment_metrics,
            "maneuver_validation": maneuver_metrics
        }

    return {
        "raw_sample": raw_df,
        "aligned_sample": aligned_sample,
        "rotation_matrix": R_p_to_v,
        "validation_results": validation_results
    }
