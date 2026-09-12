"""
maneuver.py - Retuned Rule-based Vehicle Maneuver Classifier.
Outputs deterministic categorical maneuver states:
  {stationary, straight, gentle_curve, sharp_turn, reversal}

Boundary Separation:
  - Straight vs Gentle Curve boundary is calibrated against empirical road-banking
    and sensor noise distributions on straight segments:
    - yaw_rate_gentle_thresh = 0.045 rad/s (~2.58 deg/s)
    - lat_acc_gentle_thresh  = 1.00 m/s² (to avoid road camber false positives)
  - Gentle Curve vs Sharp Turn boundary:
    - yaw_rate_sharp_thresh  = 0.120 rad/s (~6.88 deg/s)
    - lat_acc_sharp_thresh   = 1.30 m/s²
"""

import pandas as pd
import numpy as np
from scipy.ndimage import median_filter

def classify_maneuvers(
    aligned_df: pd.DataFrame,
    zupt_flag: pd.Series,
    speed_mps: pd.Series = None,
    yaw_rate_gentle_thresh: float = 0.045, # ~2.58 deg/s (calibrated above straight road noise)
    yaw_rate_sharp_thresh: float = 0.120,  # ~6.88 deg/s (90 deg turn / roundabout)
    lat_acc_gentle_thresh: float = 1.00,   # m/s² (calibrated above road camber / chassis roll)
    lat_acc_sharp_thresh: float = 1.30     # m/s²
) -> pd.Series:
    """
    Classifies each sample into a deterministic maneuver state:
      - stationary: vehicle at complete rest (ZUPT active)
      - reversal: vehicle moving in reverse (speed < -0.4 m/s)
      - straight: forward driving with low lateral dynamics
      - gentle_curve: highway curve or lane change
      - sharp_turn: 90° intersection turn, roundabout, or U-turn
    """
    n = len(aligned_df)
    code_to_name = {0: "stationary", 1: "straight", 2: "gentle_curve", 3: "sharp_turn", 4: "reversal"}
    codes = np.full(n, 1, dtype=int) # default to straight

    # 0.5s smoothed signals to eliminate transient high-frequency vibration spikes
    yaw_rate_raw = np.abs(aligned_df["gyro_vehicle_yaw"].values)
    lat_acc_raw = np.abs(aligned_df["accel_vehicle_lat"].values)

    yaw_rate = pd.Series(yaw_rate_raw).rolling(5, min_periods=1, center=True).mean().values
    lat_acc = pd.Series(lat_acc_raw).rolling(5, min_periods=1, center=True).mean().values
    zupt = zupt_flag.values.astype(bool)

    if speed_mps is not None:
        spd = speed_mps.values
    else:
        spd = np.full(n, 10.0)

    # 1. Stationary condition (ZUPT takes precedence)
    codes[zupt] = 0

    # 2. Reversal condition (if negative speed detected)
    reversal_mask = (~zupt) & (spd < -0.4)
    codes[reversal_mask] = 4

    # 3. Turning conditions (only when moving forward)
    moving_mask = (~zupt) & (~reversal_mask)

    sharp_mask = moving_mask & ((yaw_rate >= yaw_rate_sharp_thresh) | (lat_acc >= lat_acc_sharp_thresh))
    codes[sharp_mask] = 3

    gentle_mask = (
        moving_mask &
        (~sharp_mask) &
        ((yaw_rate >= yaw_rate_gentle_thresh) | (lat_acc >= lat_acc_gentle_thresh))
    )
    codes[gentle_mask] = 2

    # 4. 3-sample median filter to avoid single-sample boundary jitter
    smoothed_codes = median_filter(codes, size=3)
    smoothed_codes[zupt] = 0

    states = [code_to_name[c] for c in smoothed_codes]
    return pd.Series(states, index=aligned_df.index, name="maneuver_state")


def evaluate_maneuvers(maneuver_pred: pd.Series, gt_df: pd.DataFrame) -> dict:
    """
    Validates maneuver predictions against CAN bus Steering Angle & Yaw Rate.
    Removes zero-bias offset from steering angle sensor prior to ground-truth labeling.
    """
    pred = maneuver_pred.values

    gt_spd = gt_df["gt_speed_mps"].values
    gt_yaw = np.abs(gt_df["gt_yaw_rate_rads"].values)
    raw_steer = gt_df["gt_steering_deg"].values
    gt_steer = np.abs(raw_steer - np.median(raw_steer))
    gt_lat = np.abs(gt_df["gt_lat_accel_mps2"].values)

    gt_states = np.full(len(pred), "straight", dtype=object)
    gt_states[gt_spd < 0.1] = "stationary"
    
    gt_moving = gt_spd >= 0.1
    gt_sharp = gt_moving & ((gt_yaw >= 0.120) | (gt_steer >= 40.0) | (gt_lat >= 1.10))
    gt_states[gt_sharp] = "sharp_turn"
    
    gt_gentle = gt_moving & (~gt_sharp) & ((gt_yaw >= 0.040) | (gt_steer >= 12.0) | (gt_lat >= 0.40))
    gt_states[gt_gentle] = "gentle_curve"

    classes = ["stationary", "straight", "gentle_curve", "sharp_turn", "reversal"]
    
    matches = (pred == gt_states)
    overall_acc = np.mean(matches)

    class_stats = {}
    for c in classes:
        gt_c = (gt_states == c)
        pred_c = (pred == c)
        tp = int(np.sum(gt_c & pred_c))
        total_gt = int(np.sum(gt_c))
        total_pred = int(np.sum(pred_c))
        
        recall = tp / total_gt if total_gt > 0 else (1.0 if total_pred == 0 else 0.0)
        prec = tp / total_pred if total_pred > 0 else (1.0 if total_gt == 0 else 0.0)
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0

        class_stats[c] = {
            "gt_count": total_gt,
            "pred_count": total_pred,
            "precision": round(prec, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        }

    return {
        "overall_accuracy": round(float(overall_acc), 4),
        "class_performance": class_stats
    }
