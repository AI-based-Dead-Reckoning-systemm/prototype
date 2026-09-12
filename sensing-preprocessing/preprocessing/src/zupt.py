"""
zupt.py - Sliding-window Zero Velocity Update (ZUPT) and stationary detector.
"""

import pandas as pd
import numpy as np

def compute_zupt(
    raw_df: pd.DataFrame,
    window_size: int = 5,
    accel_var_thresh: float = 0.015,
    gyro_var_thresh: float = 0.0010,
    gyro_mag_thresh: float = 0.028
) -> pd.Series:
    """
    Computes a boolean ZUPT (stationary) flag using sliding-window variance
    of accelerometer magnitude and gyroscope magnitude.

    Parameters:
    -----------
    raw_df : pd.DataFrame with columns accel_x/y/z, gyro_x/y/z
    window_size : int, number of samples for moving variance (5 samples = 0.5s at 10Hz)
    accel_var_thresh : float, max variance of accel magnitude (m²/s⁴), tuned to 0.015
    gyro_var_thresh : float, max variance of gyro magnitude (rad²/s²), tuned to 0.0010
    gyro_mag_thresh : float, max mean gyro magnitude (rad/s), tuned to 0.028

    Returns:
    --------
    pd.Series of bool: True if vehicle is stationary, False otherwise.
    """
    ax = raw_df["accel_x"].values
    ay = raw_df["accel_y"].values
    az = raw_df["accel_z"].values

    wx = raw_df["gyro_x"].values
    wy = raw_df["gyro_y"].values
    wz = raw_df["gyro_z"].values

    # Compute 3D norms
    acc_norm = np.sqrt(ax**2 + ay**2 + (az - 9.80665)**2)
    gyro_norm = np.sqrt(wx**2 + wy**2 + wz**2)

    # Sliding window variance and mean over 0.5s (5 samples at 10Hz)
    acc_var = pd.Series(acc_norm).rolling(window_size, min_periods=1).var().fillna(0.0).values
    gyro_var = pd.Series(gyro_norm).rolling(window_size, min_periods=1).var().fillna(0.0).values
    gyro_mean = pd.Series(gyro_norm).rolling(window_size, min_periods=1).mean().fillna(0.0).values

    # ZUPT condition
    is_acc_stat = acc_var < accel_var_thresh
    is_gyro_stat = (gyro_var < gyro_var_thresh) & (gyro_mean < gyro_mag_thresh)

    zupt_flag = is_acc_stat & is_gyro_stat
    return pd.Series(zupt_flag, index=raw_df.index, name="zupt_flag")


def evaluate_zupt(zupt_pred: pd.Series, gt_stationary: pd.Series) -> dict:
    """
    Evaluates ZUPT detection performance against ground truth CAN vehicle speed.
    """
    pred = zupt_pred.values.astype(bool)
    gt = gt_stationary.values.astype(bool)

    tp = int(np.sum(pred & gt))
    tn = int(np.sum(~pred & ~gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))

    total = len(gt)
    total_stat = int(np.sum(gt))
    total_moving = int(np.sum(~gt))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / total if total > 0 else 0.0
    fpr = fp / total_moving if total_moving > 0 else 0.0
    fnr = fn / total_stat if total_stat > 0 else 0.0

    return {
        "total_samples": total,
        "stationary_samples": total_stat,
        "moving_samples": total_moving,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4)
    }
