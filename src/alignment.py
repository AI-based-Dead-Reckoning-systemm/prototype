"""
alignment.py - Fixed Phone-to-Vehicle Alignment Engine.
Transforms sensor measurements from Phone Body Frame to Vehicle Body Frame:
  - Vehicle X: Forward (+ = acceleration, - = braking)
  - Vehicle Y: Left (+ = turning left, centripetal accel)
  - Vehicle Z: Up (vertical, gravity removed)

Assumes dashboard-mounted phone in this axis convention; verified against IO-VNBD CAN ground truth.
Fixed Axis Mapping:
  - Phone gyro_y (column index 16) -> Vehicle Yaw Rate (+Z_v)
  - Phone gyro_x (column index 15) -> Vehicle Pitch Rate (+Y_v)
  - Phone gyro_z (column index 17) -> Vehicle Roll Rate (+X_v)
  - Phone (accel_x, accel_y) rotated by fixed boresight angle psi = 316.0 deg -> Vehicle Forward & Lateral Accel
  - Phone accel_z - 9.80665 -> Vehicle Upward Accel
"""

import numpy as np
import pandas as pd

# Hardcoded fixed boresight angle for dashboard cradle mounting (verified on IO-VNBD dataset)
FIXED_BORESIGHT_DEG = 316.0
FIXED_BORESIGHT_RAD = np.deg2rad(FIXED_BORESIGHT_DEG)

# Fixed 3x3 Orthogonal Rotation Matrix
FIXED_R_P_TO_V = np.array([
    [ np.cos(FIXED_BORESIGHT_RAD),  np.sin(FIXED_BORESIGHT_RAD),  0.0],
    [-np.sin(FIXED_BORESIGHT_RAD),  np.cos(FIXED_BORESIGHT_RAD),  0.0],
    [                         0.0,                          0.0,  1.0]
])


def get_fixed_rotation_matrix() -> np.ndarray:
    """
    Returns the fixed 3x3 Phone-to-Vehicle rotation matrix based on the dashboard mount convention.
    """
    return FIXED_R_P_TO_V.copy()


def apply_alignment(raw_df: pd.DataFrame, R_p_to_v: np.ndarray = None) -> pd.DataFrame:
    """
    Applies fixed coordinate mapping and boresight rotation to produce Vehicle Body Frame signals:
      - accel_vehicle_fwd (m/s²)  [+X: Forward]
      - accel_vehicle_lat (m/s²)  [+Y: Left]
      - accel_vehicle_up  (m/s²)  [+Z: Up, gravity removed]
      - gyro_vehicle_yaw   (rad/s) [around +Z, + = Left turn, from gyro_y]
      - gyro_vehicle_pitch (rad/s) [around +Y, from gyro_x]
      - gyro_vehicle_roll  (rad/s) [around +X, from gyro_z]

    Assumes dashboard-mounted phone in this axis convention; verified against IO-VNBD CAN ground truth.
    """
    if R_p_to_v is None:
        R_p_to_v = FIXED_R_P_TO_V

    ax = raw_df["accel_x"].values
    ay = raw_df["accel_y"].values
    az = raw_df["accel_z"].values

    wx = raw_df["gyro_x"].values
    wy = raw_df["gyro_y"].values
    wz = raw_df["gyro_z"].values

    acc_p = np.vstack([ax, ay, az]).T
    acc_v = (R_p_to_v @ acc_p.T).T

    # Remove 1g gravity from vehicle vertical axis
    acc_v[:, 2] = acc_v[:, 2] - 9.80665

    # Hardcoded fixed Gyroscope Mapping: Phone gyro_y is Vehicle Yaw Rate
    gyro_yaw = wy
    gyro_pitch = wx
    gyro_roll = wz

    aligned_df = pd.DataFrame({
        "accel_vehicle_fwd": np.round(acc_v[:, 0], 4),
        "accel_vehicle_lat": np.round(acc_v[:, 1], 4),
        "accel_vehicle_up": np.round(acc_v[:, 2], 4),
        "gyro_vehicle_yaw": np.round(gyro_yaw, 4),
        "gyro_vehicle_pitch": np.round(gyro_pitch, 4),
        "gyro_vehicle_roll": np.round(gyro_roll, 4),
    })

    return aligned_df


def evaluate_alignment_correlation(aligned_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    """
    Computes Pearson correlation coefficient between estimated vehicle-frame accelerations/rates
    and CAN bus Ground Truth indicated accelerations.
    """
    a_fwd = aligned_df["accel_vehicle_fwd"].values
    a_lat = aligned_df["accel_vehicle_lat"].values
    w_yaw = aligned_df["gyro_vehicle_yaw"].values

    gt_fwd = gt_df["gt_long_accel_mps2"].values
    gt_lat = gt_df["gt_lat_accel_mps2"].values
    gt_yaw = gt_df["gt_yaw_rate_rads"].values

    a_fwd_s = pd.Series(a_fwd).rolling(5, min_periods=1, center=True).mean().values
    a_lat_s = pd.Series(a_lat).rolling(5, min_periods=1, center=True).mean().values
    gt_fwd_s = pd.Series(gt_fwd).rolling(5, min_periods=1, center=True).mean().values
    gt_lat_s = pd.Series(gt_lat).rolling(5, min_periods=1, center=True).mean().values

    dyn_mask = (np.abs(gt_fwd) > 0.4) | (np.abs(gt_lat) > 0.4)

    r_fwd = np.corrcoef(a_fwd, gt_fwd)[0, 1] if np.std(a_fwd) > 1e-4 and np.std(gt_fwd) > 1e-4 else 0.0
    r_lat = np.corrcoef(a_lat, gt_lat)[0, 1] if np.std(a_lat) > 1e-4 and np.std(gt_lat) > 1e-4 else 0.0
    r_yaw = np.corrcoef(w_yaw, gt_yaw)[0, 1] if np.std(w_yaw) > 1e-4 and np.std(gt_yaw) > 1e-4 else 0.0

    r_fwd_s = np.corrcoef(a_fwd_s, gt_fwd_s)[0, 1] if np.std(a_fwd_s) > 1e-4 and np.std(gt_fwd_s) > 1e-4 else 0.0
    r_lat_s = np.corrcoef(a_lat_s, gt_lat_s)[0, 1] if np.std(a_lat_s) > 1e-4 and np.std(gt_lat_s) > 1e-4 else 0.0

    r_fwd_dyn = np.corrcoef(a_fwd_s[dyn_mask], gt_fwd_s[dyn_mask])[0, 1] if dyn_mask.sum() > 10 else r_fwd_s
    r_lat_dyn = np.corrcoef(a_lat_s[dyn_mask], gt_lat_s[dyn_mask])[0, 1] if dyn_mask.sum() > 10 else r_lat_s

    return {
        "pearson_r_yaw_rate": round(float(abs(r_yaw)), 4),
        "pearson_r_longitudinal_raw": round(float(r_fwd), 4),
        "pearson_r_lateral_raw": round(float(r_lat), 4),
        "pearson_r_longitudinal_smoothed": round(float(r_fwd_s), 4),
        "pearson_r_lateral_smoothed": round(float(r_lat_s), 4),
        "pearson_r_longitudinal_dynamic": round(float(r_fwd_dyn), 4),
        "pearson_r_lateral_dynamic": round(float(r_lat_dyn), 4)
    }
