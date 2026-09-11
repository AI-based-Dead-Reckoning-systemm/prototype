"""
loader.py - Robust, standardized loader for IO-VNBD smartphone and CAN Bus datasets.
Produces:
  1. RawSample (Normalized smartphone schema)
  2. GroundTruthSample (Synchronized CAN Bus ground truth)
"""

import os
import re
import pandas as pd
import numpy as np

def load_raw_smartphone(filepath: str) -> pd.DataFrame:
    """
    Loads an IO-VNBD smartphone CSV file (S-*.csv) with latin1 encoding,
    strips header whitespace, standardizes column names, converts units,
    and returns a clean RawSample DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"IO-VNBD smartphone file not found at: {filepath}")

    df = pd.read_csv(filepath, encoding="latin1")

    # Clean header names
    clean_cols = {col: col.strip() for col in df.columns}
    df = df.rename(columns=clean_cols)

    def find_col(patterns):
        for p in patterns:
            for c in df.columns:
                if re.search(p, c, re.IGNORECASE):
                    return c
        return None

    c_lat = find_col([r"^GPS LATITUDE", r"LATITUDE"])
    c_lon = find_col([r"^GPS LONGITUDE", r"LONGITUDE"])
    c_alt = find_col([r"^GPS ALTITUDE", r"ALTITUDE"])
    c_speed = find_col([r"^GPS SPEED", r"SPEED"])
    c_acc = find_col([r"^GPS ACCURACY", r"ACCURACY"])
    c_head = find_col([r"^GPS ORIENTATION", r"ORIENTATION.*AZIMUTH", r"HEADING"])
    c_sats = find_col([r"^GPS SATELLITES", r"SATELLITES"])
    
    c_time_ms = find_col([r"^TIME SINCE START", r"TIME.*MS"])

    c_ax = find_col([r"^ACCELEROMETER X", r"^ACCEL.*X"])
    c_ay = find_col([r"^ACCELEROMETER Y", r"^ACCEL.*Y"])
    c_az = find_col([r"^ACCELEROMETER Z", r"^ACCEL.*Z"])

    c_gx = find_col([r"^GRAVITY X"])
    c_gy = find_col([r"^GRAVITY Y"])
    c_gz = find_col([r"^GRAVITY Z"])

    # Locate gyro columns by index / pattern
    gyro_cols = [c for c in df.columns if re.search(r"GYROSCOPE", c, re.IGNORECASE)]
    if len(gyro_cols) >= 3:
        c_wx, c_wy, c_wz = gyro_cols[0], gyro_cols[1], gyro_cols[2]
    else:
        c_wx = find_col([r"^GYROSCOPE X", r"^GYROSCOPE.*YAW", r"^GYRO.*X"])
        c_wy = find_col([r"^GYROSCOPE Y", r"^GYROSCOPE.*PITCH", r"^GYRO.*Y"])
        c_wz = find_col([r"^GYROSCOPE Z", r"^GYROSCOPE.*ROLL", r"^GYRO.*Z"])

    c_mx = find_col([r"^MAGNETIC FIELD X", r"^MAG.*X"])
    c_my = find_col([r"^MAGNETIC FIELD Y", r"^MAG.*Y"])
    c_mz = find_col([r"^MAGNETIC FIELD Z", r"^MAG.*Z"])

    # Timestamps (seconds from start)
    if c_time_ms is not None:
        t_ms = df[c_time_ms].values.astype(float)
        timestamp_s = (t_ms - t_ms[0]) / 1000.0
    else:
        timestamp_s = np.arange(len(df)) * 0.1

    # Extract satellite count
    if c_sats is not None:
        sat_counts = []
        for val in df[c_sats]:
            if isinstance(val, str) and "/" in val:
                try:
                    sat_counts.append(int(val.split("/")[0].strip()))
                except ValueError:
                    sat_counts.append(0)
            elif pd.notnull(val):
                try:
                    sat_counts.append(int(float(val)))
                except ValueError:
                    sat_counts.append(0)
            else:
                sat_counts.append(0)
        sat_series = np.array(sat_counts)
    else:
        sat_series = np.zeros(len(df), dtype=int)

    # Convert GPS speed (km/h -> m/s)
    if c_speed is not None:
        gps_speed_mps = df[c_speed].fillna(0.0).values / 3.6
    else:
        gps_speed_mps = np.zeros(len(df))

    # Standardized RawSample schema
    raw_sample = pd.DataFrame({
        "timestamp_s": np.round(timestamp_s, 4),
        "accel_x": df[c_ax].astype(float).values if c_ax else np.zeros(len(df)),
        "accel_y": df[c_ay].astype(float).values if c_ay else np.zeros(len(df)),
        "accel_z": df[c_az].astype(float).values if c_az else np.zeros(len(df)),
        "gyro_x": df[c_wx].astype(float).values if c_wx else np.zeros(len(df)),
        "gyro_y": df[c_wy].astype(float).values if c_wy else np.zeros(len(df)),
        "gyro_z": df[c_wz].astype(float).values if c_wz else np.zeros(len(df)),
        "mag_x": df[c_mx].astype(float).values if c_mx else np.zeros(len(df)),
        "mag_y": df[c_my].astype(float).values if c_my else np.zeros(len(df)),
        "mag_z": df[c_mz].astype(float).values if c_mz else np.zeros(len(df)),
        "gravity_x": df[c_gx].astype(float).values if c_gx else np.zeros(len(df)),
        "gravity_y": df[c_gy].astype(float).values if c_gy else np.zeros(len(df)),
        "gravity_z": df[c_gz].astype(float).values if c_gz else np.zeros(len(df)),
        "gps_lat": df[c_lat].astype(float).values if c_lat else np.zeros(len(df)),
        "gps_lon": df[c_lon].astype(float).values if c_lon else np.zeros(len(df)),
        "gps_alt": df[c_alt].astype(float).values if c_alt else np.zeros(len(df)),
        "gps_speed_mps": np.round(gps_speed_mps, 3),
        "gps_accuracy_m": df[c_acc].astype(float).values if c_acc else np.zeros(len(df)),
        "gps_heading_deg": df[c_head].astype(float).values if c_head else np.zeros(len(df)),
        "gps_satellites": sat_series
    })

    return raw_sample


def load_ground_truth_can(filepath: str) -> pd.DataFrame:
    """
    Loads an IO-VNBD CAN Bus Ground Truth file (V-*.csv),
    standardizing units to SI (m/s, rad/s, m/s²).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CAN Bus GT file not found at: {filepath}")

    df = pd.read_csv(filepath, encoding="latin1")
    clean_cols = {col: col.strip() for col in df.columns}
    df = df.rename(columns=clean_cols)

    def find_col(patterns):
        for p in patterns:
            for c in df.columns:
                if re.search(p, c, re.IGNORECASE):
                    return c
        return None

    c_time = find_col([r"^Time Since Start", r"^Time"])
    c_lat = find_col([r"^Latitude"])
    c_lon = find_col([r"^Longitude"])
    c_speed = find_col([r"^Indicated Vehicle Speed", r"^Velocity"])
    c_heading = find_col([r"^Heading"])
    c_yaw_rate = find_col([r"^Yaw Rate"])
    c_steer = find_col([r"^Steering Angle"])
    c_lon_acc = find_col([r"^Indicated Longitudinal Acceleration"])
    c_lat_acc = find_col([r"^Indicated Lateral Acceleration"])

    if c_time is not None:
        t_vals = df[c_time].values.astype(float)
        timestamp_s = t_vals - t_vals[0]
    else:
        timestamp_s = np.arange(len(df)) * 0.1

    speed_mps = (df[c_speed].fillna(0.0).values / 3.6) if c_speed else np.zeros(len(df))
    yaw_rate_rads = np.deg2rad(df[c_yaw_rate].fillna(0.0).values) if c_yaw_rate else np.zeros(len(df))
    G = 9.80665
    lon_acc_mps2 = (df[c_lon_acc].fillna(0.0).values * G) if c_lon_acc else np.zeros(len(df))
    lat_acc_mps2 = (df[c_lat_acc].fillna(0.0).values * G) if c_lat_acc else np.zeros(len(df))

    gt_df = pd.DataFrame({
        "timestamp_s": np.round(timestamp_s, 4),
        "gt_speed_mps": np.round(speed_mps, 3),
        "gt_lat": df[c_lat].astype(float).values if c_lat else np.zeros(len(df)),
        "gt_lon": df[c_lon].astype(float).values if c_lon else np.zeros(len(df)),
        "gt_heading_deg": df[c_heading].astype(float).values if c_heading else np.zeros(len(df)),
        "gt_yaw_rate_rads": np.round(yaw_rate_rads, 4),
        "gt_yaw_rate_degs": df[c_yaw_rate].astype(float).values if c_yaw_rate else np.zeros(len(df)),
        "gt_steering_deg": df[c_steer].astype(float).values if c_steer else np.zeros(len(df)),
        "gt_long_accel_mps2": np.round(lon_acc_mps2, 4),
        "gt_lat_accel_mps2": np.round(lat_acc_mps2, 4),
        "gt_is_stationary": (speed_mps < 0.1).astype(bool)
    })

    return gt_df
