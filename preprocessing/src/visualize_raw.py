"""
visualize_raw.py - Inspects and plots raw IMU signals for stationary baseline vs dynamic driving.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loader import load_iovnbd_smartphone_csv

def plot_stationary_baseline(csv_path: str, out_path: str = "plots/01_stationary_baseline.png"):
    raw = load_iovnbd_smartphone_csv(csv_path)
    
    # Compute norms and moving variance
    acc_norm = np.sqrt(raw["accel_x"]**2 + raw["accel_y"]**2 + raw["accel_z"]**2)
    gyro_norm = np.sqrt(raw["gyro_x"]**2 + raw["gyro_y"]**2 + raw["gyro_z"]**2)
    
    win = 5 # 0.5s at 10Hz
    acc_var = pd.Series(acc_norm).rolling(win, min_periods=1).var().fillna(0)
    gyro_var = pd.Series(gyro_norm).rolling(win, min_periods=1).var().fillna(0)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    # Subplot 1: Raw Accel
    t = raw["timestamp_s"]
    axes[0].plot(t, raw["accel_x"], label="Accel X", color="#e74c3c", alpha=0.7, lw=1)
    axes[0].plot(t, raw["accel_y"], label="Accel Y", color="#2ecc71", alpha=0.7, lw=1)
    axes[0].plot(t, raw["accel_z"], label="Accel Z", color="#3498db", alpha=0.7, lw=1)
    axes[0].plot(t, acc_norm, label="||Accel||", color="#2c3e50", lw=1.5, ls="--")
    axes[0].set_ylabel("Accel (m/s²)")
    axes[0].set_title("Stationary Baseline (S-Vw15) — Phone at Rest")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", ncol=4)
    
    # Subplot 2: Raw Gyro
    axes[1].plot(t, raw["gyro_x"], label="Gyro X", color="#e67e22", alpha=0.7, lw=1)
    axes[1].plot(t, raw["gyro_y"], label="Gyro Y", color="#9b59b6", alpha=0.7, lw=1)
    axes[1].plot(t, raw["gyro_z"], label="Gyro Z", color="#1abc9c", alpha=0.7, lw=1)
    axes[1].plot(t, gyro_norm, label="||Gyro||", color="#34495e", lw=1.5, ls="--")
    axes[1].set_ylabel("Gyro (rad/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", ncol=4)
    
    # Subplot 3: Moving Variances
    axes[2].plot(t, acc_var, label="Accel Norm Var (0.5s win)", color="#c0392b", lw=1.2)
    axes[2].plot(t, gyro_var, label="Gyro Norm Var (0.5s win)", color="#8e44ad", lw=1.2)
    axes[2].axhline(0.02, color="red", ls=":", label="ZUPT Accel Var Thresh (0.02)")
    axes[2].axhline(0.005, color="purple", ls=":", label="ZUPT Gyro Var Thresh (0.005)")
    axes[2].set_ylabel("Variance")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3, which="both")
    axes[2].legend(loc="upper right", ncol=4)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Stationary baseline plot saved to {out_path}")
    print(f"Stationary Stats — Accel Norm Mean: {acc_norm.mean():.4f}, Std: {acc_norm.std():.4f}, Max Acc Var: {acc_var.max():.6f}")
    print(f"Stationary Stats — Gyro Norm Mean: {gyro_norm.mean():.4f}, Std: {gyro_norm.std():.4f}, Max Gyro Var: {gyro_var.max():.6f}")

def plot_driving_clip(csv_path: str, out_path: str = "plots/02_driving_raw_signal.png", max_seconds: float = 300.0):
    raw = load_iovnbd_smartphone_csv(csv_path)
    if max_seconds:
        raw = raw[raw["timestamp_s"] <= max_seconds].copy()
    
    t = raw["timestamp_s"]
    acc_norm = np.sqrt(raw["accel_x"]**2 + raw["accel_y"]**2 + raw["accel_z"]**2)
    gyro_norm = np.sqrt(raw["gyro_x"]**2 + raw["gyro_y"]**2 + raw["gyro_z"]**2)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    
    # 1. GPS Speed
    axes[0].plot(t, raw["gps_speed_mps"], color="#2980b9", lw=1.5, label="GPS Speed (m/s)")
    axes[0].fill_between(t, 0, raw["gps_speed_mps"], color="#3498db", alpha=0.2)
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title(f"Driving Trip Segment (S-S1) — First {max_seconds:.0f}s")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")
    
    # 2. Accel Components
    axes[1].plot(t, raw["accel_x"], label="Phone Accel X", color="#e74c3c", lw=1, alpha=0.7)
    axes[1].plot(t, raw["accel_y"], label="Phone Accel Y", color="#2ecc71", lw=1, alpha=0.7)
    axes[1].plot(t, raw["accel_z"], label="Phone Accel Z", color="#9b59b6", lw=1, alpha=0.7)
    axes[1].set_ylabel("Accel (m/s²)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", ncol=3)
    
    # 3. Gyro Components
    axes[2].plot(t, raw["gyro_x"], label="Phone Gyro X", color="#e67e22", lw=1, alpha=0.7)
    axes[2].plot(t, raw["gyro_y"], label="Phone Gyro Y", color="#1abc9c", lw=1, alpha=0.7)
    axes[2].plot(t, raw["gyro_z"], label="Phone Gyro Z", color="#34495e", lw=1, alpha=0.7)
    axes[2].set_ylabel("Gyro (rad/s)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", ncol=3)
    
    # 4. Moving Variance to highlight stationary vs moving
    win = 5
    acc_var = pd.Series(acc_norm).rolling(win, min_periods=1).var().fillna(0)
    axes[3].plot(t, acc_var, color="#d35400", lw=1.2, label="Accel Norm Variance (0.5s)")
    axes[3].axhline(0.04, color="red", ls="--", label="ZUPT Threshold (0.04 m²/s⁴)")
    axes[3].set_ylabel("Variance")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_yscale("log")
    axes[3].grid(True, alpha=0.3, which="both")
    axes[3].legend(loc="upper right", ncol=2)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Driving clip plot saved to {out_path}")

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    plot_stationary_baseline("data/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-Vw15.csv")
    plot_driving_clip("data/IO-VNBD/Synchronised V abd S datasets/Uncategorised IOVNB Dataset/S-Dataset/S-S1.csv")
