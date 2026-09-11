"""
evaluator.py - Comprehensive Evaluation Suite for AI Speed Estimator.
Computes MAE, RMSE, Pearson r, error breakdowns across maneuver states, ZUPT states,
and speed buckets (checking for train/test distribution shifts), and generates plots.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPEED_BUCKETS = [
    ("0.0 - 0.5 m/s (Stationary)", 0.0, 0.5),
    ("0.5 - 5.0 m/s (Low / Stop-Go)", 0.5, 5.0),
    ("5.0 - 15.0 m/s (Urban Arterial)", 5.0, 15.0),
    ("15.0 - 25.0 m/s (High Speed)", 15.0, 25.0),
    ("25.0+ m/s (Highway / Extrap)", 25.0, 100.0),
]

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes MAE, RMSE, Pearson r, Max Error, and Bias."""
    if len(y_true) == 0:
        return {"mae": 0.0, "rmse": 0.0, "pearson_r": 0.0, "max_error": 0.0, "bias": 0.0, "count": 0}
    
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = np.mean(abs_err)
    rmse = np.sqrt(np.mean(err**2))
    bias = np.mean(err)
    max_err = np.max(abs_err)
    
    if np.std(y_true) > 1e-4 and np.std(y_pred) > 1e-4:
        r = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        r = 1.0 if mae < 0.1 else 0.0

    return {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "pearson_r": round(float(r), 4),
        "max_error": round(float(max_err), 4),
        "bias": round(float(bias), 4),
        "count": int(len(y_true))
    }

def evaluate_predictions(
    estimated_df: pd.DataFrame,
    gt_speed_mps: np.ndarray,
    trip_name: str = "Trip"
) -> dict:
    """
    Performs multi-dimensional evaluation of estimated speed vs CAN ground truth:
      - Overall metrics
      - Breakdown by Maneuver State
      - Breakdown by ZUPT State
      - Breakdown by Speed Bucket (identifying distribution shift / extrapolation)
    """
    n = min(len(estimated_df), len(gt_speed_mps))
    y_pred = estimated_df["estimated_speed_mps"].values[:n]
    y_true = gt_speed_mps[:n]
    
    maneuver = estimated_df["maneuver_state"].values[:n]
    zupt = estimated_df["zupt_flag"].values[:n]
    active_source = estimated_df["active_source"].values[:n] if "active_source" in estimated_df.columns else np.array(["ml_model"]*n)

    # 1. Overall Performance
    overall = compute_metrics(y_true, y_pred)
    ml_usage_pct = (active_source == "ml_model").mean() * 100.0

    # 2. Breakdown by Maneuver State
    maneuver_breakdown = {}
    for m in ["stationary", "straight", "gentle_curve", "sharp_turn", "reversal"]:
        mask = (maneuver == m)
        if np.any(mask):
            maneuver_breakdown[m] = compute_metrics(y_true[mask], y_pred[mask])
        else:
            maneuver_breakdown[m] = {"mae": 0.0, "rmse": 0.0, "pearson_r": 0.0, "count": 0}

    # 3. Breakdown by ZUPT Status
    zupt_true_mask = (zupt == True)
    zupt_false_mask = (zupt == False)
    zupt_breakdown = {
        "stationary_zupt_active": compute_metrics(y_true[zupt_true_mask], y_pred[zupt_true_mask]),
        "moving_zupt_inactive": compute_metrics(y_true[zupt_false_mask], y_pred[zupt_false_mask])
    }

    # 4. Breakdown by Speed Bucket (Train/Test distribution shift check)
    bucket_breakdown = {}
    for label, v_min, v_max in SPEED_BUCKETS:
        mask = (y_true >= v_min) & (y_true < v_max)
        if np.any(mask):
            bucket_breakdown[label] = compute_metrics(y_true[mask], y_pred[mask])
        else:
            bucket_breakdown[label] = {"mae": 0.0, "rmse": 0.0, "pearson_r": 0.0, "count": 0}

    return {
        "trip_name": trip_name,
        "overall": overall,
        "ml_usage_pct": round(ml_usage_pct, 2),
        "maneuver_breakdown": maneuver_breakdown,
        "zupt_breakdown": zupt_breakdown,
        "speed_bucket_breakdown": bucket_breakdown
    }

def plot_speed_evaluation(
    estimated_df: pd.DataFrame,
    gt_speed_mps: np.ndarray,
    trip_name: str,
    output_png_path: str,
    max_duration_sec: float = 600.0 # First 10 minutes for clear time-series visualization
):
    """
    Generates high-resolution visualization of estimated speed vs CAN ground truth speed:
      - Subplot 1: Speed trajectories (CAN GT vs AI Predicted with 1-sigma uncertainty band)
      - Subplot 2: Instantaneous error (m/s)
      - Subplot 3: Confidence track & Active Fallback timeline strip
    """
    os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
    
    n = min(len(estimated_df), len(gt_speed_mps))
    n_plot = min(int(max_duration_sec * 10), n)
    
    t = estimated_df["timestamp_s"].values[:n_plot]
    y_pred = estimated_df["estimated_speed_mps"].values[:n_plot]
    y_true = gt_speed_mps[:n_plot]
    std_pred = np.sqrt(estimated_df["speed_variance"].values[:n_plot])
    conf = estimated_df["speed_confidence"].values[:n_plot]
    active_src = estimated_df["active_source"].values[:n_plot] if "active_source" in estimated_df.columns else np.array(["ml_model"]*n_plot)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    
    # Subplot 1: Speed Trajectories
    axes[0].plot(t, y_true, color="#2c3e50", lw=1.8, label="CAN Bus Ground Truth Speed (m/s)")
    axes[0].plot(t, y_pred, color="#e74c3c", lw=1.3, alpha=0.9, label="AI Speed Estimator Output (m/s)")
    axes[0].fill_between(t, np.maximum(0, y_pred - std_pred), y_pred + std_pred, color="#e74c3c", alpha=0.2, label="Estimated Uncertainty (±1σ)")
    axes[0].set_ylabel("Speed (m/s)", fontsize=11, fontweight="bold")
    axes[0].set_title(f"AI Speed Estimator vs CAN Ground Truth — {trip_name}", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", framealpha=0.9)

    # Subplot 2: Error
    err = y_pred - y_true
    axes[1].plot(t, err, color="#8e44ad", lw=1.2, label="Estimation Error: (Pred - GT) m/s")
    axes[1].axhline(0.0, color="black", ls="--", lw=0.8, alpha=0.7)
    axes[1].axhline(1.0, color="red", ls=":", lw=0.8, alpha=0.5)
    axes[1].axhline(-1.0, color="red", ls=":", lw=0.8, alpha=0.5)
    axes[1].fill_between(t, 0, err, color="#9b59b6", alpha=0.25)
    axes[1].set_ylabel("Error (m/s)", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", framealpha=0.9)

    # Subplot 3: Confidence & Fallback Source Strip
    axes[2].plot(t, conf, color="#27ae60", lw=1.4, label="Speed Confidence Score [0, 1]")
    fallback_mask = (active_src == "physics_fallback")
    if np.any(fallback_mask):
        axes[2].scatter(t[fallback_mask], [0.15]*np.sum(fallback_mask), color="#d35400", s=15, label="Active Source: Physics Fallback", zorder=5)
    axes[2].axhline(0.35, color="#e67e22", ls="--", lw=1.0, label="Fallback Threshold (0.35)")
    axes[2].set_ylabel("Confidence", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_png_path, dpi=200)
    plt.close()
