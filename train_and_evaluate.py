"""
train_and_evaluate.py - Complete Training & Benchmark Suite for AI Speed Estimator.
Trains a unified Random Forest speed regressor with uncertainty on S1 + M,
and evaluates across:
  1. Clean Unseen Validation Trip (Trip S2)
  2. Cradle-Swivel Edge-Case Trip (Trip S3c)
  3. In-Distribution Training Trips (Trip S1, Trip M)
"""

import os
import sys
import importlib.util

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "sih_idr_preprocessing"))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if UPSTREAM_DIR not in sys.path:
    sys.path.insert(1, UPSTREAM_DIR)

# Load upstream preprocessing modules explicitly
loader_spec = importlib.util.spec_from_file_location("upstream_loader", os.path.join(UPSTREAM_DIR, "src", "loader.py"))
upstream_loader = importlib.util.module_from_spec(loader_spec)
loader_spec.loader.exec_module(upstream_loader)
load_raw_smartphone = upstream_loader.load_raw_smartphone
load_ground_truth_can = upstream_loader.load_ground_truth_can

pipeline_spec = importlib.util.spec_from_file_location("upstream_pipeline", os.path.join(UPSTREAM_DIR, "src", "pipeline.py"))
upstream_pipeline = importlib.util.module_from_spec(pipeline_spec)
pipeline_spec.loader.exec_module(upstream_pipeline)
preprocess_trip = upstream_pipeline.process_trip

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features import extract_window_features
from src.model import SpeedEstimatorModel
from src.fallback import PhysicsFallbackDispatcher
from src.evaluator import evaluate_predictions, plot_speed_evaluation, SPEED_BUCKETS

def load_and_preprocess_trip(s_path: str, v_path: str, trip_name: str, lag_correction: bool = False):
    """
    Loads raw phone and CAN ground truth, runs preprocessing alignment, and returns clean DataFrames.
    """
    print(f"Loading & Preprocessing {trip_name}...")
    res = preprocess_trip(s_csv_path=s_path, v_csv_path=v_path)
    aligned_df = res["aligned_sample"]
    gt_df = load_ground_truth_can(v_path)
    
    n = min(len(aligned_df), len(gt_df))
    aligned_df = aligned_df.iloc[:n].copy()
    gt_df = gt_df.iloc[:n].copy()

    if lag_correction and "M" in trip_name:
        # Piecewise lag alignment for Trip M (Android clock drift correction)
        segments = [(0, 42000, 9), (42000, 51000, 21), (51000, n, 32)]
        a_parts, g_parts = [], []
        for s, e, lag in segments:
            e_adj = min(e, n)
            a_parts.append(aligned_df.iloc[s:e_adj - lag])
            g_parts.append(gt_df.iloc[s + lag:e_adj])
        aligned_df = pd.concat(a_parts, ignore_index=True)
        gt_df = pd.concat(g_parts, ignore_index=True)
        print(f"  Applied piecewise lag alignment to {trip_name}: N={len(aligned_df)}")

    return aligned_df, gt_df

def print_speed_distribution_summary(trips_dict: dict):
    """
    Prints comparative summary of vehicle speed distributions across all trips
    to explicitly check for train/test distribution shifts.
    """
    print("\n" + "=" * 80)
    print("SPEED DISTRIBUTION AUDIT (TRAIN VS VALIDATION TRIPS)")
    print("=" * 80)
    print(f"{'Trip Name':<20} | {'Role':<12} | {'Samples':<8} | {'Mean (m/s)':<10} | {'Max (m/s)':<10} | {'Max (km/h)':<10} | {'% Stationary':<12}")
    print("-" * 88)
    
    for name, data in trips_dict.items():
        spd = data["gt_df"]["gt_speed_mps"].values
        stat_pct = (spd < 0.1).mean() * 100.0
        print(f"{name:<20} | {data['role']:<12} | {len(spd):<8d} | {spd.mean():<10.2f} | {spd.max():<10.2f} | {spd.max()*3.6:<10.1f} | {stat_pct:<12.1f}%")

def main():
    print("=" * 80)
    print("AI SPEED ESTIMATOR — TRAINING & MULTI-TRIP VALIDATION SUITE")
    print("=" * 80)

    os.makedirs("models", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # 1. Define Trip Paths
    trip_configs = {
        "Trip S1": {
            "s_path": os.path.join(UPSTREAM_DIR, "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv"),
            "v_path": os.path.join(UPSTREAM_DIR, "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/V-S1.csv"),
            "role": "TRAIN",
            "lag_corr": False
        },
        "Trip M": {
            "s_path": os.path.join(UPSTREAM_DIR, "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv"),
            "v_path": os.path.join(UPSTREAM_DIR, "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv"),
            "role": "TRAIN",
            "lag_corr": True
        },
        "Trip S2": {
            "s_path": os.path.join(UPSTREAM_DIR, "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/S-S2.csv"),
            "v_path": os.path.join(UPSTREAM_DIR, "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/V-S2.csv"),
            "role": "VAL (Clean)",
            "lag_corr": False
        },
        "Trip S3c": {
            "s_path": os.path.join(UPSTREAM_DIR, "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/S-S3c.csv"),
            "v_path": os.path.join(UPSTREAM_DIR, "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/V-S3c.csv"),
            "role": "VAL (Swivel)",
            "lag_corr": False
        }
    }

    # 2. Ingest and Preprocess All Trips
    trips_data = {}
    for name, cfg in trip_configs.items():
        aligned_df, gt_df = load_and_preprocess_trip(
            s_path=cfg["s_path"],
            v_path=cfg["v_path"],
            trip_name=name,
            lag_correction=cfg["lag_corr"]
        )
        trips_data[name] = {
            "aligned_df": aligned_df,
            "gt_df": gt_df,
            "role": cfg["role"]
        }

    # 3. Check Train vs Test Speed Distributions
    print_speed_distribution_summary(trips_data)

    # 4. Extract Multi-Domain Sliding Window Features (2.0s window = 20 samples @ 10Hz)
    window_size = 20
    step_size = 1

    print("\n" + "=" * 80)
    print(f"EXTRACTING FEATURES OVER {window_size*0.1:.1f}s SLIDING WINDOWS (2.0s @ 10Hz, hop=0.1s)")
    print("=" * 80)

    for name, data in trips_data.items():
        feat_df, end_idx = extract_window_features(
            data["aligned_df"],
            window_size=window_size,
            step_size=step_size
        )
        data["features_df"] = feat_df
        data["end_indices"] = end_idx
        data["gt_speed_aligned"] = data["gt_df"]["gt_speed_mps"].values[end_idx]
        print(f"  {name:<10}: Extracted {len(feat_df)} windows ({len(feat_df.columns)} features)")

    # 5. Build Training Matrix (Trip S1 + Trip M)
    X_train = pd.concat([trips_data["Trip S1"]["features_df"], trips_data["Trip M"]["features_df"]], ignore_index=True)
    y_train = np.concatenate([trips_data["Trip S1"]["gt_speed_aligned"], trips_data["Trip M"]["gt_speed_aligned"]])
    print(f"\nTraining Dataset (S1 + M): {len(X_train)} samples across 2 drivers ({len(X_train)*0.1/3600:.2f} hours)")

    # 6. Train Unified AI Speed Estimator Model
    print("\n" + "=" * 80)
    print("TRAINING UNIFIED SPEED ESTIMATOR (RandomForest with Ensemble Uncertainty)")
    print("=" * 80)
    model = SpeedEstimatorModel(
        n_estimators=100,
        max_depth=14,
        min_samples_leaf=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    model_path = "models/speed_estimator_rf.pkl"
    model.save(model_path)
    print(f"Model successfully trained and saved to {model_path}")

    # Feature importances
    importances = model.model.feature_importances_
    feat_imp = pd.Series(importances, index=model.feature_names).sort_values(ascending=False)
    print("\nTop 10 Most Important Features:")
    for f, imp in feat_imp.head(10).items():
        print(f"  {f:<22}: {imp*100:.2f}%")

    # 7. Evaluate Model Across All Trips (with Fallback Dispatcher)
    print("\n" + "=" * 80)
    print("MULTI-TRIP BENCHMARK EVALUATION (MAE, RMSE, Error Breakdowns & Speed Buckets)")
    print("=" * 80)

    dispatcher = PhysicsFallbackDispatcher(confidence_threshold=0.35, max_variance_threshold=4.0)
    eval_summaries = []
    bucket_all_results = {}

    for name, data in trips_data.items():
        # Predict with physics constraints
        preds_df = model.predict_with_physics(
            features_df=data["features_df"],
            aligned_df=data["aligned_df"],
            window_end_indices=data["end_indices"]
        )
        
        # Apply fallback dispatcher
        final_df = dispatcher.apply_fallback(
            model_pred_df=preds_df,
            aligned_df=data["aligned_df"],
            window_end_indices=data["end_indices"]
        )

        # Save output contract
        out_csv = f"output/SpeedEstimates_{name.replace(' ', '_')}.csv"
        final_df.to_csv(out_csv, index=False)
        
        # Evaluate
        eval_res = evaluate_predictions(
            estimated_df=final_df,
            gt_speed_mps=data["gt_speed_aligned"],
            trip_name=name
        )
        data["eval_results"] = eval_res
        data["estimates_df"] = final_df

        # Generate plot
        plot_path = f"plots/0{len(eval_summaries)+1}_speed_estimation_{name.replace(' ', '_')}.png"
        plot_speed_evaluation(
            estimated_df=final_df,
            gt_speed_mps=data["gt_speed_aligned"],
            trip_name=f"{name} ({data['role']})",
            output_png_path=plot_path
        )
        print(f"Saved plot -> {plot_path}")

        ov = eval_res["overall"]
        eval_summaries.append({
            "Trip": name,
            "Role": data["role"],
            "Samples": ov["count"],
            "MAE (m/s)": ov["mae"],
            "RMSE (m/s)": ov["rmse"],
            "Pearson r": ov["pearson_r"],
            "Max Err": ov["max_error"],
            "Bias": ov["bias"],
            "ML Use %": eval_res["ml_usage_pct"]
        })
        bucket_all_results[name] = eval_res["speed_bucket_breakdown"]

    # 8. Print Overall Summary Table
    print("\n" + "=" * 90)
    print("CONSOLIDATED MULTI-TRIP SPEED ESTIMATION BENCHMARKS")
    print("=" * 90)
    summary_df = pd.DataFrame(eval_summaries)
    print(summary_df.to_string(index=False))

    # 9. Print Maneuver Breakdown Table
    print("\n" + "=" * 90)
    print("PERFORMANCE BREAKDOWN BY MANEUVER STATE (MAE in m/s)")
    print("=" * 90)
    maneuver_rows = []
    for name, data in trips_data.items():
        mb = data["eval_results"]["maneuver_breakdown"]
        row = {"Trip": name, "Role": data["role"]}
        for m in ["stationary", "straight", "gentle_curve", "sharp_turn"]:
            row[m] = f"{mb[m]['mae']:.3f} (N={mb[m]['count']})" if mb[m]['count'] > 0 else "N/A"
        maneuver_rows.append(row)
    print(pd.DataFrame(maneuver_rows).to_string(index=False))

    # 10. Print Speed Bucket Breakdown Table (Train/Test Distribution Shift)
    print("\n" + "=" * 90)
    print("PERFORMANCE BREAKDOWN BY SPEED BUCKET (MAE in m/s — Distribution Shift Check)")
    print("=" * 90)
    bucket_rows = []
    for name, data in trips_data.items():
        sb = data["eval_results"]["speed_bucket_breakdown"]
        row = {"Trip": name, "Role": data["role"]}
        for label, _, _ in SPEED_BUCKETS:
            b_info = sb.get(label, {"mae": 0.0, "count": 0})
            row[label.split('(')[0].strip()] = f"{b_info['mae']:.3f} (N={b_info['count']})" if b_info['count'] > 0 else "N/A"
        bucket_rows.append(row)
    print(pd.DataFrame(bucket_rows).to_string(index=False))

    print("\n" + "=" * 80)
    print("ALL RUNS AND BENCHMARKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
