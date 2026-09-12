"""
test_untouched_blind_trips.py - True Out-of-Sample Blind Evaluation on Untouched Trips.
Tests the locked production model and fallback dispatcher on:
1. Trip S4 (Driver A - held out)
2. Trip S3a (Driver A - held out)
3. Trip S3b (Driver A - held out)
4. Trip Vfa01 (Driver E - held out driver & vehicle!)
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
UPSTREAM_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "preprocessing"))

def resolve_data_path(rel_path: str) -> str:
    cand1 = os.path.join(ROOT_DIR, rel_path)
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(UPSTREAM_DIR, rel_path)
    if os.path.exists(cand2):
        return cand2
    return cand1

if UPSTREAM_DIR not in sys.path:
    sys.path.insert(0, UPSTREAM_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(1, CURRENT_DIR)

loader_spec = importlib.util.spec_from_file_location("upstream_loader", os.path.join(UPSTREAM_DIR, "src", "loader.py"))
upstream_loader = importlib.util.module_from_spec(loader_spec)
loader_spec.loader.exec_module(upstream_loader)
load_ground_truth_can = upstream_loader.load_ground_truth_can

pipeline_spec = importlib.util.spec_from_file_location("upstream_pipeline", os.path.join(UPSTREAM_DIR, "src", "pipeline.py"))
upstream_pipeline = importlib.util.module_from_spec(pipeline_spec)
pipeline_spec.loader.exec_module(upstream_pipeline)
preprocess_trip = upstream_pipeline.process_trip

from src.features import extract_window_features
from src.model import SpeedEstimatorModel
from src.fallback import PhysicsFallbackDispatcher
from src.evaluator import evaluate_predictions, SPEED_BUCKETS

def evaluate_blind_trip(s_path, v_path, trip_name, model, dispatcher):
    print(f"\n================================================================================")
    print(f"EVALUATING BLIND UNTOUCHED TRIP: {trip_name}")
    print(f"================================================================================")
    
    # 1. Upstream preprocessing
    res = preprocess_trip(s_path, v_path)
    aligned_df = res["aligned_sample"]
    gt_df = load_ground_truth_can(v_path)
    
    n = min(len(aligned_df), len(gt_df))
    aligned_df = aligned_df.iloc[:n].copy()
    gt_df = gt_df.iloc[:n].copy()
    
    print(f"Total aligned samples: {len(aligned_df)} rows ({len(aligned_df)*0.1/60:.1f} minutes)")
    
    # 2. Feature extraction (2.0s window, 20s orientation-invariant rolling baseline)
    features_df, window_end_indices = extract_window_features(aligned_df, window_size=20, step_size=1)
    gt_speed = gt_df["gt_speed_mps"].values[window_end_indices]
    
    # 3. Model inference + Physics post-processing
    model_preds_df = model.predict_with_physics(
        features_df=features_df,
        aligned_df=aligned_df,
        window_end_indices=window_end_indices
    )
    
    # 4. Fallback Dispatcher
    final_df = dispatcher.apply_fallback(
        model_pred_df=model_preds_df,
        aligned_df=aligned_df,
        window_end_indices=window_end_indices
    )
    
    # 5. Evaluate vs Ground Truth CAN
    eval_res = evaluate_predictions(final_df, gt_speed, trip_name=trip_name)
    
    ov = eval_res["overall"]
    sb = eval_res["source_breakdown"]
    mb = eval_res["maneuver_breakdown"]
    bk = eval_res["speed_bucket_breakdown"]
    
    print("\n--- OVERALL METRICS ---")
    print(f"  Overall MAE:      {ov['mae']:.4f} m/s ({ov['mae']*3.6:.2f} km/h)")
    print(f"  Overall RMSE:     {ov['rmse']:.4f} m/s")
    print(f"  Pearson r:        {ov['pearson_r']:.4f}")
    print(f"  Bias:             {ov['bias']:+.4f} m/s")
    print(f"  Max Error:        {ov['max_error']:.4f} m/s")
    print(f"  ML Usage %:       {eval_res['ml_usage_pct']:.1f}%")
    print(f"  ML-Active MAE:    {sb['ml_model']['mae']:.4f} m/s (N={sb['ml_model']['count']})")
    print(f"  Fallback MAE:     {sb['physics_fallback']['mae']:.4f} m/s (N={sb['physics_fallback']['count']})")
    
    print("\n--- MANEUVER BREAKDOWN (MAE in m/s) ---")
    for m in ["stationary", "straight", "gentle_curve", "sharp_turn"]:
        if m in mb and mb[m]["count"] > 0:
            print(f"  {m:<14}: {mb[m]['mae']:.3f} m/s (N={mb[m]['count']}, r={mb[m]['pearson_r']:.3f})")
            
    print("\n--- SPEED BUCKET BREAKDOWN (MAE in m/s) ---")
    for label, _, _ in SPEED_BUCKETS:
        if label in bk and bk[label]["count"] > 0:
            print(f"  {label:<30}: {bk[label]['mae']:.3f} m/s (N={bk[label]['count']})")
            
    return {
        "trip_name": trip_name,
        "samples": ov["count"],
        "overall_mae": ov["mae"],
        "overall_rmse": ov["rmse"],
        "pearson_r": ov["pearson_r"],
        "ml_pct": eval_res["ml_usage_pct"],
        "ml_mae": sb["ml_model"]["mae"],
        "fb_mae": sb["physics_fallback"]["mae"] if sb["physics_fallback"]["count"] > 0 else 0.0,
        "bias": ov["bias"]
    }

def main():
    model_path = os.path.join(CURRENT_DIR, "models", "speed_estimator_rf.pkl")
    print(f"Loading production model from {model_path}...")
    model = SpeedEstimatorModel.load(model_path)
    dispatcher = PhysicsFallbackDispatcher(confidence_threshold=0.25, max_variance_threshold=16.0)
    
    blind_trips = [
        {
            "name": "Trip S4 (Held-out Clean)",
            "s_path": resolve_data_path("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S4/S-S4.csv"),
            "v_path": resolve_data_path("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S4/V-S4.csv")
        },
        {
            "name": "Trip S3a (Held-out)",
            "s_path": resolve_data_path("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/S-S3a.csv"),
            "v_path": resolve_data_path("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3a/V-S3a.csv")
        },
        {
            "name": "Trip S3b (Held-out)",
            "s_path": resolve_data_path("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3b/S-S3b.csv"),
            "v_path": resolve_data_path("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3b/V-S3b.csv")
        },
        {
            "name": "Trip Vfa01 (Held-out Driver E)",
            "s_path": resolve_data_path("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vf (Driver E)/V-Vfa01/S-Vfa01.csv"),
            "v_path": resolve_data_path("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vf (Driver E)/V-Vfa01/V-Vfa01.csv")
        }
    ]
    
    results = []
    for bt in blind_trips:
        if os.path.exists(bt["s_path"]) and os.path.exists(bt["v_path"]):
            r = evaluate_blind_trip(bt["s_path"], bt["v_path"], bt["name"], model, dispatcher)
            results.append(r)
            
    print("\n" + "=" * 115)
    print("CONSOLIDATED SUMMARY OF UNTOUCHED BLIND VALIDATION TRIPS")
    print("=" * 115)
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
