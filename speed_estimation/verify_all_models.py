"""
verify_all_models.py - Multi-Model Head-to-Head Comparison.
Compares:
1. Original Unregularized Baseline (Turn 8: D=18, L=2, F=1.0, raw features)
2. Turn 9 Aggressive Regularization (Turn 9: D=12, L=8, F=1.0, raw features)
3. Turn 10 Balanced Version (Turn 10: D=16, L=4, F='sqrt', roll60s, old fallback thresholds)
4. New Optimized Production Candidate (D=16, L=2, F=0.5 / 'sqrt', roll20s amag_only, tuned fallback thresholds)
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "preprocessing"))

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
from src.fallback import PhysicsFallbackDispatcher
from src.evaluator import compute_metrics, SPEED_BUCKETS

def get_trip(s_rel, v_rel, lag=0, is_m=False):
    s_p = os.path.join(UPSTREAM_DIR, s_rel)
    v_p = os.path.join(UPSTREAM_DIR, v_rel)
    res = preprocess_trip(s_p, v_p)
    aligned = res["aligned_sample"]
    gt = load_ground_truth_can(v_p)
    n = min(len(aligned), len(gt))
    aligned = aligned.iloc[:n].copy()
    gt = gt.iloc[:n].copy()
    
    if is_m:
        segments = [(0, 42000, 9), (42000, 51000, 21), (51000, n, 32)]
        a_p, g_p = [], []
        for s, e, l in segments:
            e_adj = min(e, n)
            a_p.append(aligned.iloc[s:e_adj - l])
            g_p.append(gt.iloc[s + l:e_adj])
        aligned = pd.concat(a_p, ignore_index=True)
        gt = pd.concat(g_p, ignore_index=True)
    elif lag > 0:
        aligned = aligned.iloc[:-lag].reset_index(drop=True)
        gt = gt.iloc[lag:].reset_index(drop=True)
    elif lag < 0:
        trim = -lag
        aligned = aligned.iloc[trim:].reset_index(drop=True)
        gt = gt.iloc[:-trim].reset_index(drop=True)
    
    return aligned, gt

print("Loading trip data...")
s1_al, s1_gt = get_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/V-S1.csv", lag=43)
m_al, m_gt = get_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv", is_m=True)
s2_al, s2_gt = get_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S2/S-S2.csv" if os.path.exists(os.path.join(UPSTREAM_DIR, "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S2/S-S2.csv")) else "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/S-S2.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/V-S2.csv", lag=-41)
s3c_al, s3c_gt = get_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/S-S3c.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/V-S3c.csv", lag=42)

s1_f_raw, s1_idx = extract_window_features(s1_al, window_size=20, step_size=1)
m_f_raw, m_idx = extract_window_features(m_al, window_size=20, step_size=1)
s2_f_raw, s2_idx = extract_window_features(s2_al, window_size=20, step_size=1)
s3c_f_raw, s3c_idx = extract_window_features(s3c_al, window_size=20, step_size=1)

# Drop old rolling features from raw base if present
cols_to_drop = [c for c in ["aup_std_rel_road", "amag_std_rel_road", "aup_std_ratio_road", "amag_std_ratio_road"] if c in s1_f_raw.columns]
s1_f_base = s1_f_raw.drop(columns=cols_to_drop)
m_f_base = m_f_raw.drop(columns=cols_to_drop)
s2_f_base = s2_f_raw.drop(columns=cols_to_drop)
s3c_f_base = s3c_f_raw.drop(columns=cols_to_drop)

s1_y = s1_gt["gt_speed_mps"].values[s1_idx]
m_y = m_gt["gt_speed_mps"].values[m_idx]
s2_y = s2_gt["gt_speed_mps"].values[s2_idx]
s3c_y = s3c_gt["gt_speed_mps"].values[s3c_idx]

def get_roll_features(base_df, window=200):
    df = base_df.copy()
    r_amag = df["amag_std"].rolling(window, min_periods=30).min().bfill().ffill()
    df["amag_std_rel_road"] = df["amag_std"] - r_amag
    df["amag_std_ratio_road"] = df["amag_std"] / np.maximum(r_amag, 0.05)
    return df

s1_f_roll20 = get_roll_features(s1_f_base, 200)
m_f_roll20 = get_roll_features(m_f_base, 200)
s2_f_roll20 = get_roll_features(s2_f_base, 200)
s3c_f_roll20 = get_roll_features(s3c_f_base, 200)

s1_f_roll60 = get_roll_features(s1_f_base, 600)
m_f_roll60 = get_roll_features(m_f_base, 600)
s2_f_roll60 = get_roll_features(s2_f_base, 600)
s3c_f_roll60 = get_roll_features(s3c_f_base, 600)

models_to_test = [
    {
        "name": "1. Original Unregularized (Turn 8)",
        "features": (s1_f_base, m_f_base, s2_f_base, s3c_f_base),
        "rf": RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=2, max_features=1.0, random_state=42, n_jobs=-1),
        "dispatcher": PhysicsFallbackDispatcher(confidence_threshold=0.35, max_variance_threshold=4.0)
    },
    {
        "name": "2. Over-Regularized (Turn 9)",
        "features": (s1_f_base, m_f_base, s2_f_base, s3c_f_base),
        "rf": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=8, max_features=1.0, random_state=42, n_jobs=-1),
        "dispatcher": PhysicsFallbackDispatcher(confidence_threshold=0.35, max_variance_threshold=4.0)
    },
    {
        "name": "3. Turn 10 Balanced (Old Dispatcher)",
        "features": (s1_f_roll60, m_f_roll60, s2_f_roll60, s3c_f_roll60),
        "rf": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=4, max_features="sqrt", random_state=42, n_jobs=-1),
        "dispatcher": PhysicsFallbackDispatcher(confidence_threshold=0.35, max_variance_threshold=4.0)
    },
    {
        "name": "4. New Candidate A (Roll20s amag, D=16, L=2, F=0.5, Tuned Dispatcher)",
        "features": (s1_f_roll20, m_f_roll20, s2_f_roll20, s3c_f_roll20),
        "rf": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=2, max_features=0.5, random_state=42, n_jobs=-1),
        "dispatcher": PhysicsFallbackDispatcher(confidence_threshold=0.25, max_variance_threshold=16.0)
    },
    {
        "name": "5. New Candidate B (Roll20s amag, D=18, L=4, F=0.5, Tuned Dispatcher)",
        "features": (s1_f_roll20, m_f_roll20, s2_f_roll20, s3c_f_roll20),
        "rf": RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=4, max_features=0.5, random_state=42, n_jobs=-1),
        "dispatcher": PhysicsFallbackDispatcher(confidence_threshold=0.25, max_variance_threshold=16.0)
    },
    {
        "name": "6. New Candidate C (Roll20s amag, D=16, L=4, F=0.7, Tuned Dispatcher)",
        "features": (s1_f_roll20, m_f_roll20, s2_f_roll20, s3c_f_roll20),
        "rf": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=4, max_features=0.7, random_state=42, n_jobs=-1),
        "dispatcher": PhysicsFallbackDispatcher(confidence_threshold=0.25, max_variance_threshold=16.0)
    },
]

def eval_full(model_dict):
    name = model_dict["name"]
    s1_f, m_f, s2_f, s3c_f = model_dict["features"]
    rf = model_dict["rf"]
    dispatcher = model_dict["dispatcher"]

    X_tr = pd.concat([s1_f, m_f], ignore_index=True)
    y_tr = np.concatenate([s1_y, m_y])
    rf.fit(X_tr, y_tr)

    def run_trip(f_df, al_df, idx, y_gt):
        all_tree_preds = np.array(
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(tree.predict)(f_df.values) for tree in rf.estimators_
            )
        )
        pred_mean = np.mean(all_tree_preds, axis=0)
        pred_var = np.var(all_tree_preds, axis=0)
        pred_std = np.sqrt(pred_var)
        pred_conf = 1.0 / (1.0 + (pred_std / 1.5))
        
        n = len(pred_mean)
        zupt = al_df["zupt_flag"].values[idx]
        maneuver = al_df["maneuver_state"].values[idx]
        afwd = al_df["accel_vehicle_fwd"].values[idx]
        timestamps = al_df["timestamp_s"].values[idx]

        bounded_v = np.clip(pred_mean, 0.0, 45.0)
        bounded_v[zupt | (maneuver == "stationary")] = 0.0
        
        prev_v = 0.0
        for i in range(n):
            dv = bounded_v[i] - prev_v
            if abs(dv) > 0.5:
                bounded_v[i] = prev_v + np.sign(dv) * 0.5
            prev_v = bounded_v[i]
            
        model_preds_df = pd.DataFrame({
            "timestamp_s": timestamps,
            "predicted_speed_mps": bounded_v,
            "speed_variance": pred_var,
            "speed_confidence": pred_conf,
            "raw_model_speed": pred_mean
        })

        final_df = dispatcher.apply_fallback(model_preds_df, al_df, idx)
        
        overall_mae = np.mean(np.abs(final_df["estimated_speed_mps"].values - y_gt))
        ml_mask = (final_df["active_source"].values == "ml_model")
        ml_pct = ml_mask.mean() * 100.0
        ml_mae = np.mean(np.abs(final_df.loc[ml_mask, "estimated_speed_mps"].values - y_gt[ml_mask])) if np.any(ml_mask) else 0.0
        
        u_mask = (y_gt >= 5.0) & (y_gt < 15.0)
        urban_mae = np.mean(np.abs(final_df.loc[u_mask, "estimated_speed_mps"].values - y_gt[u_mask])) if np.any(u_mask) else 0.0
        
        return overall_mae, ml_mae, urban_mae, ml_pct

    s1_mae, _, _, _ = run_trip(s1_f, s1_al, s1_idx, s1_y)
    m_mae, _, _, _ = run_trip(m_f, m_al, m_idx, m_y)
    tr_mae = (s1_mae * len(s1_y) + m_mae * len(m_y)) / (len(s1_y) + len(m_y))

    s2_mae, s2_ml_mae, s2_urban, s2_ml_pct = run_trip(s2_f, s2_al, s2_idx, s2_y)
    s3c_mae, s3c_ml_mae, s3c_urban, s3c_ml_pct = run_trip(s3c_f, s3c_al, s3c_idx, s3c_y)
    gap = s2_mae - tr_mae

    return {
        "name": name,
        "train_mae": tr_mae,
        "s1_mae": s1_mae,
        "m_mae": m_mae,
        "s2_val_mae": s2_mae,
        "s2_ml_mae": s2_ml_mae,
        "s2_urban": s2_urban,
        "gap": gap,
        "s3c_val_mae": s3c_mae,
        "s3c_ml_mae": s3c_ml_mae,
        "s3c_ml_pct": s3c_ml_pct
    }

print("\n" + "=" * 135)
print(f"{'Model Version':<42} | {'Train':<7} | {'S1':<6} | {'M':<6} | {'S2 Val':<7} | {'Gap':<7} | {'S2 Urban':<8} | {'S3c Val':<7} | {'S3c ML%':<7} | {'S3c ML MAE':<10}")
print("-" * 135)

comparison_results = []
for m_dict in models_to_test:
    res = eval_full(m_dict)
    comparison_results.append(res)
    print(f"{res['name']:<42} | {res['train_mae']:<7.3f} | {res['s1_mae']:<6.3f} | {res['m_mae']:<6.3f} | {res['s2_val_mae']:<7.3f} | {res['gap']:<+7.3f} | {res['s2_urban']:<8.3f} | {res['s3c_val_mae']:<7.3f} | {res['s3c_ml_pct']:<7.1f}% | {res['s3c_ml_mae']:<10.3f}")

print("=" * 135)
