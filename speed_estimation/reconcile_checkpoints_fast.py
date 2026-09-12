"""
reconcile_checkpoints_fast.py - Rapid high-precision evaluation of historical checkpoints.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

UPSTREAM_DIR = "/home/kavya-singla/.gemini/antigravity-ide/scratch/sih_idr_preprocessing"
CURRENT_DIR = "/home/kavya-singla/.gemini/antigravity-ide/scratch/sih_idr_speed_estimation"
sys.path.insert(0, UPSTREAM_DIR)
sys.path.insert(1, CURRENT_DIR)

from src.loader import load_ground_truth_can
from src.pipeline import process_trip
from src.features import extract_window_features
from src.model import SpeedEstimatorModel
from src.fallback import PhysicsFallbackDispatcher

def load_trip(s_rel, v_rel, lag=0, is_m=False):
    s_p = os.path.join(UPSTREAM_DIR, s_rel)
    v_p = os.path.join(UPSTREAM_DIR, v_rel)
    res = process_trip(s_p, v_p)
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

print("Loading trips...")
s1_al, s1_gt = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/V-S1.csv", lag=43)
m_al, m_gt = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv", is_m=True)
s2_al, s2_gt = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/S-S2.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/V-S2.csv", lag=-41)
s3c_al, s3c_gt = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/S-S3c.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/V-S3c.csv", lag=42)

# Extract 37-feature sets (current production src.features with 20s amag rolling baseline)
s1_f37, s1_idx = extract_window_features(s1_al, 20, 1)
m_f37, m_idx = extract_window_features(m_al, 20, 1)
s2_f37, s2_idx = extract_window_features(s2_al, 20, 1)
s3c_f37, s3c_idx = extract_window_features(s3c_al, 20, 1)

# Raw 35-feature sets (no rolling baseline features)
cols_to_drop = [c for c in ["amag_std_rel_road", "amag_std_ratio_road", "aup_std_rel_road", "aup_std_ratio_road"] if c in s1_f37.columns]
s1_f35 = s1_f37.drop(columns=cols_to_drop)
m_f35 = m_f37.drop(columns=cols_to_drop)
s2_f35 = s2_f37.drop(columns=cols_to_drop)
s3c_f35 = s3c_f37.drop(columns=cols_to_drop)

# Turn 10 38-feature sets (60s dual-axis rolling baseline)
def make_f38(f_base):
    df = f_base.copy()
    r_aup = df["aup_std"].rolling(600, min_periods=50).min().bfill().ffill()
    r_amag = df["amag_std"].rolling(600, min_periods=50).min().bfill().ffill()
    df["aup_std_rel_road"] = df["aup_std"] - r_aup
    df["amag_std_rel_road"] = df["amag_std"] - r_amag
    df["aup_std_ratio_road"] = df["aup_std"] / np.maximum(r_aup, 0.05)
    return df

s1_f38 = make_f38(s1_f35)
m_f38 = make_f38(m_f35)
s2_f38 = make_f38(s2_f35)
s3c_f38 = make_f38(s3c_f35)

s1_y = s1_gt["gt_speed_mps"].values[s1_idx]
m_y = m_gt["gt_speed_mps"].values[m_idx]
s2_y = s2_gt["gt_speed_mps"].values[s2_idx]
s3c_y = s3c_gt["gt_speed_mps"].values[s3c_idx]

old_dispatcher = PhysicsFallbackDispatcher(confidence_threshold=0.35, max_variance_threshold=4.0)
new_dispatcher = PhysicsFallbackDispatcher(confidence_threshold=0.25, max_variance_threshold=16.0)

def evaluate_run(rf_model, X_train, y_train, test_feats, test_als, test_indices, test_ys, dispatcher):
    rf_model.fit(X_train, y_train)
    
    out = {}
    for t_name, (f_df, al_df, idx, y_true) in [
        ("S1", (test_feats[0], test_als[0], test_indices[0], test_ys[0])),
        ("M", (test_feats[1], test_als[1], test_indices[1], test_ys[1])),
        ("S2", (test_feats[2], test_als[2], test_indices[2], test_ys[2])),
        ("S3c", (test_feats[3], test_als[3], test_indices[3], test_ys[3]))
    ]:
        # Fast vector prediction
        preds_all = np.array([tree.predict(f_df.values) for tree in rf_model.estimators_])
        mean_p = np.mean(preds_all, axis=0)
        var_p = np.var(preds_all, axis=0)
        std_p = np.sqrt(var_p)
        conf_p = 1.0 / (1.0 + std_p / 1.5)
        
        n = len(mean_p)
        zupt = al_df["zupt_flag"].values[idx]
        maneuver = al_df["maneuver_state"].values[idx]
        bounded_v = np.clip(mean_p, 0.0, 45.0)
        bounded_v[zupt | (maneuver == "stationary")] = 0.0
        
        prev_v = 0.0
        for i in range(n):
            dv = bounded_v[i] - prev_v
            if abs(dv) > 0.5:
                bounded_v[i] = prev_v + np.sign(dv) * 0.5
            prev_v = bounded_v[i]
            
        model_df = pd.DataFrame({
            "timestamp_s": al_df["timestamp_s"].values[idx],
            "predicted_speed_mps": bounded_v,
            "speed_variance": var_p,
            "speed_confidence": conf_p
        })
        final_df = dispatcher.apply_fallback(model_df, al_df, idx)
        
        preds = final_df["estimated_speed_mps"].values
        mae_overall = np.mean(np.abs(preds - y_true))
        
        ml_mask = (final_df["active_source"].values == "ml_model")
        ml_pct = ml_mask.mean() * 100.0
        ml_mae = np.mean(np.abs(preds[ml_mask] - y_true[ml_mask])) if np.any(ml_mask) else 0.0
        
        u_mask = (y_true >= 5.0) & (y_true < 15.0)
        u_mae = np.mean(np.abs(preds[u_mask] - y_true[u_mask])) if np.any(u_mask) else 0.0
        
        out[t_name] = {
            "overall_mae": mae_overall,
            "ml_mae": ml_mae,
            "urban_mae": u_mae,
            "ml_pct": ml_pct,
            "pure_ml_mae": np.mean(np.abs(bounded_v - y_true))
        }
        
    s1_m = out["S1"]["overall_mae"]
    m_m = out["M"]["overall_mae"]
    tr_mae = (s1_m * len(test_ys[0]) + m_m * len(test_ys[1])) / (len(test_ys[0]) + len(test_ys[1]))
    out["Train_MAE"] = tr_mae
    out["Gap"] = out["S2"]["overall_mae"] - tr_mae
    return out

runs_to_compare = [
    # 1. Turn 8 Baseline
    ("1. Turn 8 Baseline (35-feat raw, D=18, L=2, F=1.0, Old Disp)",
     RandomForestRegressor(n_estimators=100, max_depth=18, min_samples_leaf=2, max_features=1.0, random_state=42, n_jobs=-1),
     (s1_f35, m_f35, s2_f35, s3c_f35), old_dispatcher),

    # 2a. Turn 9 with max_features='sqrt' (35-feat raw)
    ("2a. Turn 9 (35-feat raw, D=12, L=8, F='sqrt', Old Disp)",
     RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=8, max_features="sqrt", random_state=42, n_jobs=-1),
     (s1_f35, m_f35, s2_f35, s3c_f35), old_dispatcher),

    # 2b. Turn 9 with max_features=1.0 (35-feat raw)
    ("2b. Turn 9 (35-feat raw, D=12, L=8, F=1.0, Old Disp)",
     RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=8, max_features=1.0, random_state=42, n_jobs=-1),
     (s1_f35, m_f35, s2_f35, s3c_f35), old_dispatcher),

    # 3a. Turn 10 Balanced Original (38-feat roll60s, D=16, L=4, F='sqrt', Old Disp)
    ("3a. Turn 10 (38-feat roll60s, D=16, L=4, F='sqrt', Old Disp)",
     RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=4, max_features="sqrt", random_state=42, n_jobs=-1),
     (s1_f38, m_f38, s2_f38, s3c_f38), old_dispatcher),

    # 3b. Turn 10 on 37-feat (20s amag)
    ("3b. Turn 10 on 37-feat (D=16, L=4, F='sqrt', Old Disp)",
     RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=4, max_features="sqrt", random_state=42, n_jobs=-1),
     (s1_f37, m_f37, s2_f37, s3c_f37), old_dispatcher),

    # 4. Final Production Model (37-feat 20s amag, D=16, L=2, F=0.5, Tuned Disp)
    ("4. Final Production (37-feat 20s amag, D=16, L=2, F=0.5, Tuned Disp)",
     RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=2, max_features=0.5, random_state=42, n_jobs=-1),
     (s1_f37, m_f37, s2_f37, s3c_f37), new_dispatcher),
]

als = (s1_al, m_al, s2_al, s3c_al)
idxs = (s1_idx, m_idx, s2_idx, s3c_idx)
ys = (s1_y, m_y, s2_y, s3c_y)

print("\n" + "=" * 140)
print(f"{'Run Configuration':<50} | {'Train':<7} | {'S1':<6} | {'M':<6} | {'S2 Val':<7} | {'Gap':<7} | {'S2 Urban':<8} | {'S3c Val':<7} | {'S3c ML%':<7} | {'S3c ML MAE':<10}")
print("-" * 140)

for label, rf, feats, disp in runs_to_compare:
    X_tr = pd.concat([feats[0], feats[1]], ignore_index=True)
    y_tr = np.concatenate([ys[0], ys[1]])
    res = evaluate_run(rf, X_tr, y_tr, feats, als, idxs, ys, disp)
    print(f"{label:<50} | {res['Train_MAE']:<7.3f} | {res['S1']['overall_mae']:<6.3f} | {res['M']['overall_mae']:<6.3f} | {res['S2']['overall_mae']:<7.3f} | {res['Gap']:<+7.3f} | {res['S2']['urban_mae']:<8.3f} | {res['S3c']['overall_mae']:<7.3f} | {res['S3c']['ml_pct']:<7.1f}% | {res['S3c']['ml_mae']:<10.3f}")

print("=" * 140)
