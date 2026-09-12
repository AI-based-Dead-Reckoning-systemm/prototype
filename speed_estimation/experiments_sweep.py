"""
experiments_sweep.py - Feature Engineering & Hyperparameter Optimization Suite.
1. Tests per-trip vibration baseline normalization / ratios.
2. Runs hyperparameter grid sweep (max_depth, min_samples_leaf, n_estimators, max_features).
3. Evaluates Train MAE, S2 (Clean Val) MAE, Generalization Gap, S2 Urban (5-15m/s) MAE, and S3c MAE.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPSTREAM_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "preprocessing"))
sys.path.insert(0, UPSTREAM_DIR)
sys.path.insert(1, CURRENT_DIR)
from src.loader import load_raw_smartphone, load_ground_truth_can
from src.pipeline import process_trip
from src.features import extract_window_features

def load_all_trips():
    def get_trip(s_rel, v_rel, lag=0, is_m=False):
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
        
        feat, end_idx = extract_window_features(aligned, window_size=20, step_size=1)
        gt_spd = gt["gt_speed_mps"].values[end_idx]
        return aligned, feat, end_idx, gt_spd

    print("Extracting features across all trips...")
    s1_al, s1_f, s1_idx, s1_y = get_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/V-S1.csv", lag=43)
    m_al, m_f, m_idx, m_y = get_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv", is_m=True)
    s2_al, s2_f, s2_idx, s2_y = get_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/S-S2.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/V-S2.csv", lag=-41)
    s3c_al, s3c_f, s3c_idx, s3c_y = get_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/S-S3c.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/V-S3c.csv", lag=42)

    return (s1_al, s1_f, s1_idx, s1_y), (m_al, m_f, m_idx, m_y), (s2_al, s2_f, s2_idx, s2_y), (s3c_al, s3c_f, s3c_idx, s3c_y)

def apply_feature_normalization(feat_df: pd.DataFrame, aligned_df: pd.DataFrame, end_idx: np.ndarray, mode: str = "raw") -> pd.DataFrame:
    """
    Applies feature engineering / normalization strategies:
    - 'raw': standard raw features
    - 'log_vib': log-compressed vibration features (compressing road roughness disparity)
    - 'baseline_sub': baseline-subtracted vibration using stationary floor
    - 'ratios': relative vibration ratios
    """
    df = feat_df.copy()
    
    if mode == "raw":
        return df
        
    elif mode == "log_vib":
        # Log-transform vibration and magnitude features
        for col in ["aup_std", "amag_std", "amag_var", "amag_range", "jerk_mean", "jerk_max", "fft_energy_low", "fft_energy_vib", "fft_energy_total"]:
            if col in df.columns:
                df[f"log_{col}"] = np.log1p(np.maximum(0, df[col].values))
        return df

    elif mode == "baseline_sub":
        # Estimate stationary vibration floor for this trip
        zupt_all = aligned_df["zupt_flag"].values[end_idx]
        if np.sum(zupt_all) > 50:
            floor_aup = np.median(df.loc[zupt_all, "aup_std"].values)
            floor_amag = np.median(df.loc[zupt_all, "amag_std"].values)
        else:
            floor_aup = np.percentile(df["aup_std"].values, 10)
            floor_amag = np.percentile(df["amag_std"].values, 10)
            
        df["aup_std_excess"] = np.maximum(0.0, df["aup_std"] - floor_aup)
        df["amag_std_excess"] = np.maximum(0.0, df["amag_std"] - floor_amag)
        df["aup_std_ratio"] = df["aup_std"] / max(floor_aup, 0.05)
        df["amag_std_ratio"] = df["amag_std"] / max(floor_amag, 0.05)
        return df

    elif mode == "combined_enhanced":
        # Combines log-vibration, baseline-subtraction, and kinematic interaction proxies
        zupt_all = aligned_df["zupt_flag"].values[end_idx]
        if np.sum(zupt_all) > 50:
            floor_aup = np.median(df.loc[zupt_all, "aup_std"].values)
            floor_amag = np.median(df.loc[zupt_all, "amag_std"].values)
        else:
            floor_aup = np.percentile(df["aup_std"].values, 10)
            floor_amag = np.percentile(df["amag_std"].values, 10)
            
        df["aup_excess"] = np.maximum(0.0, df["aup_std"] - floor_aup)
        df["amag_excess"] = np.maximum(0.0, df["amag_std"] - floor_amag)
        df["log_aup_std"] = np.log1p(np.maximum(0, df["aup_std"].values))
        df["log_amag_std"] = np.log1p(np.maximum(0, df["amag_std"].values))
        df["log_fft_vib"] = np.log1p(np.maximum(0, df["fft_energy_vib"].values))
        df["log_fft_total"] = np.log1p(np.maximum(0, df["fft_energy_total"].values))
        # Kinematic turn curvature proxy: |alat| / (|wyaw| + 0.05)
        df["kinematic_turn_v_proxy"] = df["alat_abs_mean"] / (df["wyaw_abs_mean"] + 0.05)
        return df

    return df

def run_sweep():
    (s1_al, s1_f_raw, s1_idx, s1_y), (m_al, m_f_raw, m_idx, m_y), (s2_al, s2_f_raw, s2_idx, s2_y), (s3c_al, s3c_f_raw, s3c_idx, s3c_y) = load_all_trips()
    
    print("\n" + "=" * 115)
    print("STEP 1: VIBRATION NORMALIZATION & FEATURE STRATEGY COMPARISON")
    print("=" * 115)
    
    feature_modes = ["raw", "log_vib", "baseline_sub", "combined_enhanced"]
    
    for f_mode in feature_modes:
        s1_f = apply_feature_normalization(s1_f_raw, s1_al, s1_idx, mode=f_mode)
        m_f = apply_feature_normalization(m_f_raw, m_al, m_idx, mode=f_mode)
        s2_f = apply_feature_normalization(s2_f_raw, s2_al, s2_idx, mode=f_mode)
        s3c_f = apply_feature_normalization(s3c_f_raw, s3c_al, s3c_idx, mode=f_mode)
        
        X_tr = pd.concat([s1_f, m_f], ignore_index=True)
        y_tr = np.concatenate([s1_y, m_y])
        
        # Test baseline RF (depth=14, leaf=4, max_feat=1.0)
        rf = RandomForestRegressor(n_estimators=100, max_depth=14, min_samples_leaf=4, max_features=1.0, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        
        # Train MAE
        pred_tr = np.clip(rf.predict(X_tr), 0.0, 45.0)
        train_mae = np.mean(np.abs(pred_tr - y_tr))
        
        # S2 Eval
        pred_s2 = np.clip(rf.predict(s2_f), 0.0, 45.0)
        pred_s2[s2_al["zupt_flag"].values[s2_idx]] = 0.0
        s2_mae = np.mean(np.abs(pred_s2 - s2_y))
        
        u_mask = (s2_y >= 5.0) & (s2_y < 15.0)
        s2_urban_mae = np.mean(np.abs(pred_s2[u_mask] - s2_y[u_mask]))
        
        gap = s2_mae - train_mae
        print(f"Mode: {f_mode:<18} | Train MAE: {train_mae:.3f} | S2 Val MAE: {s2_mae:.3f} | Gap: {gap:+.3f} | S2 Urban (5-15m/s): {s2_urban_mae:.3f}")

    print("\n" + "=" * 115)
    print("STEP 2: HYPERPARAMETER GRID SWEEP (SEARCHING FOR BEST S2 VAL MAE & MINIMAL GAP)")
    print("=" * 115)
    print(f"{'Config Name':<38} | {'Depth':<6} | {'Leaf':<5} | {'Feat':<8} | {'Trees':<6} | {'Train MAE':<10} | {'S2 Val MAE':<11} | {'Gap':<8} | {'S2 Urban':<9} | {'S3c Val':<8}")
    print("-" * 125)

    # Use best feature set: combined_enhanced
    s1_f = apply_feature_normalization(s1_f_raw, s1_al, s1_idx, mode="combined_enhanced")
    m_f = apply_feature_normalization(m_f_raw, m_al, m_idx, mode="combined_enhanced")
    s2_f = apply_feature_normalization(s2_f_raw, s2_al, s2_idx, mode="combined_enhanced")
    s3c_f = apply_feature_normalization(s3c_f_raw, s3c_al, s3c_idx, mode="combined_enhanced")
    
    X_tr = pd.concat([s1_f, m_f], ignore_index=True)
    y_tr = np.concatenate([s1_y, m_y])

    results = []

    grid = [
        # (max_depth, min_samples_leaf, max_features, n_estimators)
        (14, 4, 1.0, 100),       # Original baseline
        (16, 4, 1.0, 100),
        (18, 4, 1.0, 100),
        (20, 4, 1.0, 100),
        (16, 2, 1.0, 100),
        (18, 2, 1.0, 100),
        (16, 4, 0.8, 100),
        (18, 4, 0.8, 100),
        (16, 4, 0.6, 100),
        (18, 4, 0.6, 100),
        (16, 4, "sqrt", 100),
        (18, 4, "sqrt", 100),
        (20, 2, 0.8, 150),
        (18, 2, 0.8, 150),
        (16, 2, 0.8, 150),
        (20, 4, 0.8, 150),
        (18, 4, 0.8, 150),
        (22, 2, 0.8, 150),
    ]

    for depth, leaf, feat_param, trees in grid:
        rf = RandomForestRegressor(
            n_estimators=trees,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features=feat_param,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        
        pred_tr = np.clip(rf.predict(X_tr), 0.0, 45.0)
        tr_mae = np.mean(np.abs(pred_tr - y_tr))
        
        pred_s2 = np.clip(rf.predict(s2_f), 0.0, 45.0)
        pred_s2[s2_al["zupt_flag"].values[s2_idx]] = 0.0
        s2_mae = np.mean(np.abs(pred_s2 - s2_y))
        
        u_mask = (s2_y >= 5.0) & (s2_y < 15.0)
        s2_urban_mae = np.mean(np.abs(pred_s2[u_mask] - s2_y[u_mask]))
        
        pred_s3c = np.clip(rf.predict(s3c_f), 0.0, 45.0)
        pred_s3c[s3c_al["zupt_flag"].values[s3c_idx]] = 0.0
        s3c_mae = np.mean(np.abs(pred_s3c - s3c_y))
        
        gap = s2_mae - tr_mae
        cfg_name = f"RF_d{depth}_l{leaf}_f{str(feat_param)[:4]}_t{trees}"
        
        results.append({
            "cfg_name": cfg_name,
            "depth": depth,
            "leaf": leaf,
            "feat": str(feat_param),
            "trees": trees,
            "tr_mae": tr_mae,
            "s2_mae": s2_mae,
            "gap": gap,
            "s2_urban": s2_urban_mae,
            "s3c_mae": s3c_mae
        })
        
        print(f"{cfg_name:<38} | {depth:<6} | {leaf:<5} | {str(feat_param):<8} | {trees:<6} | {tr_mae:<10.3f} | {s2_mae:<11.3f} | {gap:<+8.3f} | {s2_urban_mae:<9.3f} | {s3c_mae:<8.3f}")

    # Sort results by S2 Val MAE
    res_df = pd.DataFrame(results).sort_values("s2_mae")
    print("\n" + "=" * 115)
    print("TOP 5 BEST PERFORMING MODELS (RANKED BY UNSEEN VALIDATION S2 MAE):")
    print("=" * 115)
    print(res_df.head(5).to_string(index=False))

if __name__ == "__main__":
    run_sweep()
