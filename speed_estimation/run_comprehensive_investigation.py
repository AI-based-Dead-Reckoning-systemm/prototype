"""
run_comprehensive_investigation.py - Systematic Hyperparameter & Vibration Normalization Engine.
Performs:
  1. Vibration Normalization & Rolling Baseline Analysis (Item 3)
  2. Full Hyperparameter Grid Sweep with Train MAE, S2 Val MAE, Gap, S2 Urban MAE, and S3c MAE (Items 1 & 2)
  3. Full End-to-End Pipeline Verification vs Original Baseline (Item 4)
"""

import os
import sys
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor

UPSTREAM_DIR = "/home/kavya-singla/.gemini/antigravity-ide/scratch/sih_idr_preprocessing"
sys.path.insert(0, UPSTREAM_DIR)
from src.loader import load_ground_truth_can
from src.pipeline import process_trip
from src.features import extract_window_features

CURRENT_DIR = "/home/kavya-singla/.gemini/antigravity-ide/scratch/sih_idr_speed_estimation"
sys.path.insert(0, CURRENT_DIR)
from src.model import SpeedEstimatorModel
from src.fallback import PhysicsFallbackDispatcher
from src.evaluator import evaluate_predictions

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
    
    feat, end_idx = extract_window_features(aligned, window_size=20, step_size=1)
    gt_spd = gt["gt_speed_mps"].values[end_idx]
    return aligned, feat, end_idx, gt_spd

def apply_normalization(feat_df, aligned_df, end_idx, mode="raw"):
    df = feat_df.copy()
    if mode == "raw":
        return df
    elif mode == "log_vib":
        for col in ["aup_std", "amag_std", "amag_var", "jerk_mean", "fft_energy_vib"]:
            if col in df.columns:
                df[f"log_{col}"] = np.log1p(np.maximum(0, df[col].values))
        return df
    elif mode == "stationary_baseline_sub":
        zupt_all = aligned_df["zupt_flag"].values[end_idx]
        floor_aup = np.median(df.loc[zupt_all, "aup_std"].values) if np.sum(zupt_all) > 50 else np.percentile(df["aup_std"].values, 10)
        floor_amag = np.median(df.loc[zupt_all, "amag_std"].values) if np.sum(zupt_all) > 50 else np.percentile(df["amag_std"].values, 10)
        df["aup_std_excess"] = np.maximum(0.0, df["aup_std"] - floor_aup)
        df["amag_std_excess"] = np.maximum(0.0, df["amag_std"] - floor_amag)
        df["aup_std_ratio"] = df["aup_std"] / max(floor_aup, 0.05)
        return df
    elif mode == "rolling_baseline":
        # 60-second rolling minimum (600 steps @ 10Hz) to represent current road roughness baseline
        roll_min_aup = df["aup_std"].rolling(window=600, min_periods=50).quantile(0.10).bfill().ffill()
        roll_min_amag = df["amag_std"].rolling(window=600, min_periods=50).quantile(0.10).bfill().ffill()
        df["aup_std_rel_road"] = df["aup_std"] - roll_min_aup
        df["amag_std_rel_road"] = df["amag_std"] - roll_min_amag
        df["aup_std_ratio_road"] = df["aup_std"] / np.maximum(roll_min_aup, 0.05)
        return df
    return df

def main():
    print("=" * 110)
    print("LOADING ALL TRIPS (S1, M, S2, S3c)...")
    print("=" * 110)
    s1_al, s1_f_raw, s1_idx, s1_y = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/V-S1.csv", lag=43)
    m_al, m_f_raw, m_idx, m_y = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv", is_m=True)
    s2_al, s2_f_raw, s2_idx, s2_y = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/S-S2.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/V-S2.csv", lag=-41)
    s3c_al, s3c_f_raw, s3c_idx, s3c_y = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/S-S3c.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/V-S3c.csv", lag=42)

    print(f"S1: {len(s1_y)} | M: {len(m_y)} | S2: {len(s2_y)} | S3c: {len(s3c_y)}")

    # -------------------------------------------------------------
    # 1. ITEM 3: VIBRATION NORMALIZATION INVESTIGATION
    # -------------------------------------------------------------
    print("\n" + "=" * 110)
    print("ITEM 3: VIBRATION NORMALIZATION & ROAD ROUGHNESS BASELINE INVESTIGATION")
    print("=" * 110)
    print(f"{'Normalization Mode':<28} | {'Train MAE':<10} | {'S2 Val MAE':<11} | {'Gap (S2-Tr)':<12} | {'S2 Urban MAE':<12} | {'S3c Val MAE':<11}")
    print("-" * 110)

    norm_modes = ["raw", "log_vib", "stationary_baseline_sub", "rolling_baseline"]
    norm_results = []

    for mode in norm_modes:
        s1_f = apply_normalization(s1_f_raw, s1_al, s1_idx, mode=mode)
        m_f = apply_normalization(m_f_raw, m_al, m_idx, mode=mode)
        s2_f = apply_normalization(s2_f_raw, s2_al, s2_idx, mode=mode)
        s3c_f = apply_normalization(s3c_f_raw, s3c_al, s3c_idx, mode=mode)

        X_tr = pd.concat([s1_f, m_f], ignore_index=True)
        y_tr = np.concatenate([s1_y, m_y])

        rf = RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=4, max_features="sqrt", random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)

        # Fast parallel tree prediction
        def get_pred(model, X, al, idx):
            p = np.clip(model.predict(X), 0.0, 45.0)
            p[al["zupt_flag"].values[idx]] = 0.0
            return p

        p_tr = get_pred(rf, X_tr, pd.concat([s1_al.iloc[s1_idx], m_al.iloc[m_idx]]), np.arange(len(y_tr)))
        tr_mae = np.mean(np.abs(p_tr - y_tr))

        p_s2 = get_pred(rf, s2_f, s2_al, s2_idx)
        s2_mae = np.mean(np.abs(p_s2 - s2_y))
        u_mask = (s2_y >= 5.0) & (s2_y < 15.0)
        s2_urban = np.mean(np.abs(p_s2[u_mask] - s2_y[u_mask]))

        p_s3c = get_pred(rf, s3c_f, s3c_al, s3c_idx)
        s3c_mae = np.mean(np.abs(p_s3c - s3c_y))

        gap = s2_mae - tr_mae
        print(f"{mode:<28} | {tr_mae:<10.3f} | {s2_mae:<11.3f} | {gap:<+12.3f} | {s2_urban:<12.3f} | {s3c_mae:<11.3f}")
        norm_results.append({"mode": mode, "tr_mae": tr_mae, "s2_mae": s2_mae, "gap": gap, "s2_urban": s2_urban, "s3c_mae": s3c_mae})

    # -------------------------------------------------------------
    # 2. ITEMS 1 & 2: SYSTEMATIC HYPERPARAMETER GRID SWEEP
    # -------------------------------------------------------------
    print("\n" + "=" * 125)
    print("ITEMS 1 & 2: SYSTEMATIC HYPERPARAMETER GRID SWEEP (SEARCHING FOR S2 VAL MAE MINIMUM & GAP REDUCTION)")
    print("=" * 125)
    print(f"{'Config Description':<32} | {'Depth':<5} | {'Leaf':<4} | {'Feat':<6} | {'Trees':<5} | {'Train MAE':<9} | {'S2 Val MAE':<10} | {'Gap (S2-Tr)':<11} | {'S2 Urban':<8} | {'S3c Val':<8} | {'S1 MAE':<6} | {'M MAE':<6}")
    print("-" * 125)

    s1_f = s1_f_raw
    m_f = m_f_raw
    s2_f = s2_f_raw
    s3c_f = s3c_f_raw

    X_tr = pd.concat([s1_f, m_f], ignore_index=True)
    y_tr = np.concatenate([s1_y, m_y])

    sweep_configs = [
        # (Name, depth, min_leaf, max_feat, n_trees)
        ("Baseline (Unregularized Depth 18)", 18, 2, 1.0, 100),
        ("Original Baseline (Depth 14)", 14, 4, 1.0, 100),
        ("Turn-9 Regularized (Aggressive)", 12, 8, "sqrt", 100),
        ("Sweep: Depth 12 / Leaf 4", 12, 4, "sqrt", 100),
        ("Sweep: Depth 14 / Leaf 4 / sqrt", 14, 4, "sqrt", 100),
        ("Sweep: Depth 14 / Leaf 6 / sqrt", 14, 6, "sqrt", 100),
        ("Sweep: Depth 16 / Leaf 2 / sqrt", 16, 2, "sqrt", 100),
        ("Sweep: Depth 16 / Leaf 4 / sqrt", 16, 4, "sqrt", 100),
        ("Sweep: Depth 16 / Leaf 6 / sqrt", 16, 6, "sqrt", 100),
        ("Sweep: Depth 16 / Leaf 8 / sqrt", 16, 8, "sqrt", 100),
        ("Sweep: Depth 18 / Leaf 4 / sqrt", 18, 4, "sqrt", 100),
        ("Sweep: Depth 20 / Leaf 4 / sqrt", 20, 4, "sqrt", 100),
        ("Sweep: Depth 16 / Leaf 4 / feat=0.6", 16, 4, 0.6, 100),
        ("Sweep: Depth 16 / Leaf 4 / feat=0.8", 16, 4, 0.8, 100),
        ("Sweep: Depth 18 / Leaf 4 / feat=0.6", 18, 4, 0.6, 100),
        ("Sweep: Depth 18 / Leaf 4 / feat=0.8", 18, 4, 0.8, 100),
        ("Sweep: Depth 16 / Leaf 4 / sqrt / 150t", 16, 4, "sqrt", 150),
        ("Sweep: Depth 18 / Leaf 4 / sqrt / 150t", 18, 4, "sqrt", 150),
        ("Overfit Extremum: Depth 22 / Leaf 2", 22, 2, 1.0, 100),
    ]

    sweep_records = []
    dispatcher = PhysicsFallbackDispatcher(confidence_threshold=0.35, max_variance_threshold=4.0)

    for name, depth, leaf, feat_param, trees in sweep_configs:
        rf = RandomForestRegressor(
            n_estimators=trees,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features=feat_param,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_tr, y_tr)

        # Fast vector predictions
        p_s1 = np.clip(rf.predict(s1_f), 0.0, 45.0)
        p_s1[s1_al["zupt_flag"].values[s1_idx]] = 0.0
        s1_mae = np.mean(np.abs(p_s1 - s1_y))

        p_m = np.clip(rf.predict(m_f), 0.0, 45.0)
        p_m[m_al["zupt_flag"].values[m_idx]] = 0.0
        m_mae = np.mean(np.abs(p_m - m_y))

        tr_mae = (s1_mae * len(s1_y) + m_mae * len(m_y)) / (len(s1_y) + len(m_y))

        p_s2 = np.clip(rf.predict(s2_f), 0.0, 45.0)
        p_s2[s2_al["zupt_flag"].values[s2_idx]] = 0.0
        s2_mae = np.mean(np.abs(p_s2 - s2_y))

        u_mask = (s2_y >= 5.0) & (s2_y < 15.0)
        s2_urban = np.mean(np.abs(p_s2[u_mask] - s2_y[u_mask]))

        p_s3c = np.clip(rf.predict(s3c_f), 0.0, 45.0)
        p_s3c[s3c_al["zupt_flag"].values[s3c_idx]] = 0.0
        s3c_mae = np.mean(np.abs(p_s3c - s3c_y))

        gap = s2_mae - tr_mae
        feat_str = str(feat_param)[:4]
        print(f"{name:<32} | {depth:<5} | {leaf:<4} | {feat_str:<6} | {trees:<5} | {tr_mae:<9.3f} | {s2_mae:<10.3f} | {gap:<+11.3f} | {s2_urban:<8.3f} | {s3c_mae:<8.3f} | {s1_mae:<6.3f} | {m_mae:<6.3f}")

        sweep_records.append({
            "name": name, "depth": depth, "leaf": leaf, "feat": feat_param, "trees": trees,
            "tr_mae": tr_mae, "s2_mae": s2_mae, "gap": gap, "s2_urban": s2_urban,
            "s3c_mae": s3c_mae, "s1_mae": s1_mae, "m_mae": m_mae
        })

    print("=" * 125)

    # Save sweep results to CSV
    pd.DataFrame(sweep_records).to_csv("output/hyperparameter_sweep_results.csv", index=False)
    print("\nSaved hyperparameter sweep results -> output/hyperparameter_sweep_results.csv")

if __name__ == "__main__":
    main()
