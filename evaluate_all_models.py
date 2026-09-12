import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

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

def main():
    print("Loading all datasets...", flush=True)
    s1_al, s1_f, s1_idx, s1_y = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/V-S1.csv", lag=43)
    m_al, m_f, m_idx, m_y = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv", is_m=True)
    s2_al, s2_f, s2_idx, s2_y = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/S-S2.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/V-S2.csv", lag=-41)
    s3c_al, s3c_f, s3c_idx, s3c_y = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/S-S3c.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/V-S3c.csv", lag=42)

    X_tr = pd.concat([s1_f, m_f], ignore_index=True)
    y_tr = np.concatenate([s1_y, m_y])

    dispatcher = PhysicsFallbackDispatcher(confidence_threshold=0.35, max_variance_threshold=4.0)

    # Systematic Grid
    grid = [
        # (Name, max_depth, min_samples_leaf, max_features, n_estimators)
        ("Baseline (Unregularized)", 18, 2, 1.0, 100),
        ("Turn-9 Regularized (Aggressive)", 12, 8, "sqrt", 100),
        ("Sweep Candidate 1", 16, 4, 1.0, 100),
        ("Sweep Candidate 2", 18, 4, 0.8, 100),
        ("Sweep Candidate 3", 20, 2, 0.8, 100),
        ("Sweep Candidate 4", 20, 4, 0.8, 150),
        ("Sweep Candidate 5", 22, 2, 0.8, 150),
        ("Sweep Candidate 6", 18, 2, 0.6, 100),
        ("Sweep Candidate 7", 20, 2, 0.6, 150),
        ("Sweep Candidate 8", 16, 2, "sqrt", 100),
        ("Sweep Candidate 9", 20, 2, "sqrt", 150),
    ]

    print("\n" + "=" * 130, flush=True)
    print(f"{'Model Configuration':<32} | {'Depth':<5} | {'Leaf':<4} | {'Feat':<6} | {'Train MAE':<9} | {'S2 Val MAE':<10} | {'Gap (S2-Tr)':<11} | {'S2 Urban':<8} | {'S3c Val':<8} | {'S1 MAE':<6} | {'M MAE':<6}", flush=True)
    print("-" * 130, flush=True)

    results = []

    for name, depth, leaf, feat_param, trees in grid:
        model = SpeedEstimatorModel(
            n_estimators=trees,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features=feat_param,
            random_state=42
        )
        model.fit(X_tr, y_tr)

        # Evaluate S1
        p_s1 = model.predict_with_physics(s1_f, s1_al, s1_idx)
        f_s1 = dispatcher.apply_fallback(p_s1, s1_al, s1_idx)
        s1_mae = np.mean(np.abs(f_s1["estimated_speed_mps"].values - s1_y))

        # Evaluate M
        p_m = model.predict_with_physics(m_f, m_al, m_idx)
        f_m = dispatcher.apply_fallback(p_m, m_al, m_idx)
        m_mae = np.mean(np.abs(f_m["estimated_speed_mps"].values - m_y))

        # Overall Train MAE
        tr_mae = (s1_mae * len(s1_y) + m_mae * len(m_y)) / (len(s1_y) + len(m_y))

        # Evaluate S2 (Val Clean)
        p_s2 = model.predict_with_physics(s2_f, s2_al, s2_idx)
        f_s2 = dispatcher.apply_fallback(p_s2, s2_al, s2_idx)
        s2_preds = f_s2["estimated_speed_mps"].values
        s2_mae = np.mean(np.abs(s2_preds - s2_y))

        u_mask = (s2_y >= 5.0) & (s2_y < 15.0)
        s2_urban = np.mean(np.abs(s2_preds[u_mask] - s2_y[u_mask]))

        # Evaluate S3c (Val Swivel)
        p_s3c = model.predict_with_physics(s3c_f, s3c_al, s3c_idx)
        f_s3c = dispatcher.apply_fallback(p_s3c, s3c_al, s3c_idx)
        s3c_mae = np.mean(np.abs(f_s3c["estimated_speed_mps"].values - s3c_y))

        gap = s2_mae - tr_mae

        results.append({
            "name": name, "depth": depth, "leaf": leaf, "feat": feat_param, "trees": trees,
            "tr_mae": tr_mae, "s2_mae": s2_mae, "gap": gap, "s2_urban": s2_urban,
            "s3c_mae": s3c_mae, "s1_mae": s1_mae, "m_mae": m_mae
        })

        feat_str = str(feat_param)[:4]
        print(f"{name:<32} | {depth:<5} | {leaf:<4} | {feat_str:<6} | {tr_mae:<9.3f} | {s2_mae:<10.3f} | {gap:<+11.3f} | {s2_urban:<8.3f} | {s3c_mae:<8.3f} | {s1_mae:<6.3f} | {m_mae:<6.3f}", flush=True)

    print("=" * 130, flush=True)

if __name__ == "__main__":
    main()
