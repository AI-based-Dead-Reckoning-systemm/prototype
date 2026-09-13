"""
test_production_model.py - Evaluates the final candidate model with rolling baseline feature normalization
and optimal balanced hyperparameters (max_depth=16, min_samples_leaf=4, max_features='sqrt', n_estimators=100)
end-to-end with the full post-processing and fallback pipeline.
"""

import os
import sys
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
UPSTREAM_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "preprocessing"))
sys.path.insert(0, UPSTREAM_DIR)
sys.path.insert(1, CURRENT_DIR)

def resolve_data_path(rel_path: str) -> str:
    cand1 = os.path.join(ROOT_DIR, rel_path)
    if os.path.exists(cand1):
        return cand1
    cand2 = os.path.join(UPSTREAM_DIR, rel_path)
    if os.path.exists(cand2):
        return cand2
    return cand1

from src.model import SpeedEstimatorModel
from src.fallback import PhysicsFallbackDispatcher
from src.evaluator import evaluate_predictions, SPEED_BUCKETS
from src.loader import load_ground_truth_can
from src.pipeline import process_trip
from src.features import extract_window_features

def load_trip(s_rel, v_rel, lag=0, is_m=False):
    s_p = resolve_data_path(s_rel)
    v_p = resolve_data_path(v_rel)
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

def apply_rolling_normalization(feat_df):
    df = feat_df.copy()
    # Fast 60s rolling minimum (600 steps @ 10Hz)
    roll_min_aup = df["aup_std"].rolling(window=600, min_periods=50).min().bfill().ffill()
    roll_min_amag = df["amag_std"].rolling(window=600, min_periods=50).min().bfill().ffill()
    df["aup_std_rel_road"] = df["aup_std"] - roll_min_aup
    df["amag_std_rel_road"] = df["amag_std"] - roll_min_amag
    df["aup_std_ratio_road"] = df["aup_std"] / np.maximum(roll_min_aup, 0.05)
    return df

def main():
    print("Loading datasets...")
    s1_al, s1_f_raw, s1_idx, s1_y = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/V-S1.csv", lag=43)
    m_al, m_f_raw, m_idx, m_y = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv", is_m=True)
    s2_al, s2_f_raw, s2_idx, s2_y = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/S-S2.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/V-S2.csv", lag=-41)
    s3c_al, s3c_f_raw, s3c_idx, s3c_y = load_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/S-S3c.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/V-S3c.csv", lag=42)

    # 1. Feature normalization
    s1_f = apply_rolling_normalization(s1_f_raw)
    m_f = apply_rolling_normalization(m_f_raw)
    s2_f = apply_rolling_normalization(s2_f_raw)
    s3c_f = apply_rolling_normalization(s3c_f_raw)

    X_tr = pd.concat([s1_f, m_f], ignore_index=True)
    y_tr = np.concatenate([s1_y, m_y])

    # 2. Fit Balanced Model
    print("Training Balanced SpeedEstimatorModel (Depth 16, Leaf 4, Feat 'sqrt')...")
    model = SpeedEstimatorModel(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42
    )
    model.fit(X_tr, y_tr)

    dispatcher = PhysicsFallbackDispatcher(confidence_threshold=0.35, max_variance_threshold=4.0)

    trips = [
        ("Trip S1", "TRAIN (Driver A)", s1_f, s1_al, s1_idx, s1_y),
        ("Trip M", "TRAIN (Driver B)", m_f, m_al, m_idx, m_y),
        ("Trip S2", "VAL (Clean Unseen)", s2_f, s2_al, s2_idx, s2_y),
        ("Trip S3c", "VAL (Swivel Corrupted)", s3c_f, s3c_al, s3c_idx, s3c_y),
    ]

    summaries = []
    bucket_results = {}

    print("\n" + "=" * 115)
    print("FINAL END-TO-END PIPELINE PERFORMANCE (WITH ROLLING BASELINE NORMALIZATION & BALANCED MODEL)")
    print("=" * 115)

    for name, role, feat, al, idx, gt_y in trips:
        preds = model.predict_with_physics(feat, al, idx)
        final_df = dispatcher.apply_fallback(preds, al, idx)
        eval_res = evaluate_predictions(final_df, gt_y, trip_name=name)

        ov = eval_res["overall"]
        sb = eval_res["source_breakdown"]
        summaries.append({
            "Trip": name,
            "Role": role,
            "Samples": ov["count"],
            "Overall MAE": ov["mae"],
            "Overall RMSE": ov["rmse"],
            "Pearson r": ov["pearson_r"],
            "ML MAE": sb["ml_model"]["mae"],
            "ML Use %": eval_res["ml_usage_pct"],
            "Fallback MAE": sb["physics_fallback"]["mae"] if sb["physics_fallback"]["count"] > 0 else "N/A",
            "Bias": ov["bias"]
        })
        bucket_results[name] = eval_res["speed_bucket_breakdown"]

    print(pd.DataFrame(summaries).to_string(index=False))

    print("\n" + "=" * 115)
    print("PERFORMANCE BREAKDOWN BY SPEED BUCKET (MAE in m/s)")
    print("=" * 115)
    b_rows = []
    for name, role, _, _, _, _ in trips:
        sb = bucket_results[name]
        row = {"Trip": name, "Role": role}
        for label, _, _ in SPEED_BUCKETS:
            b_info = sb.get(label, {"mae": 0.0, "count": 0})
            row[label.split('(')[0].strip()] = f"{b_info['mae']:.3f} (N={b_info['count']})" if b_info['count'] > 0 else "N/A"
        b_rows.append(row)
    print(pd.DataFrame(b_rows).to_string(index=False))

if __name__ == "__main__":
    main()
