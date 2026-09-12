"""
investigate_and_optimize.py - Deep Investigation and Multi-Factor Optimization Suite.
"""

import os
import sys
import importlib.util
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
from scipy.fft import rfft, rfftfreq

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

from src.evaluator import compute_metrics, SPEED_BUCKETS

MANEUVER_CLASSES = ["stationary", "straight", "gentle_curve", "sharp_turn", "reversal"]

def extract_base_features(aligned_df: pd.DataFrame, window_size: int = 20, step_size: int = 1):
    n_samples = len(aligned_df)
    afwd = aligned_df["accel_vehicle_fwd"].values.astype(np.float64)
    alat = aligned_df["accel_vehicle_lat"].values.astype(np.float64)
    aup = aligned_df["accel_vehicle_up"].values.astype(np.float64)
    
    wyaw = aligned_df["gyro_vehicle_yaw"].values.astype(np.float64)
    wpitch = aligned_df["gyro_vehicle_pitch"].values.astype(np.float64)
    wroll = aligned_df["gyro_vehicle_roll"].values.astype(np.float64)
    
    zupt = aligned_df["zupt_flag"].values.astype(bool)
    maneuver = aligned_df["maneuver_state"].values
    
    az_raw = aup + 9.80665
    amag = np.sqrt(afwd**2 + alat**2 + az_raw**2)
    
    jerk = np.zeros_like(amag)
    jerk[1:] = np.abs(np.diff(amag)) / 0.1

    wmag = np.sqrt(wyaw**2 + wpitch**2 + wroll**2)

    fs = 10.0
    freqs = rfftfreq(window_size, d=1.0/fs)
    
    low_band_mask = (freqs >= 0.5) & (freqs <= 2.0)
    vib_band_mask = (freqs > 2.0) & (freqs <= 4.5)
    ac_mask = (freqs >= 0.5) & (freqs <= 4.5)

    rows = []
    end_indices = []

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        end_indices.append(end - 1)
        
        w_afwd = afwd[start:end]
        w_alat = alat[start:end]
        w_aup  = aup[start:end]
        
        w_wyaw = wyaw[start:end]
        w_wpitch = wpitch[start:end]
        w_wroll = wroll[start:end]
        
        w_amag = amag[start:end]
        w_jerk = jerk[start:end]
        w_wmag = wmag[start:end]
        w_zupt = zupt[start:end]
        last_maneuver = maneuver[end - 1]

        afwd_mean = np.mean(w_afwd)
        afwd_std = np.std(w_afwd)
        afwd_min = np.min(w_afwd)
        afwd_max = np.max(w_afwd)
        afwd_range = afwd_max - afwd_min
        
        alat_mean = np.mean(w_alat)
        alat_std = np.std(w_alat)
        alat_abs_mean = np.mean(np.abs(w_alat))
        alat_max = np.max(np.abs(w_alat))

        aup_mean = np.mean(w_aup)
        aup_std = np.std(w_aup)

        wyaw_mean = np.mean(w_wyaw)
        wyaw_std = np.std(w_wyaw)
        wyaw_abs_mean = np.mean(np.abs(w_wyaw))
        wyaw_max = np.max(np.abs(w_wyaw))

        wmag_mean = np.mean(w_wmag)
        wmag_std = np.std(w_wmag)

        amag_mean = np.mean(w_amag)
        amag_std = np.std(w_amag)
        amag_var = np.var(w_amag)
        amag_range = np.max(w_amag) - np.min(w_amag)
        
        jerk_mean = np.mean(w_jerk)
        jerk_max = np.max(w_jerk)

        amag_ac = w_amag - amag_mean
        fft_vals = np.abs(rfft(amag_ac))
        
        fft_energy_low = np.sum(fft_vals[low_band_mask]**2) if np.any(low_band_mask) else 0.0
        fft_energy_vib = np.sum(fft_vals[vib_band_mask]**2) if np.any(vib_band_mask) else 0.0
        fft_energy_total = np.sum(fft_vals[ac_mask]**2) if np.any(ac_mask) else 0.0

        ac_amplitudes = fft_vals[ac_mask]
        ac_freqs = freqs[ac_mask]
        sum_amp = np.sum(ac_amplitudes)
        if sum_amp > 1e-6:
            spectral_centroid = np.sum(ac_amplitudes * ac_freqs) / sum_amp
            dominant_freq = ac_freqs[np.argmax(ac_amplitudes)]
        else:
            spectral_centroid = 0.0
            dominant_freq = 0.0

        centripetal_proxy = alat_abs_mean * wyaw_abs_mean
        zupt_ratio = np.mean(w_zupt.astype(float))
        is_zupt_end = float(w_zupt[-1])

        feat = {
            "afwd_mean": afwd_mean, "afwd_std": afwd_std, "afwd_min": afwd_min, "afwd_max": afwd_max, "afwd_range": afwd_range,
            "alat_mean": alat_mean, "alat_std": alat_std, "alat_abs_mean": alat_abs_mean, "alat_max": alat_max,
            "aup_mean": aup_mean, "aup_std": aup_std,
            "wyaw_mean": wyaw_mean, "wyaw_std": wyaw_std, "wyaw_abs_mean": wyaw_abs_mean, "wyaw_max": wyaw_max,
            "wmag_mean": wmag_mean, "wmag_std": wmag_std,
            "amag_mean": amag_mean, "amag_std": amag_std, "amag_var": amag_var, "amag_range": amag_range,
            "jerk_mean": jerk_mean, "jerk_max": jerk_max,
            "fft_energy_low": fft_energy_low, "fft_energy_vib": fft_energy_vib, "fft_energy_total": fft_energy_total,
            "spectral_centroid": spectral_centroid, "dominant_freq": dominant_freq,
            "centripetal_proxy": centripetal_proxy, "zupt_ratio": zupt_ratio, "is_zupt_end": is_zupt_end
        }
        for m_class in MANEUVER_CLASSES:
            feat[f"maneuver_{m_class}"] = float(last_maneuver == m_class)

        rows.append(feat)

    return pd.DataFrame(rows), np.array(end_indices)

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

print("1. Loading raw trip data...")
s1_al, s1_gt = get_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/S-S1.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1/V-S1.csv", lag=43)
m_al, m_gt = get_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/S-M.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)/V-M.csv", is_m=True)
s2_al, s2_gt = get_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/S-S2.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2/V-S2.csv", lag=-41)
s3c_al, s3c_gt = get_trip("data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/S-S3c.csv", "data/IO-VNBD/Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S3c/V-S3c.csv", lag=42)

print("2. Extracting base multi-domain features...")
s1_base_f, s1_idx = extract_base_features(s1_al)
m_base_f, m_idx = extract_base_features(m_al)
s2_base_f, s2_idx = extract_base_features(s2_al)
s3c_base_f, s3c_idx = extract_base_features(s3c_al)

s1_y = s1_gt["gt_speed_mps"].values[s1_idx]
m_y = m_gt["gt_speed_mps"].values[m_idx]
s2_y = s2_gt["gt_speed_mps"].values[s2_idx]
s3c_y = s3c_gt["gt_speed_mps"].values[s3c_idx]

def get_normalized_features(s1_f, m_f, s2_f, s3c_f, norm_mode: str):
    s1 = s1_f.copy()
    m = m_f.copy()
    s2 = s2_f.copy()
    s3c = s3c_f.copy()
    
    if norm_mode == "none":
        return s1, m, s2, s3c
        
    elif norm_mode == "roll60s_both":
        for df in [s1, m, s2, s3c]:
            r_aup = df["aup_std"].rolling(600, min_periods=50).min().bfill().ffill()
            r_amag = df["amag_std"].rolling(600, min_periods=50).min().bfill().ffill()
            df["aup_std_rel_road"] = df["aup_std"] - r_aup
            df["amag_std_rel_road"] = df["amag_std"] - r_amag
            df["aup_std_ratio_road"] = df["aup_std"] / np.maximum(r_aup, 0.05)
        return s1, m, s2, s3c
        
    elif norm_mode == "roll20s_both":
        for df in [s1, m, s2, s3c]:
            r_aup = df["aup_std"].rolling(200, min_periods=30).min().bfill().ffill()
            r_amag = df["amag_std"].rolling(200, min_periods=30).min().bfill().ffill()
            df["aup_std_rel_road"] = df["aup_std"] - r_aup
            df["amag_std_rel_road"] = df["amag_std"] - r_amag
            df["aup_std_ratio_road"] = df["aup_std"] / np.maximum(r_aup, 0.05)
        return s1, m, s2, s3c

    elif norm_mode == "roll60s_amag_only":
        for df in [s1, m, s2, s3c]:
            r_amag = df["amag_std"].rolling(600, min_periods=50).min().bfill().ffill()
            df["amag_std_rel_road"] = df["amag_std"] - r_amag
            df["amag_std_ratio_road"] = df["amag_std"] / np.maximum(r_amag, 0.05)
        return s1, m, s2, s3c

    elif norm_mode == "roll20s_amag_only":
        for df in [s1, m, s2, s3c]:
            r_amag = df["amag_std"].rolling(200, min_periods=30).min().bfill().ffill()
            df["amag_std_rel_road"] = df["amag_std"] - r_amag
            df["amag_std_ratio_road"] = df["amag_std"] / np.maximum(r_amag, 0.05)
        return s1, m, s2, s3c
        
    elif norm_mode == "roll10s_amag_only":
        for df in [s1, m, s2, s3c]:
            r_amag = df["amag_std"].rolling(100, min_periods=20).min().bfill().ffill()
            df["amag_std_rel_road"] = df["amag_std"] - r_amag
            df["amag_std_ratio_road"] = df["amag_std"] / np.maximum(r_amag, 0.05)
        return s1, m, s2, s3c

    return s1, m, s2, s3c

def eval_fast(rf, X_feat, aligned_df, end_idx, gt_speed, conf_thresh=0.25, max_var_thresh=16.0):
    all_tree_preds = np.array(
        Parallel(n_jobs=-1, prefer="threads")(
            delayed(tree.predict)(X_feat.values) for tree in rf.estimators_
        )
    )
    pred_mean = np.mean(all_tree_preds, axis=0)
    pred_var = np.var(all_tree_preds, axis=0)
    pred_std = np.sqrt(pred_var)
    pred_conf = 1.0 / (1.0 + (pred_std / 1.5))
    
    n = len(pred_mean)
    zupt = aligned_df["zupt_flag"].values[end_idx]
    maneuver = aligned_df["maneuver_state"].values[end_idx]
    afwd = aligned_df["accel_vehicle_fwd"].values[end_idx]
    
    bounded_v = np.clip(pred_mean, 0.0, 45.0)
    bounded_v[zupt | (maneuver == "stationary")] = 0.0
    
    prev_v = 0.0
    for i in range(n):
        dv = bounded_v[i] - prev_v
        if abs(dv) > 0.5:
            bounded_v[i] = prev_v + np.sign(dv) * 0.5
        prev_v = bounded_v[i]
        
    final_speed = np.zeros(n)
    active_source = []
    kinematic_v = 0.0
    dt = 0.1
    
    for i in range(n):
        is_low_conf = (pred_conf[i] < conf_thresh) or (pred_var[i] > max_var_thresh)
        if zupt[i] or maneuver[i] == "stationary":
            kinematic_v = 0.0
        else:
            kinematic_v = max(0.0, kinematic_v + afwd[i] * dt)
            
        if is_low_conf and not zupt[i]:
            final_speed[i] = kinematic_v
            active_source.append("physics_fallback")
        else:
            final_speed[i] = bounded_v[i]
            active_source.append("ml_model")
            kinematic_v = bounded_v[i]
            
    active_source = np.array(active_source)
    ml_pct = (active_source == "ml_model").mean() * 100.0
    
    mae_overall = np.mean(np.abs(final_speed - gt_speed))
    mae_bounded_ml = np.mean(np.abs(bounded_v - gt_speed))
    
    u_mask = (gt_speed >= 5.0) & (gt_speed < 15.0)
    mae_urban = np.mean(np.abs(final_speed[u_mask] - gt_speed[u_mask])) if np.any(u_mask) else 0.0

    return mae_overall, mae_bounded_ml, mae_urban, ml_pct

print("\n" + "=" * 135)
print("COMPREHENSIVE HYPERPARAMETER & NORMALIZATION SWEEP")
print("=" * 135)

norm_modes = ["none", "roll60s_both", "roll20s_both", "roll60s_amag_only", "roll20s_amag_only"]
configs = [
    # (max_depth, min_samples_leaf, max_features, n_estimators)
    # 1. Unregularized baseline equivalents
    (None, 1, 1.0, 100),
    (22, 2, 1.0, 100),
    (20, 2, 1.0, 100),
    (18, 2, 1.0, 100),
    
    # 2. Balanced feature-fraction models
    (20, 2, 0.7, 100),
    (18, 2, 0.7, 100),
    (18, 4, 0.7, 100),
    (16, 2, 0.7, 100),
    (16, 4, 0.7, 100),
    (16, 2, 0.5, 100),
    (16, 4, 0.5, 100),
    (18, 2, 0.5, 100),
    (18, 4, 0.5, 100),
    
    # 3. sqrt feature models
    (20, 2, "sqrt", 100),
    (18, 2, "sqrt", 100),
    (18, 4, "sqrt", 100),
    (16, 2, "sqrt", 100),
    (16, 4, "sqrt", 100),
    (14, 4, "sqrt", 100),
    (14, 2, 1.0, 100),
    (14, 4, 1.0, 100),
    (12, 8, 1.0, 100),
]

sweep_results = []

for n_mode in norm_modes:
    s1_f, m_f, s2_f, s3c_f = get_normalized_features(s1_base_f, m_base_f, s2_base_f, s3c_base_f, n_mode)
    X_tr = pd.concat([s1_f, m_f], ignore_index=True)
    y_tr = np.concatenate([s1_y, m_y])
    
    for (depth, leaf, max_feat, n_est) in configs:
        rf = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=depth,
            min_samples_leaf=leaf,
            max_features=max_feat,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        
        # Training evaluation
        pred_tr = np.clip(rf.predict(X_tr), 0.0, 45.0)
        train_mae = np.mean(np.abs(pred_tr - y_tr))
        
        # S1 & M individual MAEs
        pred_s1 = np.clip(rf.predict(s1_f), 0.0, 45.0)
        pred_s1[s1_al["zupt_flag"].values[s1_idx]] = 0.0
        s1_mae = np.mean(np.abs(pred_s1 - s1_y))
        
        pred_m = np.clip(rf.predict(m_f), 0.0, 45.0)
        pred_m[m_al["zupt_flag"].values[m_idx]] = 0.0
        m_mae = np.mean(np.abs(pred_m - m_y))
        
        # S2 Eval (Clean validation)
        s2_overall_mae, s2_ml_mae, s2_urban_mae, s2_ml_pct = eval_fast(rf, s2_f, s2_al, s2_idx, s2_y, conf_thresh=0.25, max_var_thresh=16.0)
        
        # S3c Eval (Swivel validation)
        s3c_overall_mae, s3c_ml_mae, s3c_urban_mae, s3c_ml_pct = eval_fast(rf, s3c_f, s3c_al, s3c_idx, s3c_y, conf_thresh=0.25, max_var_thresh=16.0)
        
        gap = s2_overall_mae - train_mae
        
        rec = {
            "norm": n_mode,
            "depth": str(depth),
            "leaf": leaf,
            "max_feat": str(max_feat),
            "n_est": n_est,
            "train_mae": round(train_mae, 3),
            "s1_mae": round(s1_mae, 3),
            "m_mae": round(m_mae, 3),
            "s2_val_mae": round(s2_overall_mae, 3),
            "s2_ml_mae": round(s2_ml_mae, 3),
            "s2_urban_mae": round(s2_urban_mae, 3),
            "gap": round(gap, 3),
            "s3c_val_mae": round(s3c_overall_mae, 3),
            "s3c_ml_mae": round(s3c_ml_mae, 3),
            "s3c_ml_pct": round(s3c_ml_pct, 1)
        }
        sweep_results.append(rec)
        print(f"[{n_mode:<17}] D={str(depth):<4} L={leaf:<2} F={str(max_feat):<4} | Train: {train_mae:.3f} | S2 Val: {s2_overall_mae:.3f} (Urban: {s2_urban_mae:.3f}) | Gap: +{gap:.3f} | S3c Val: {s3c_overall_mae:.3f} (ML: {s3c_ml_pct:.1f}%)")

results_df = pd.DataFrame(sweep_results)
results_df.to_csv("sweep_results_complete.csv", index=False)
print("\nResults saved to sweep_results_complete.csv")

# Print Top 15 sorted by S2 Validation MAE
print("\n" + "=" * 135)
print("TOP 15 CONFIGURATIONS SORTED BY S2 VALIDATION MAE (TARGET: BEAT 3.801 m/s)")
print("=" * 135)
top_s2 = results_df.sort_values(by="s2_val_mae").head(15)
print(top_s2.to_string(index=False))

# Print Top 15 sorted by combined S2 + S3c Validation MAE
results_df["val_sum"] = results_df["s2_val_mae"] + results_df["s3c_val_mae"]
print("\n" + "=" * 135)
print("TOP 15 BALANCED CONFIGURATIONS (SORTED BY S2 + S3c SUM)")
print("=" * 135)
top_sum = results_df.sort_values(by="val_sum").head(15)
print(top_sum.to_string(index=False))

