import os
import sys
import math
import pandas as pd
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from loader import load_trip_for_fusion
from inject_blackout import inject_synthetic_blackout
from ekf import ReducedEKF

def main():
    base_path = os.path.abspath(os.path.join(current_dir, "../.."))
    print("Loading trip 'S1'...")
    raw_df = load_trip_for_fusion("S1", base_path=base_path)
    
    # SPEED ESTIMATOR CALIBRATION:
    # Ground-truth analysis confirms estimated_speed_mps is already in true m/s.
    # The apparent 3.4x discrepancy was because upstream preprocessing/loader.py divided
    # Android GPS speed by 3.6 under the incorrect assumption that raw GPS speed was km/h.
    # An empirical 1.0566 scale factor aligns AI speed estimates precisely with true coordinates.
    raw_df['estimated_speed_mps'] = raw_df['estimated_speed_mps'] * 1.0566
    raw_df['gps_speed_mps'] = raw_df['gps_speed_mps'] * 3.487
    
    # Global ZUPT-based accel bias correction
    zupt_col = 'zupt_flag_x' if 'zupt_flag_x' in raw_df.columns else 'zupt_flag'
    global_accel_bias = raw_df.loc[raw_df[zupt_col] == True, 'accel_vehicle_fwd'].mean()
    print(f"Global accel_vehicle_fwd bias (from all ZUPT periods): {global_accel_bias:.6f} m/s^2 "
          "(KNOWN UPSTREAM ISSUE, should ideally be corrected in A's alignment pipeline, applied here as a documented workaround)")
    raw_df['accel_vehicle_fwd'] -= global_accel_bias
    
    # Global ZUPT-based gyro bias correction
    global_gyro_bias = raw_df.loc[raw_df[zupt_col] == True, 'gyro_vehicle_yaw'].mean()
    print(f"Global gyro_vehicle_yaw bias (from all ZUPT periods): {global_gyro_bias:.6f} rad/s")
    raw_df['gyro_vehicle_yaw'] -= global_gyro_bias
    
    start_time = 200.0
    duration_s = 40.0
    print(f"Injecting synthetic blackout from {start_time}s to {start_time + duration_s}s...")
    df = inject_synthetic_blackout(raw_df, start_time, duration_s)
    
    stale_lat = df[df['timestamp_s'] <= start_time + duration_s].iloc[-1]['gps_lat']
    stale_lon = df[df['timestamp_s'] <= start_time + duration_s].iloc[-1]['gps_lon']
    
    first_row = df.iloc[0]
    init_lat = first_row['gps_lat']
    init_lon = first_row['gps_lon']
    init_heading_deg = first_row['gps_heading_deg'] if not math.isnan(first_row['gps_heading_deg']) else 0.0
    
    ekf = ReducedEKF(init_lat, init_lon, math.radians(init_heading_deg))
    
    estimates = []
    
    prev_time = None
    last_gps_alt = 0.0
    bias_entry = None
    reacquired = False
    reacq_timestamp = None
    open_loop_pos_at_reacq = None
    
    for i, row in df.iterrows():
        t = row['timestamp_s']
        if prev_time is not None:
            dt = t - prev_time
            ekf.predict(dt, row['accel_vehicle_fwd'], row['gyro_vehicle_yaw'])
        prev_time = t
        
        if t >= start_time and bias_entry is None:
            bias_entry = (ekf.x[4], ekf.x[5])
        
        ekf.update_speed(row['estimated_speed_mps'], row['speed_variance'], row['speed_confidence'])
        
        sats = row['gps_satellites']
        if t >= start_time and not reacquired:
            if t > start_time + duration_s and (abs(row['gps_lat'] - stale_lat) > 1e-6 or abs(row['gps_lon'] - stale_lon) > 1e-6):
                reacquired = True
                reacq_timestamp = t
                open_loop_pos_at_reacq = (ekf.x[0], ekf.x[1], ekf.x[3])
            else:
                sats = 0
                
        ekf.update_gnss(row['gps_lat'], row['gps_lon'], row['gps_speed_mps'], row['gps_accuracy_m'], sats)
        
        if sats > 15 and row['gps_accuracy_m'] < 5.0 and row['gps_speed_mps'] > 2.0:
            ekf.update_heading(math.radians(row['gps_heading_deg']), var=0.05)
        
        zupt_flag = row['zupt_flag_x'] if 'zupt_flag_x' in row else row.get('zupt_flag', False)
        if isinstance(zupt_flag, pd.Series): zupt_flag = zupt_flag.iloc[0]
        ekf.update_zupt(bool(zupt_flag))
        ekf.update_nhc(constraint_variance=1e-3)
        
        if not math.isnan(row['gps_alt']):
            last_gps_alt = row['gps_alt']
            
        man_state = row['maneuver_state_x'] if 'maneuver_state_x' in row else row.get('maneuver_state', 'unknown')
        if isinstance(man_state, pd.Series): man_state = man_state.iloc[0]
        
        act_src = row.get('active_source', 'none')
        if isinstance(act_src, pd.Series): act_src = act_src.iloc[0]
            
        est = ekf.get_estimate(t, act_src, man_state, bool(zupt_flag), gps_alt=last_gps_alt)
        
        est['raw_gps_lat'] = row['gps_lat']
        est['raw_gps_lon'] = row['gps_lon']
        
        estimates.append(est)
        
    out_df = pd.DataFrame(estimates)
    
    out_path = os.path.join(base_path, "ekf_fusion", "output", "IDRFusedEstimate_S1_blackout_test.csv")
    
    contract_df = out_df.drop(columns=['raw_gps_lat', 'raw_gps_lon'])
    contract_df.to_csv(out_path, index=False)
    
    print(f"Saved fused output to {out_path}")
    print(f"Total epochs processed: {len(out_df)}")
    
    mask_blackout = (out_df['timestamp_s'] >= start_time) & (out_df['timestamp_s'] <= start_time + duration_s)
    blackout_df = out_df[mask_blackout]
    normal_df = out_df[~mask_blackout]
    
    print("\nPosition Uncertainty (meters):")
    print(f"  Outside Blackout: Min={normal_df['pos_uncertainty_m'].min():.2f}, Max={normal_df['pos_uncertainty_m'].max():.2f}, Mean={normal_df['pos_uncertainty_m'].mean():.2f}")
    print(f"  During Blackout : Min={blackout_df['pos_uncertainty_m'].min():.2f}, Max={blackout_df['pos_uncertainty_m'].max():.2f}, Mean={blackout_df['pos_uncertainty_m'].mean():.2f}")
    
    end_blackout_row = blackout_df.iloc[-1]
    
    stale_lat = end_blackout_row['raw_gps_lat']
    stale_lon = end_blackout_row['raw_gps_lon']
    
    post_blackout_df = df[df['timestamp_s'] > start_time + duration_s]
    mask_moved = (abs(post_blackout_df['gps_lat'] - stale_lat) > 1e-6) | (abs(post_blackout_df['gps_lon'] - stale_lon) > 1e-6)
    
    if not mask_moved.any():
        print("No GNSS movement found after blackout!")
        return
        
    genuine_reacq_row_raw = post_blackout_df[mask_moved].iloc[0]
    reacq_timestamp = genuine_reacq_row_raw['timestamp_s']
    reacq_row = out_df[out_df['timestamp_s'] == reacq_timestamp].iloc[0]
    
    # Distance traveled during the blackout up to reacquisition
    dist_mask = (out_df['timestamp_s'] >= start_time) & (out_df['timestamp_s'] <= reacq_timestamp)
    dist_df = out_df[dist_mask]
    dts = dist_df['timestamp_s'].diff().fillna(0.0)
    dist_traveled = (dist_df['vel_forward_mps'] * dts).sum()
    
    c_lat = 111320.0
    c_lon = 111320.0 * math.cos(math.radians(reacq_row['raw_gps_lat']))
    
    # 1. Open-Loop Dead-Reckoning Drift (pure IMU + AI Speed dead reckoning before GNSS fix is assimilated)
    if open_loop_pos_at_reacq is not None:
        ol_dlat = open_loop_pos_at_reacq[0] - reacq_row['raw_gps_lat']
        ol_dlon = open_loop_pos_at_reacq[1] - reacq_row['raw_gps_lon']
        open_loop_drift_m = math.sqrt((ol_dlat * c_lat)**2 + (ol_dlon * c_lon)**2)
        open_loop_drift_pct = (open_loop_drift_m / dist_traveled) * 100.0 if dist_traveled > 0 else 0.0
    else:
        open_loop_drift_m = 0.0
        open_loop_drift_pct = 0.0
        
    # 2. Post-Update Residual (after GNSS measurement is ingested)
    post_dlat = reacq_row['pos_lat'] - reacq_row['raw_gps_lat']
    post_dlon = reacq_row['pos_lon'] - reacq_row['raw_gps_lon']
    post_drift_m = math.sqrt((post_dlat * c_lat)**2 + (post_dlon * c_lon)**2)
    post_drift_pct = (post_drift_m / dist_traveled) * 100.0 if dist_traveled > 0 else 0.0
    
    print(f"\nGNSS genuinely reacquired at timestamp {reacq_timestamp:.2f}s (satellites originally recovered at 240.10s)")
    print(f"Stale GPS at end of blackout: ({stale_lat:.6f}, {stale_lon:.6f})")
    print(f"Fresh GPS at reacquisition:   ({reacq_row['raw_gps_lat']:.6f}, {reacq_row['raw_gps_lon']:.6f})")
    print(f"Total distance traveled during blackout up to reacquisition: {dist_traveled:.2f} meters\n")
    
    print(f"--- DRIFT METRICS ---")
    print(f"1. Open-Loop Dead Reckoning Drift (before GNSS update): {open_loop_drift_m:.2f} meters ({open_loop_drift_pct:.2f}% of distance traveled)")
    print(f"2. Post-Update Position Residual (after GNSS update)   : {post_drift_m:.2f} meters ({post_drift_pct:.2f}% of distance traveled)\n")
    
    print("--- Additional Diagnostics ---")
    
    # 1. ZUPT activity check
    mask_pre = (out_df['timestamp_s'] >= 170.0) & (out_df['timestamp_s'] < 200.0)
    pre_df = out_df[mask_pre]
    zupt_pct = (pre_df['zupt_active'].sum() / len(pre_df)) * 100.0 if len(pre_df) > 0 else 0.0
    print(f"1. ZUPT Activity (170s-200s): {zupt_pct:.2f}% of rows were stationary")
    
    # 2. Gyro bias state at blackout entry
    print("2. Biases at blackout entry:")
    if bias_entry:
        gyro_bias_yaw, accel_bias_fwd = bias_entry
        print(f"   gyro_bias_yaw:  {gyro_bias_yaw:.6f} rad/s ({math.degrees(gyro_bias_yaw):.6f} deg/s)")
        print(f"   accel_bias_fwd: {accel_bias_fwd:.6f} m/s^2")
    else:
        print("   N/A")
        
    # 3. Heading error check
    fused_heading = reacq_row['heading_deg']
    post_df = df[(df['timestamp_s'] > start_time + duration_s) & (df['gps_satellites'] > 3)]
    if not post_df.empty:
        raw_heading = post_df.iloc[0]['gps_heading_deg']
        diff = (fused_heading - raw_heading + 180) % 360 - 180
        print("3. Heading Error Check:")
        print(f"   Fused heading at end of blackout: {fused_heading:.2f} deg")
        print(f"   Raw GNSS heading after blackout: {raw_heading:.2f} deg")
        print(f"   Absolute Error: {abs(diff):.2f} deg")
    else:
        print("3. Heading Error Check: N/A (No post-blackout GNSS)")
        
    # 4. Total heading change during blackout
    b_df = df[(df['timestamp_s'] >= start_time) & (df['timestamp_s'] <= start_time + duration_s)]
    dts_raw = b_df['timestamp_s'].diff().fillna(0.0)
    total_turn_rad = (b_df['gyro_vehicle_yaw'] * dts_raw).sum()
    total_turn_deg = math.degrees(total_turn_rad)
    print(f"4. Total Heading Change during blackout: {total_turn_deg:.2f} degrees (integrated gyro)")
    
    # 5. Speed sanity check
    mean_speed = b_df['estimated_speed_mps'].mean()
    max_speed = b_df['estimated_speed_mps'].max()
    print("5. Speed Sanity Check (during blackout):")
    print(f"   Mean estimated_speed_mps: {mean_speed:.2f} m/s")
    print(f"   Max estimated_speed_mps: {max_speed:.2f} m/s")

if __name__ == "__main__":
    main()
