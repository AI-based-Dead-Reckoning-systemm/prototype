"""
features.py - Feature Extraction Engine for AI Speed Estimation.
Extracts rich, physically-grounded time-domain, frequency-domain, orientation-invariant,
and maneuver-contextual features over sliding windows of upstream AlignedSample streams.
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq

# Maneuver categorical mapping
MANEUVER_CLASSES = ["stationary", "straight", "gentle_curve", "sharp_turn", "reversal"]

def extract_window_features(
    aligned_df: pd.DataFrame,
    window_size: int = 20, # 2.0 seconds at 10 Hz
    step_size: int = 1     # 0.1 second hop (10 Hz streaming output)
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Extracts multi-domain feature vectors from an AlignedSample DataFrame over sliding windows.
    
    Window Length: 20 samples (2.0s at 10 Hz) -> Resolves frequency bins down to 0.5 Hz.
    Step Size: 1 sample (0.1s hop) -> Produces 10 Hz continuous speed estimates.
    
    Returns:
      - features_df: DataFrame of extracted features (one row per window)
      - window_end_indices: Array of integer row indices corresponding to the end of each window
    """
    n_samples = len(aligned_df)
    if n_samples < window_size:
        raise ValueError(f"Trip length ({n_samples}) is shorter than window_size ({window_size})")

    # Raw arrays for high-speed vectorized slicing
    afwd = aligned_df["accel_vehicle_fwd"].values.astype(np.float64)
    alat = aligned_df["accel_vehicle_lat"].values.astype(np.float64)
    aup = aligned_df["accel_vehicle_up"].values.astype(np.float64)
    
    wyaw = aligned_df["gyro_vehicle_yaw"].values.astype(np.float64)
    wpitch = aligned_df["gyro_vehicle_pitch"].values.astype(np.float64)
    wroll = aligned_df["gyro_vehicle_roll"].values.astype(np.float64)
    
    zupt = aligned_df["zupt_flag"].values.astype(bool)
    maneuver = aligned_df["maneuver_state"].values
    
    # Orientation-invariant total acceleration magnitude
    # Using raw acceleration components (including 1g vertical)
    az_raw = aup + 9.80665
    amag = np.sqrt(afwd**2 + alat**2 + az_raw**2)
    
    # Jerk magnitude (approximate derivative of amag)
    jerk = np.zeros_like(amag)
    jerk[1:] = np.abs(np.diff(amag)) / 0.1 # m/s³

    # Gyro total angular rate magnitude
    wmag = np.sqrt(wyaw**2 + wpitch**2 + wroll**2)

    # FFT setup for 20-sample window at 10 Hz
    # Frequencies: [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0] Hz
    fs = 10.0
    freqs = rfftfreq(window_size, d=1.0/fs)
    
    # Frequency band masks
    # Low-frequency motion band: 0.5 Hz to 2.0 Hz
    low_band_mask = (freqs >= 0.5) & (freqs <= 2.0)
    # Road vibration / cadence band: 2.0 Hz to 4.5 Hz (within Nyquist limit of 5.0 Hz)
    vib_band_mask = (freqs > 2.0) & (freqs <= 4.5)
    ac_mask = (freqs >= 0.5) & (freqs <= 4.5)

    rows = []
    end_indices = []

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        end_indices.append(end - 1)
        
        # Slices
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

        # 1. Time-Domain Acceleration Features
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

        # 2. Time-Domain Gyroscope Features
        wyaw_mean = np.mean(w_wyaw)
        wyaw_std = np.std(w_wyaw)
        wyaw_abs_mean = np.mean(np.abs(w_wyaw))
        wyaw_max = np.max(np.abs(w_wyaw))

        wmag_mean = np.mean(w_wmag)
        wmag_std = np.std(w_wmag)

        # 3. Orientation-Invariant Magnitude & Jerk Features
        amag_mean = np.mean(w_amag)
        amag_std = np.std(w_amag)
        amag_var = np.var(w_amag)
        amag_range = np.max(w_amag) - np.min(w_amag)
        
        jerk_mean = np.mean(w_jerk)
        jerk_max = np.max(w_jerk)

        # 4. Frequency-Domain Cadence & Vibration Features (FFT on demeaned accel norm)
        amag_ac = w_amag - amag_mean
        fft_vals = np.abs(rfft(amag_ac)) # Real FFT amplitudes
        
        # Band energies
        fft_energy_low = np.sum(fft_vals[low_band_mask]**2) if np.any(low_band_mask) else 0.0
        fft_energy_vib = np.sum(fft_vals[vib_band_mask]**2) if np.any(vib_band_mask) else 0.0
        fft_energy_total = np.sum(fft_vals[ac_mask]**2) if np.any(ac_mask) else 0.0

        # Spectral Centroid within 0.5 - 4.5 Hz
        ac_amplitudes = fft_vals[ac_mask]
        ac_freqs = freqs[ac_mask]
        sum_amp = np.sum(ac_amplitudes)
        if sum_amp > 1e-6:
            spectral_centroid = np.sum(ac_amplitudes * ac_freqs) / sum_amp
            dominant_freq = ac_freqs[np.argmax(ac_amplitudes)]
        else:
            spectral_centroid = 0.0
            dominant_freq = 0.0

        # 5. Kinematic & Turn Interaction Features
        # Centripetal interaction: |alat| * |wyaw|
        centripetal_proxy = alat_abs_mean * wyaw_abs_mean

        # 6. Maneuver & ZUPT State Encoding
        zupt_ratio = np.mean(w_zupt.astype(float))
        is_zupt_end = float(w_zupt[-1])

        # Assemble feature dict
        feat = {
            # Time-Domain Accel
            "afwd_mean": afwd_mean,
            "afwd_std": afwd_std,
            "afwd_min": afwd_min,
            "afwd_max": afwd_max,
            "afwd_range": afwd_range,
            "alat_mean": alat_mean,
            "alat_std": alat_std,
            "alat_abs_mean": alat_abs_mean,
            "alat_max": alat_max,
            "aup_mean": aup_mean,
            "aup_std": aup_std,
            
            # Time-Domain Gyro
            "wyaw_mean": wyaw_mean,
            "wyaw_std": wyaw_std,
            "wyaw_abs_mean": wyaw_abs_mean,
            "wyaw_max": wyaw_max,
            "wmag_mean": wmag_mean,
            "wmag_std": wmag_std,
            
            # Orientation-Invariant
            "amag_mean": amag_mean,
            "amag_std": amag_std,
            "amag_var": amag_var,
            "amag_range": amag_range,
            "jerk_mean": jerk_mean,
            "jerk_max": jerk_max,
            
            # Frequency-Domain
            "fft_energy_low": fft_energy_low,
            "fft_energy_vib": fft_energy_vib,
            "fft_energy_total": fft_energy_total,
            "spectral_centroid": spectral_centroid,
            "dominant_freq": dominant_freq,
            
            # Kinematic interactions & ZUPT
            "centripetal_proxy": centripetal_proxy,
            "zupt_ratio": zupt_ratio,
            "is_zupt_end": is_zupt_end
        }

        # One-hot encoded maneuver states
        for m_class in MANEUVER_CLASSES:
            feat[f"maneuver_{m_class}"] = float(last_maneuver == m_class)

        rows.append(feat)

    features_df = pd.DataFrame(rows)

    # 7. Orientation-Invariant Road-Roughness Rolling Baseline Normalization (20-second horizon @ 10Hz)
    # Uses |a| magnitude standard deviation to normalize cross-trip pavement roughness shifts
    # while remaining completely immune to azimuthal cradle swivels and mounting tilt changes
    roll_min_amag = features_df["amag_std"].rolling(window=200, min_periods=30).min().bfill().ffill()
    features_df["amag_std_rel_road"] = features_df["amag_std"] - roll_min_amag
    features_df["amag_std_ratio_road"] = features_df["amag_std"] / np.maximum(roll_min_amag, 0.05)

    return features_df, np.array(end_indices)
