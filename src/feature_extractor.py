# src/feature_extractor.py
import os
import numpy as np
import pandas as pd

# ---------- windowing ----------
def _make_windows(df, win_seconds: float, overlap: float, hz: int):
    """
    Returns a list of slices (start_idx, end_idx) over the rows of df
    using fixed-size windows with fixed hop.
    """
    if df.empty: 
        return []
    win = int(round(win_seconds * hz))
    hop = int(round(win * (1.0 - overlap)))
    n = len(df)
    out = []
    i = 0
    while i + win <= n:
        out.append((i, i + win))
        i += hop if hop > 0 else win
    return out

# ---------- features per window ----------
def _rfft_features(x, hz):
    """dominant frequency and spectral energy from real FFT."""
    # real FFT
    F = np.fft.rfft(x)
    P = (np.abs(F)**2)
    freqs = np.fft.rfftfreq(len(x), d=1.0/hz)
    # ignore the DC bin for dominant frequency
    if len(P) > 1:
        dom_idx = int(np.argmax(P[1:])) + 1
        dom_freq = float(freqs[dom_idx])
    else:
        dom_freq = 0.0
    energy = float(P.sum() / len(P))
    return dom_freq, energy

def _time_stats(x):
    x = np.asarray(x)
    return {
        "mean": float(np.mean(x)),
        "std":  float(np.std(x, ddof=0)),
        "var":  float(np.var(x, ddof=0)),
        "ptp":  float(np.ptp(x)),     # peak-to-peak (NumPy-safe)
        "rms":  float(np.sqrt(np.mean(x**2))),
    }

def _sma(ax, ay, az):
    """Signal Magnitude Area on accel axes."""
    return float((np.mean(np.abs(ax)) + np.mean(np.abs(ay)) + np.mean(np.abs(az))))

def features_for_file(csv_path: str, win_seconds: float, overlap: float, hz: int) -> pd.DataFrame:
    """
    Compute features for every window in one cleaned CSV.
    Returns a DataFrame where each row is a window.
    """
    df = pd.read_csv(csv_path)
    # ensure required columns exist
    needed = ["time_s","ax","ay","az","gx","gy","gz","activity","split","recording_id"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {os.path.basename(csv_path)}")
    # windows
    idxs = _make_windows(df, win_seconds, overlap, hz)
    rows = []
    for (i0, i1) in idxs:
        w = df.iloc[i0:i1]
        ax, ay, az = w["ax"].to_numpy(), w["ay"].to_numpy(), w["az"].to_numpy()
        gx, gy, gz = w["gx"].to_numpy(), w["gy"].to_numpy(), w["gz"].to_numpy()
        mag = np.sqrt(ax**2 + ay**2 + az**2)

        # time-domain stats
        t_ax, t_ay, t_az = _time_stats(ax), _time_stats(ay), _time_stats(az)
        t_gx, t_gy, t_gz = _time_stats(gx), _time_stats(gy), _time_stats(gz)
        t_mag = _time_stats(mag)

        # correlations (accel)
        corr_xy = float(np.corrcoef(ax, ay)[0,1]) if len(ax) > 1 else 0.0
        corr_xz = float(np.corrcoef(ax, az)[0,1]) if len(ax) > 1 else 0.0
        corr_yz = float(np.corrcoef(ay, az)[0,1]) if len(ax) > 1 else 0.0

        # SMA
        sma = _sma(ax, ay, az)

        # frequency features (dominant freq + spectral energy) from accel magnitude
        domf_mag, ener_mag = _rfft_features(mag, hz)

        row = {
            # labels / meta
            "activity": w["activity"].iloc[0],
            "split":    w["split"].iloc[0],
            "recording_id": int(w["recording_id"].iloc[0]),
            "t_start":  float(w["time_s"].iloc[0]),
            "t_end":    float(w["time_s"].iloc[-1]),
            # accel stats
            "ax_mean": t_ax["mean"], "ax_std": t_ax["std"], "ax_var": t_ax["var"], "ax_ptp": t_ax["ptp"], "ax_rms": t_ax["rms"],
            "ay_mean": t_ay["mean"], "ay_std": t_ay["std"], "ay_var": t_ay["var"], "ay_ptp": t_ay["ptp"], "ay_rms": t_ay["rms"],
            "az_mean": t_az["mean"], "az_std": t_az["std"], "az_var": t_az["var"], "az_ptp": t_az["ptp"], "az_rms": t_az["rms"],
            # gyro stats
            "gx_mean": t_gx["mean"], "gx_std": t_gx["std"], "gx_var": t_gx["var"], "gx_ptp": t_gx["ptp"], "gx_rms": t_gx["rms"],
            "gy_mean": t_gy["mean"], "gy_std": t_gy["std"], "gy_var": t_gy["var"], "gy_ptp": t_gy["ptp"], "gy_rms": t_gy["rms"],
            "gz_mean": t_gz["mean"], "gz_std": t_gz["std"], "gz_var": t_gz["var"], "gz_ptp": t_gz["ptp"], "gz_rms": t_gz["rms"],
            # accel magnitude + correlations
            "amag_mean": t_mag["mean"], "amag_std": t_mag["std"], "amag_var": t_mag["var"], "amag_ptp": t_mag["ptp"], "amag_rms": t_mag["rms"],
            "acc_corr_xy": corr_xy, "acc_corr_xz": corr_xz, "acc_corr_yz": corr_yz,
            "acc_sma": sma,
            # freq features
            "amag_domfreq": domf_mag,
            "amag_energy":  ener_mag,
        }
        rows.append(row)

    return pd.DataFrame(rows)

def features_for_many(csv_paths, win_seconds: float, overlap: float, hz: int) -> pd.DataFrame:
    """Concatenate features for a list of cleaned CSVs."""
    all_rows = []
    for p in csv_paths:
        f = features_for_file(p, win_seconds, overlap, hz)
        if not f.empty:
            all_rows.append(f)
    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
