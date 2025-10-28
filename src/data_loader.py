# data_loader.py
from pathlib import Path
import io, zipfile, re
import numpy as np
import pandas as pd
from src.config import TARGET_HZ, EDGE_TRIM_SEC, MERGE_TOL_SEC  # <- from your config.py

# Filenames like: setA_Jumping_1-2025-10-25_13-52-04.zip
NAME_RE = re.compile(
    r"set([AB])_(Standing|Walking|Jumping|Still)_(\d+)-\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.zip",
    re.IGNORECASE,
)

def _parse_name(zpath: Path):
    m = NAME_RE.match(zpath.name)
    if not m: raise ValueError(f"Bad filename: {zpath.name}")
    set_label, activity, rec_id = m.group(1), m.group(2).lower(), int(m.group(3))
    split = "train" if set_label.upper() == "A" else "test"
    return split, activity, rec_id

def _read_sensor(zp: zipfile.ZipFile, keywords):
    """Return first CSV matching keywords → DataFrame with [time_s, x, y, z]."""
    target = None
    for info in zp.infolist():
        nm = info.filename.lower()
        if nm.endswith(".csv") and any(k in nm for k in keywords):
            target = info; break
    if target is None: 
        return None

    with zp.open(target) as f:
        raw = f.read()
    df = pd.read_csv(io.BytesIO(raw))

    # pick time column
    lower = {c.lower(): c for c in df.columns}
    for cand in ("seconds_elapsed","timestamp","time","epoch(s)","epoch(ms)"):
        if cand in lower:
            tcol = lower[cand]; break
    else:
        tcol = df.columns[0]  # fallback

    t = df[tcol].astype(float).to_numpy()
    if t.max() > 1e6:  # epoch ms → seconds
        t = t / 1000.0
    t = t - t[0]

    # pick x,y,z
    def pick(*names):
        for n in names:
            if n in df: return df[n].astype(float).to_numpy()
            if n.upper() in df: return df[n.upper()].astype(float).to_numpy()
        return None

    x = pick("x","ax","accx")
    y = pick("y","ay","accy")
    z = pick("z","az","accz")
    if x is None or y is None or z is None:
        # fallback: first 3 numeric after time
        others = [c for c in df.columns if c != tcol and pd.api.types.is_numeric_dtype(df[c])]
        if len(others) >= 3:
            x = df[others[0]].to_numpy(float)
            y = df[others[1]].to_numpy(float)
            z = df[others[2]].to_numpy(float)
        else:
            raise ValueError("x/y/z axes not found")

    out = pd.DataFrame({"time_s": t, "x": x, "y": y, "z": z})
    out = out.drop_duplicates(subset="time_s").sort_values("time_s").reset_index(drop=True)
    return out

def _estimate_hz(t):
    if len(t) < 2: return np.nan
    dt = np.diff(t); dt = dt[(dt > 0) & np.isfinite(dt)]
    return 1.0/np.median(dt) if len(dt) else np.nan

def _resample(df: pd.DataFrame, hz: int) -> pd.DataFrame:
    if df is None or df.empty: return df
    t0, t1 = df["time_s"].iloc[0], df["time_s"].iloc[-1]
    step = 1.0 / hz
    new_t = np.arange(t0, t1 + 1e-9, step)
    out = pd.DataFrame({"time_s": new_t})
    for c in ("x","y","z"):
        out[c] = np.interp(new_t, df["time_s"].to_numpy(), df[c].to_numpy())
    return out

def _merge(acc: pd.DataFrame, gyr: pd.DataFrame, tol: float) -> pd.DataFrame:
    """Nearest-time merge; returns [time_s, ax, ay, az, gx, gy, gz]."""
    if acc is None and gyr is None:
        return pd.DataFrame(columns=["time_s","ax","ay","az","gx","gy","gz"])
    if acc is None:
        g = gyr.rename(columns={"x":"gx","y":"gy","z":"gz"}).copy()
        g[["ax","ay","az"]] = 0.0
        return g[["time_s","ax","ay","az","gx","gy","gz"]]
    if gyr is None:
        a = acc.rename(columns={"x":"ax","y":"ay","z":"az"}).copy()
        a[["gx","gy","gz"]] = 0.0
        return a[["time_s","ax","ay","az","gx","gy","gz"]]

    # nearest merge via merge_asof
    A = acc.sort_values("time_s")
    G = gyr.sort_values("time_s").rename(columns={"x":"gx","y":"gy","z":"gz"})
    m = pd.merge_asof(A, G, on="time_s", tolerance=tol, direction="nearest")
    m = m.dropna(subset=["gx","gy","gz"])
    m = m.rename(columns={"x":"ax","y":"ay","z":"az"})
    return m[["time_s","ax","ay","az","gx","gy","gz"]].reset_index(drop=True)

def _trim(df: pd.DataFrame, sec: float) -> pd.DataFrame:
    if df.empty: return df
    t0 = df["time_s"].iloc[0] + sec
    t1 = df["time_s"].iloc[-1] - sec
    if t1 <= t0: return df.iloc[0:0].copy()
    return df[(df["time_s"] >= t0) & (df["time_s"] <= t1)].reset_index(drop=True)

def unpack_and_clean_dir(raw_dir: str | Path, out_dir: str | Path):
    """Read each ZIP in raw_dir and write a *_cleaned.csv to out_dir."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for zpath in sorted(raw_dir.glob("*.zip")):
        split, activity, rec_id = _parse_name(zpath)
        with zipfile.ZipFile(zpath, "r") as zp:
            acc = _read_sensor(zp, ["acc", "accelerometer"])
            gyr = _read_sensor(zp, ["gyro", "gyroscope"])

        # resample if needed
        if acc is not None:
            hz = _estimate_hz(acc["time_s"].to_numpy())
            if np.isfinite(hz) and abs(hz - TARGET_HZ) > 1.0:
                acc = _resample(acc, TARGET_HZ)
        if gyr is not None:
            hz = _estimate_hz(gyr["time_s"].to_numpy())
            if np.isfinite(hz) and abs(hz - TARGET_HZ) > 1.0:
                gyr = _resample(gyr, TARGET_HZ)

        m = _merge(acc, gyr, MERGE_TOL_SEC)
        m = _trim(m, EDGE_TRIM_SEC)

        m["activity"] = activity
        m["split"] = split
        m["recording_id"] = rec_id

        m.to_csv(out_dir / f"{zpath.stem}_cleaned.csv", index=False)


def read_split_activity(csv_path):
    """Return (split, activity) from the cleaned CSV (reads 1 row)."""
    s = pd.read_csv(csv_path, usecols=["split","activity"], nrows=1)
    return s.loc[0, "split"], s.loc[0, "activity"]

def estimate_hz_csv(csv_path):
    """Estimate sampling rate (Hz) from a cleaned CSV (median Δt)."""
    s = pd.read_csv(csv_path, usecols=["time_s"])
    t = s["time_s"].to_numpy()
    if len(t) < 2:
        return float("nan")
    dt = t[1:] - t[:-1]
    dt = dt[dt > 0]
    return float(1.0 / pd.Series(dt).median()) if len(dt) else float("nan")

def duration_seconds_csv(csv_path):
    """Duration (seconds) from first to last timestamp in a cleaned CSV."""
    s = pd.read_csv(csv_path, usecols=["time_s"])
    if s.empty:
        return 0.0
    return float(s["time_s"].iloc[-1] - s["time_s"].iloc[0])