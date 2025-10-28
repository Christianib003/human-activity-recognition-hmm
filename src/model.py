import pandas as pd
from src.config import STATES

META_COLS = ["activity", "split", "recording_id", "t_start", "t_end"]

def get_feature_cols(df: pd.DataFrame):
    """Return model feature columns (exclude meta/labels)."""
    return [c for c in df.columns if c not in META_COLS]

def fit_standardizer(df: pd.DataFrame, cols):
    """Compute train means/stds for Z-score (std zeros -> 1.0)."""
    mean = df[cols].mean()
    std = df[cols].std(ddof=0).replace(0, 1.0)
    return {"mean": mean, "std": std, "cols": list(cols)}

def apply_standardizer(df: pd.DataFrame, stats):
    """Apply Z-score using provided stats (returns a copy)."""
    cols = stats["cols"]
    out = df.copy()
    out[cols] = (out[cols] - stats["mean"]) / stats["std"]
    return out
