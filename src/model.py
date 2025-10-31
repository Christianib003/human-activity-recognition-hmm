import pandas as pd
import numpy as np
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

def _variance_floor(diag_var, floor=1e-3):
    v = np.asarray(diag_var, dtype=float)
    v[v < floor] = floor
    return v

def supervised_init_params(train_df: pd.DataFrame, feat_cols, cov_floor=1e-3):
    """
    Estimate initial HMM parameters from labeled windows:
      - means[c], diag_vars[c] from windows with activity == STATES[c]
      - initial pi from first window per recording
      - transitions A from consecutive labeled windows within each recording
    Returns dict with keys: 'means','vars','A','pi' (all numpy)
    """
    C = len(STATES)
    D = len(feat_cols)
    means = np.zeros((C, D), float)
    vars_ = np.ones((C, D), float)
    # emissions
    for c, name in enumerate(STATES):
        Xc = train_df.loc[train_df["activity"] == name, feat_cols].to_numpy()
        if len(Xc) == 0:
            means[c] = 0.0
            vars_[c] = 1.0
        else:
            means[c] = Xc.mean(axis=0)
            vars_[c] = _variance_floor(Xc.var(axis=0, ddof=0), cov_floor)

    # initial pi and transitions
    pi_counts = np.ones(C, float)  # +1 smoothing
    A_counts  = np.ones((C, C), float)  # +1 smoothing
    for rec_id, g in train_df.groupby("recording_id", sort=False):
        g = g.sort_values("t_start")
        y = g["activity"].map({name:i for i,name in enumerate(STATES)}).to_numpy()
        if len(y) == 0: 
            continue
        pi_counts[y[0]] += 1.0
        for i in range(len(y)-1):
            A_counts[y[i], y[i+1]] += 1.0
    pi = pi_counts / pi_counts.sum()
    A  = A_counts / A_counts.sum(axis=1, keepdims=True)
    return {"means": means, "vars": vars_, "A": A, "pi": pi}

def _log_gaussian_diag(X, mean, var):
    """
    Log N(X | mean, diag(var)) for a batch X [T,D] vs one state.
    """
    D = X.shape[1]
    inv = 1.0 / var
    diff = X - mean
    quad = np.sum(diff*diff*inv, axis=1)
    logdet = np.sum(np.log(var))
    return -0.5*(quad + logdet + D*np.log(2.0*np.pi))

def viterbi_log(obs_logprob, logA, logpi):
    """
    Viterbi on log-domain:
      obs_logprob: [T, C]
      logA: [C, C], log transitions
      logpi: [C],  log initial
    Returns backpointer path (ints length T).
    """
    T, C = obs_logprob.shape
    dp = np.full((T, C), -np.inf)
    bp = np.zeros((T, C), dtype=int)
    dp[0] = logpi + obs_logprob[0]
    for t in range(1, T):
        # dp[t-1][:,None] + logA -> [C,C], take max over prev-state
        M = dp[t-1][:,None] + logA
        bp[t] = np.argmax(M, axis=0)
        dp[t] = M[bp[t], np.arange(C)] + obs_logprob[t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(dp[-1]))
    for t in range(T-2, -1, -1):
        path[t] = bp[t+1, path[t+1]]
    return path

def decode_many(df: pd.DataFrame, feat_cols, params):
    """
    Run Viterbi per recording_id (preserving time order).
    Returns a copy with predicted integer 'y_pred' and name 'pred_activity'.
    """
    C = len(STATES)
    out_parts = []
    means = params["means"]; vars_ = params["vars"]
    logA  = np.log(params["A"] + 1e-12)
    logpi = np.log(params["pi"] + 1e-12)
    for rec_id, g in df.groupby("recording_id", sort=False):
        g = g.sort_values("t_start").copy()
        X = g[feat_cols].to_numpy()
        # emission log-probs per state
        obs_log = np.zeros((len(g), C), float)
        for c in range(C):
            obs_log[:, c] = _log_gaussian_diag(X, means[c], vars_[c])
        path = viterbi_log(obs_log, logA, logpi)
        g["y_pred"] = path
        idx2name = {i:name for i,name in enumerate(STATES)}
        g["pred_activity"] = g["y_pred"].map(idx2name)
        out_parts.append(g)
    return pd.concat(out_parts, ignore_index=True) if out_parts else df.copy()


def confusion_from_decoded(df_dec: pd.DataFrame):
    """Return confusion matrix (C×C, y=true rows, yhat=cols) and class order."""
    idx = {s:i for i,s in enumerate(STATES)}
    y_true = df_dec["activity"].map(idx).to_numpy()
    y_pred = df_dec["pred_activity"].map(idx).to_numpy()
    C = len(STATES)
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm, list(STATES)

def class_metrics_from_cm(cm: np.ndarray):
    """Per-class sensitivity (recall) & specificity, and overall accuracy."""
    C = cm.shape[0]
    totals = cm.sum()
    per = []
    for i in range(C):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = totals - TP - FN - FP
        sens = TP / (TP + FN) if (TP + FN) else 0.0
        spec = TN / (TN + FP) if (TN + FP) else 0.0
        per.append({"activity": STATES[i], "sensitivity": sens, "specificity": spec})
    overall_acc = np.trace(cm) / totals if totals else 0.0
    return pd.DataFrame(per), overall_acc


