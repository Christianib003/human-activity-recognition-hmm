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


def _logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    s = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(s, axis=axis)

def _emission_logprob(X, means, vars_):
    """Return obs_logprob [T,C] for diagonal Gaussians."""
    T, D = X.shape
    C = means.shape[0]
    out = np.empty((T, C), dtype=float)
    const = -0.5 * (D * np.log(2.0 * np.pi))
    for c in range(C):
        inv = 1.0 / vars_[c]
        diff = X - means[c]
        quad = np.sum(diff * diff * inv, axis=1)
        logdet = np.sum(np.log(vars_[c]))
        out[:, c] = const - 0.5 * quad - 0.5 * logdet
    return out

def _forward_backward_log(obs_log, logA, logpi):
    """
    Forward-backward in log-space.
    Returns: loglik, gamma [T,C], xi [T-1,C,C]
    """
    T, C = obs_log.shape
    alpha = np.empty((T, C))
    beta  = np.empty((T, C))
    # forward
    alpha[0] = logpi + obs_log[0]
    for t in range(1, T):
        alpha[t] = obs_log[t] + _logsumexp(alpha[t-1][:, None] + logA, axis=0)
    loglik = _logsumexp(alpha[-1], axis=0)

    # backward
    beta[-1] = 0.0
    for t in range(T-2, -1, -1):
        beta[t] = _logsumexp(logA + (obs_log[t+1] + beta[t+1])[None, :], axis=1)

    # posteriors
    gamma = alpha + beta - loglik  # log gamma
    gamma = np.exp(gamma)
    # xi
    xi = np.empty((T-1, C, C))
    for t in range(T-1):
        m = alpha[t][:, None] + logA + (obs_log[t+1] + beta[t+1])[None, :]
        m -= _logsumexp(m, axis=None)  # normalize
        xi[t] = np.exp(m)
    return float(loglik), gamma, xi

def em_baum_welch(train_df: pd.DataFrame, feat_cols, init_params,
                  max_iter=25, tol=1e-3, var_floor=1e-3):
    """
    Run EM over labeled *sequences* (we ignore labels; use only order).
    Returns: params_em, loglik_list
    """
    C = len(STATES)
    means = init_params["means"].copy()
    vars_ = init_params["vars"].copy()
    A     = init_params["A"].copy()
    pi    = init_params["pi"].copy()

    loglik_hist = []
    for it in range(max_iter):
        # E-step accumulators
        gamma_sum   = np.zeros(C)
        xi_sum      = np.zeros((C, C))
        mu_num      = np.zeros((C, len(feat_cols)))
        var_num     = np.zeros((C, len(feat_cols)))
        pi_accum    = np.zeros(C)

        total_loglik = 0.0

        logA  = np.log(A + 1e-12)
        logpi = np.log(pi + 1e-12)

        for rec_id, g in train_df.groupby("recording_id", sort=False):
            g = g.sort_values("t_start")
            X = g[feat_cols].to_numpy()

            obs_log = _emission_logprob(X, means, vars_)
            loglik, gamma, xi = _forward_backward_log(obs_log, logA, logpi)

            total_loglik += loglik
            # posteriors
            gamma_sum += gamma.sum(axis=0)
            xi_sum    += xi.sum(axis=0)
            pi_accum  += gamma[0]

            # means/vars numerators
            mu_num  += gamma.T @ X
            # var numerator: sum_c,t gamma_tc * (x_t - mu_c)^2
            # (we use current means for E-step; M-step will divide by gamma_sum)
            for c in range(C):
                diff = X - means[c]
                var_num[c] += (gamma[:, c][:, None] * (diff * diff)).sum(axis=0)

        # M-step
        # transitions
        A = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)
        # initial
        pi = pi_accum / np.maximum(pi_accum.sum(), 1e-12)
        # emissions
        means = mu_num / np.maximum(gamma_sum[:, None], 1e-12)
        vars_ = var_num / np.maximum(gamma_sum[:, None], 1e-12)
        vars_[vars_ < var_floor] = var_floor

        loglik_hist.append(total_loglik)
        if it > 0 and (total_loglik - loglik_hist[-2]) < tol:
            break

    return {"means": means, "vars": vars_, "A": A, "pi": pi}, loglik_hist