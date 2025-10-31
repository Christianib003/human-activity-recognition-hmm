import numpy as np
import matplotlib.pyplot as plt
from src.config import STATES

def plot_transition_matrix(A: np.ndarray, title="HMM Transition Probabilities (A)"):
    fig, ax = plt.subplots(figsize=(6.5,6.5))
    im = ax.imshow(A, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0, aspect="equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("")

    # ticks & labels
    ax.set_xticks(range(len(STATES)))
    ax.set_yticks(range(len(STATES)))
    ax.set_xticklabels(STATES, rotation=0)
    ax.set_yticklabels(STATES)
    ax.set_xlabel("To State")
    ax.set_ylabel("From State")
    ax.set_title(title, pad=12, fontsize=16)

    # annotate numbers
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            val = A[i, j]
            ax.text(j, i, f"{val:.3f}", va="center", ha="center", color="black", fontsize=10)

    fig.tight_layout()
    plt.show()

def plot_confusion(cm: np.ndarray, labels=None, title="Confusion Matrix"):
    labels = labels or list(STATES)
    # row-normalized percentages for display under counts
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(6.5,6.5))
    im = ax.imshow(cm, cmap=plt.cm.Blues, aspect="equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ticks & labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title, pad=12, fontsize=18)

    # annotate: count + (percent)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:d}\n({pct[i,j]*100:.1f}%)",
                    ha="center", va="center", color="black", fontsize=10, linespacing=1.2)

    fig.tight_layout()
    plt.show()

def plot_timeline(df_seq, title="HMM Decoding: True vs. Predicted Activity"):
    # expects columns: t_start, activity, pred_activity
    idx = {s:i for i,s in enumerate(STATES)}
    t  = df_seq["t_start"].to_numpy()
    yt = np.array([idx[s] for s in df_seq["activity"]])
    yp = np.array([idx[s] for s in df_seq["pred_activity"]])

    fig, ax = plt.subplots(figsize=(12,3.2))
    ax.plot(t, yt, marker="o", linestyle="-", label="True Activity (Ground Truth)")
    ax.plot(t, yp, marker="x", linestyle="--", label="Predicted Activity (Viterbi)")
    ax.set_yticks(range(len(STATES)))
    ax.set_yticklabels(STATES)
    ax.set_xlabel("Time Window")
    ax.set_ylabel("Activity State")
    ax.set_title(title, pad=10, fontsize=16)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    plt.show()

def plot_emission_means(means: np.ndarray, feat_cols, title="HMM Emission Means (per state)"):
    """
    Visualize emission means as heatmap.
    means: [C, D] array where C=num_states, D=num_features
    feat_cols: list of feature names
    """
    C, D = means.shape
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(means, cmap=plt.cm.RdBu_r, aspect="auto")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Standardized Mean Value")
    
    ax.set_yticks(range(C))
    ax.set_yticklabels(STATES)
    ax.set_ylabel("Activity State")
    
    # Show subset of features if too many
    if D > 20:
        tick_step = max(1, D // 20)
        ax.set_xticks(range(0, D, tick_step))
        ax.set_xticklabels([feat_cols[i] for i in range(0, D, tick_step)], rotation=90)
    else:
        ax.set_xticks(range(D))
        ax.set_xticklabels(feat_cols, rotation=90)
    ax.set_xlabel("Feature")
    ax.set_title(title, pad=12, fontsize=16)
    
    fig.tight_layout()
    plt.show()

def plot_emission_top_features(means: np.ndarray, vars_: np.ndarray, feat_cols, top_n=8, title_prefix="Emission Distributions"):
    """
    Plot distributions (mean ± std) for top N most discriminative features.
    Discriminativeness = variance of means across states.
    """
    C, D = means.shape
    # Find most discriminative features
    discriminative = np.var(means, axis=0)  # variance across states
    top_idx = np.argsort(discriminative)[-top_n:][::-1]
    
    n_cols = 4
    n_rows = int(np.ceil(top_n / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten() if top_n > 1 else [axes]
    
    for plot_idx, feat_idx in enumerate(top_idx):
        ax = axes[plot_idx]
        feat_name = feat_cols[feat_idx]
        
        x = np.arange(C)
        y = means[:, feat_idx]
        yerr = np.sqrt(vars_[:, feat_idx])
        
        ax.bar(x, y, yerr=yerr, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:C], alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(STATES, rotation=45, ha='right')
        ax.set_ylabel("Standardized Value")
        ax.set_title(f"{feat_name}\n(discrim={discriminative[feat_idx]:.2f})", fontsize=10)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused subplots
    for idx in range(top_n, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f"{title_prefix}: Top {top_n} Discriminative Features", fontsize=16, y=1.00)
    fig.tight_layout()
    plt.show()