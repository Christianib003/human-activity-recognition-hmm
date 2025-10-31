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