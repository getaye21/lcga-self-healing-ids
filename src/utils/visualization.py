"""Shared plotting utilities for thesis figures."""
import matplotlib.pyplot as plt, seaborn as sns, numpy as np
from sklearn.metrics import confusion_matrix

PALETTE = ["#1F3864","#2E5090","#3B8BD4","#E24B4A","#1D9E75","#854F0B"]

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, figsize=(13,10)):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (row-normalised)", fontweight="bold")
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_training_history(history, save_path=None):
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for ax, metric, title in zip(axes, ["loss","accuracy"], ["Loss","Accuracy"]):
        ax.plot(history.history[metric], label="Train", color=PALETTE[2], lw=2)
        ax.plot(history.history[f"val_{metric}"], label="Val", color=PALETTE[3], lw=2, ls="--")
        ax.set_title(f"Training {title}", fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_per_class_f1(class_names, f1_scores, model_name="LCGA", save_path=None):
    idxs = np.argsort(f1_scores)
    colors = [PALETTE[3] if f < 0.85 else PALETTE[4] for f in np.array(f1_scores)[idxs]]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh([class_names[i] for i in idxs], [f1_scores[i] for i in idxs], color=colors)
    ax.axvline(0.85, color="#999", ls="--", label="0.85 threshold")
    ax.set_xlim(0,1.05); ax.set_xlabel("F1-score"); ax.legend()
    ax.set_title(f"Per-Class F1 — {model_name}", fontweight="bold")
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

def plot_comparison_bar(models, values, metric="Macro F1", save_path=None):
    fig, ax = plt.subplots(figsize=(9,4))
    bars = ax.bar(models, values, color=PALETTE[:len(models)])
    ax.set_ylim(0,1.05); ax.set_ylabel(metric); ax.grid(axis="y", alpha=0.3)
    ax.set_title(f"Model Comparison — {metric}", fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.4f}", ha="center", fontsize=9)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
