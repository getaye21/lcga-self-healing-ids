"""Evaluation metrics for detection and classification."""
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, confusion_matrix)
import numpy as np, pandas as pd

def detection_metrics(y_true, y_pred, y_prob=None):
    cm = confusion_matrix(y_true, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    fpr = float(np.mean(np.divide(fp, fp+tn, where=(fp+tn)>0)))
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_prec": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "fpr": fpr,
    }
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        m["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
    return m

def compute_isr(action_log):
    successes = sum(1 for e in action_log if e.get("outcome") == "success")
    return successes / len(action_log) if action_log else 0.0

def build_comparison_table(results_dict):
    rows = []
    for name, m in results_dict.items():
        rows.append({
            "Model": name,
            "Accuracy": f"{m['accuracy']:.4f}",
            "Macro F1": f"{m['macro_f1']:.4f}",
            "Weighted F1": f"{m['weighted_f1']:.4f}",
            "MCC": f"{m['mcc']:.4f}",
            "FPR": f"{m['fpr']:.4f}",
        })
    return pd.DataFrame(rows)
