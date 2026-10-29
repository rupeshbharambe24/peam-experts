"""Evaluation metrics (ranking + calibration)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def roc_auc(y, s):
    return float(roc_auc_score(y, s))


def pr_auc(y, s):
    return float(average_precision_score(y, s))


def brier(y, s):
    return float(brier_score_loss(y, np.clip(s, 0, 1)))


def ece(y, s, n_bins: int = 15):
    s = np.clip(np.asarray(s, float), 0, 1)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(s, edges[1:-1])
    err, n = 0.0, len(s)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            err += (m.sum() / n) * abs(y[m].mean() - s[m].mean())
    return float(err)


def all_metrics(y, s):
    return {"roc_auc": roc_auc(y, s), "pr_auc": pr_auc(y, s),
            "brier": brier(y, s), "ece": ece(y, s)}
