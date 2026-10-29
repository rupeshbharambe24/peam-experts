"""Per-group calibrated experts.

Each expert is a gradient-boosted model trained on one feature group, then
calibrated on a held-out validation split so it outputs a probability. Platt
(sigmoid) calibration is the default because it is robust under extreme class
imbalance (few positives); isotonic is available.
"""
from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator


def fit_expert(X_tr, y_tr, X_val, y_val, method: str = "sigmoid", seed: int = 0):
    """Fit a calibrated expert on one feature group.

    HistGBM is fit on train; the calibrator is then fit on validation against a
    frozen base model, so the returned object's predict_proba is a calibrated
    P(y=1) with no leakage from the calibration split into the base fit.
    """
    base = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=seed,
    )
    base.fit(X_tr, y_tr)
    cal = CalibratedClassifierCV(FrozenEstimator(base), method=method)
    cal.fit(X_val, y_val)
    return cal


def expert_proba(expert, X) -> np.ndarray:
    return expert.predict_proba(X)[:, 1]
