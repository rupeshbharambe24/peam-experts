"""PEAM: Pattern-Expert Additive Model with log-odds reason codes.

One calibrated expert per feature group; experts are combined in log-odds
(evidence) space:

    logit(p) = b0 + sum_k  beta_k * [ logit(expert_k) - logit(base_rate) ]

Each term  r_k = beta_k * (logit(expert_k) - logit(base_rate))  is an additive,
signed, exact reason code for group k: the final log-odds is b0 plus the sum of
the group reason codes, with no post-hoc approximation.

Combiners
---------
'logistic' : beta_k and b0 fit by logistic regression on the evidence terms
             (transparent group weights; the default).
'naive'    : beta_k = 1, b0 = logit(base_rate)  (naive-Bayes-style additivity).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .experts import expert_proba, fit_expert
from .grouping import correlation_groups, random_groups

_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, float), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class PatternExpertAdditiveModel:
    def __init__(self, n_groups: int = 5, grouping: str = "correlation",
                 combiner: str = "logistic", calib: str = "sigmoid", random_state: int = 0,
                 groups=None):
        self.n_groups = n_groups
        self.grouping = grouping
        self.combiner = combiner
        self.calib = calib
        self.random_state = random_state
        # Optional explicit feature grouping (list of index arrays). When set, it
        # is used directly and n_groups/grouping are ignored -- this is the
        # "domain-assigned groups" path described in the README.
        self.groups = groups

    # ------------------------------------------------------------------ fit
    def fit(self, X_tr, y_tr, X_val, y_val):
        X_tr, X_val = np.asarray(X_tr, float), np.asarray(X_val, float)
        if self.groups is not None:
            self.groups_ = [np.asarray(g, int) for g in self.groups]
        elif self.grouping == "random":
            self.groups_ = random_groups(X_tr.shape[1], self.n_groups, self.random_state)
        else:
            self.groups_ = correlation_groups(X_tr, self.n_groups, self.random_state)

        self.base_rate_ = float(np.mean(y_tr))
        self.logit_base_ = _logit(self.base_rate_)

        self.experts_ = []
        for g in self.groups_:
            exp = fit_expert(X_tr[:, g], y_tr, X_val[:, g], y_val,
                             method=self.calib, seed=self.random_state)
            self.experts_.append((g, exp))

        E_val = self._evidence(X_val)
        if self.combiner == "logistic":
            self.combiner_ = LogisticRegression(max_iter=2000, C=1.0)
            self.combiner_.fit(E_val, y_val)
            self.beta_ = self.combiner_.coef_.ravel().copy()
            self.b0_ = float(self.combiner_.intercept_[0])
        else:  # naive additive evidence
            self.combiner_ = None
            self.beta_ = np.ones(len(self.groups_))
            self.b0_ = self.logit_base_
        return self

    # ----------------------------------------------------------- internals
    def _evidence(self, X):
        """Per-group evidence = logit(expert_k) - logit(base_rate)."""
        X = np.asarray(X, float)
        cols = [_logit(expert_proba(exp, X[:, g])) - self.logit_base_
                for g, exp in self.experts_]
        return np.column_stack(cols)

    # -------------------------------------------------------------- predict
    def decision_logit(self, X):
        E = self._evidence(X)
        return self.b0_ + E @ self.beta_

    def predict_proba(self, X):
        p = _sigmoid(self.decision_logit(X))
        return np.column_stack([1 - p, p])

    def reason_codes(self, X):
        """(n_samples, n_groups) additive log-odds contribution of each group."""
        E = self._evidence(X)
        return E * self.beta_[None, :]

    def group_importance(self, X):
        """Global importance = mean |reason code| per group."""
        return np.mean(np.abs(self.reason_codes(X)), axis=0)
