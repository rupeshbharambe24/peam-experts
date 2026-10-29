"""Synthetic data with controllable interaction *order* and *locality*.

The generator builds a target whose signal is a sum of per-group terms, where
each group's term is a pure order-``r`` multiplicative interaction of that
group's features:

    z_k = standardize( x[k,0] * x[k,1] * ... * x[k,r-1] )
    logit(p) = intercept + sum_k  w_k * z_k
    y ~ Bernoulli(sigmoid(logit(p)))

Properties that make this a clean probe of a model's inductive bias:

* Groups use disjoint, independent features -> **no cross-group interaction**;
  the log-odds is *exactly additive across groups* (PEAM's structural
  assumption holds by construction).
* A product of ``r`` independent mean-zero features has zero correlation with
  any single feature and with any lower-order product of its own features, so
  the group signal requires modelling the **full order-``r`` interaction** --
  every strict subset carries no marginal signal.

Consequences for the models under test:

* order 1  -> linear; logistic regression / GAM / EBM all suffice.
* order 2  -> pairwise; EBM (additive + pairwise = GA2M) captures it.
* order >=3 -> EBM *cannot represent it* (no 3-way terms); a flexible
  within-group learner (PEAM's gradient-boosted expert) or a black box can.

``intercept`` is solved to hit a target positive rate, so the task can be made
imbalanced.
"""
from __future__ import annotations

import numpy as np


def _standardize(z):
    z = np.asarray(z, float)
    s = z.std()
    return (z - z.mean()) / (s if s > 0 else 1.0)


def _solve_intercept(eta_no_b, pos_rate, rng, iters=60):
    """Bisection for the intercept b s.t. mean(sigmoid(b + eta)) == pos_rate."""
    lo, hi = -40.0, 40.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = 1.0 / (1.0 + np.exp(-(mid + eta_no_b)))
        if p.mean() < pos_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def make_grouped_interaction(
    n=8000,
    n_groups=4,
    feats_per_group=4,
    order=3,
    n_noise=6,
    pos_rate=0.1,
    signal=3.0,
    random_state=0,
):
    """Generate (X, y, groups).

    Parameters
    ----------
    n : samples.
    n_groups : number of informative feature groups.
    feats_per_group : features in each group (>= ``order``); extra within-group
        features are irrelevant, so the group's expert must also select.
    order : interaction order of each group's signal (1 = linear).
    n_noise : extra pure-noise features appended after the groups.
    pos_rate : target fraction of positives (imbalance knob).
    signal : overall signal strength (scales the per-group weights).
    random_state : seed.

    Returns
    -------
    X : (n, n_groups*feats_per_group + n_noise) float array.
    y : (n,) int array in {0,1}.
    groups : list of index arrays -- the *true* feature grouping (noise features
        are not included in any group).
    """
    rng = np.random.default_rng(random_state)
    order = int(min(order, feats_per_group))
    blocks, groups, eta = [], [], np.zeros(n)
    w = signal / np.sqrt(n_groups)

    col = 0
    for _ in range(n_groups):
        Xg = rng.standard_normal((n, feats_per_group))
        blocks.append(Xg)
        groups.append(np.arange(col, col + feats_per_group))
        col += feats_per_group
        prod = np.prod(Xg[:, :order], axis=1)  # pure order-r interaction
        eta = eta + w * _standardize(prod)

    if n_noise > 0:
        blocks.append(rng.standard_normal((n, n_noise)))

    X = np.concatenate(blocks, axis=1)
    b = _solve_intercept(eta, pos_rate, rng)
    p = 1.0 / (1.0 + np.exp(-(b + eta)))
    y = (rng.random(n) < p).astype(int)
    return X, y, groups
