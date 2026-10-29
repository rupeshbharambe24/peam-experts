"""Partition features into coherent groups ("patterns").

For datasets with semantic feature names, groups would be assigned by domain.
For generic / anonymised features we group by correlation structure: features
that move together land in the same expert. This is reproducible and
data-driven. A random grouping is provided as an ablation control.
"""
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


def correlation_groups(X: np.ndarray, n_groups: int, random_state: int = 0) -> list[np.ndarray]:
    """Group feature indices by hierarchical clustering on |Spearman correlation|."""
    n_features = X.shape[1]
    k = int(min(n_groups, n_features))
    if k <= 1:
        return [np.arange(n_features)]

    rho, _ = spearmanr(X)
    rho = np.atleast_2d(rho)
    if rho.shape != (n_features, n_features):  # spearmanr returns scalar for 2 cols
        rho = np.corrcoef(X, rowvar=False)
    rho = np.nan_to_num(rho, nan=0.0)
    dist = 1.0 - np.abs(rho)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0

    Z = linkage(squareform(dist, checks=False), method="average")
    labels = fcluster(Z, t=k, criterion="maxclust")
    groups = [np.where(labels == g)[0] for g in np.unique(labels)]
    return [g for g in groups if len(g) > 0]


def random_groups(n_features: int, n_groups: int, random_state: int = 0) -> list[np.ndarray]:
    """Random balanced partition of feature indices (ablation control)."""
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(n_features)
    k = int(min(n_groups, n_features))
    return [np.sort(a) for a in np.array_split(idx, k)]
