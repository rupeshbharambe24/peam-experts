"""Public imbalanced binary-classification datasets (imbalanced-learn / Zenodo).

Targets mapped to {0,1} with the minority class = 1. No credentials required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_DATASETS = [
    "abalone_19",    # ~0.77% positive, biology
    "protein_homo",  # ~0.90% positive, bioinformatics (large)
    "mammography",   # ~2.32% positive, medical imaging
    "oil",           # ~4.4% positive, physics
    "ozone_level",   # ~2.9% positive, environment
    "us_crime",      # ~7.7% positive, social
    "wine_quality",  # ~3.7% positive, chemistry
    "satimage",      # ~9.7% positive, remote sensing
]


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray

    @property
    def n(self):
        return len(self.y)

    @property
    def n_features(self):
        return self.X.shape[1]

    @property
    def pos_rate(self):
        return float(self.y.mean())


def _minority_positive(target):
    target = np.asarray(target).ravel()
    labels, counts = np.unique(target, return_counts=True)
    if len(labels) != 2:
        raise ValueError(f"expected binary target, got {labels}")
    minority = labels[np.argmin(counts)]
    return (target == minority).astype(int)


def load(name, data_home=None):
    from imblearn.datasets import fetch_datasets
    b = fetch_datasets(data_home=data_home, filter_data=(name,), verbose=False)[name]
    return Dataset(name, np.asarray(b.data, float), _minority_positive(b.target))
