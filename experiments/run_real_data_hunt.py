"""Real-data hunt: where does the localized-high-order regime actually occur?

The synthetic study (run_interaction_locality.py) shows PEAM wins when
interactions are high-order AND localized within a known grouping. This script
tests that prediction on four public datasets that span the possibilities, and
writes results/real_data_hunt.md:

  satimage  (imbalanced-learn)  : engineered features, low-order -> EBM wins
  gas-drift (OpenML)            : near-separable -> everything saturates
  HIGGS     (OpenML, subsample) : high-order but DELOCALIZED (cross-object
                                  invariant masses) -> PEAM loses (the contrast)
  MADELON   (OpenML)            : high-order AND localized (5 informative feats,
                                  mutually correlated via 15 redundant) -> PEAM
                                  is the best INTERPRETABLE model

Needs network for the OpenML datasets (fetch_openml, no credentials). CPU-only.
Run:  ue-py experiments/run_real_data_hunt.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
from peam import metrics, datasets  # noqa: E402
from peam.model import PatternExpertAdditiveModel  # noqa: E402

from interpret.glassbox import ExplainableBoostingClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

CACHE = os.path.join(ROOT, "data", "openml_cache")


def _auc(y, s):
    return metrics.roc_auc(y, s)


def _split(X, y, seed):
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=seed)
    return Xtr, Xval, Xte, ytr, yval, yte


def _ebm(Xtr, ytr, seed):
    return ExplainableBoostingClassifier(random_state=seed).fit(Xtr, ytr)


def _xgb(Xtr, ytr, seed):
    return XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1,
                         random_state=seed).fit(Xtr, ytr)


# ---------------------------------------------------------------- satimage
def do_satimage(seeds):
    ds = datasets.load("satimage", data_home=os.path.join(ROOT, "data"))
    pixel = [np.arange(4 * i, 4 * i + 4) for i in range(9)]   # 9 pixels x 4 bands (pixel-major)
    rows = []
    for sd in seeds:
        Xtr, Xval, Xte, ytr, yval, yte = _split(ds.X, ds.y, sd)
        ebm = _ebm(Xtr, ytr, sd)
        pm = PatternExpertAdditiveModel(groups=pixel, random_state=sd).fit(Xtr, ytr, Xval, yval)
        pc = PatternExpertAdditiveModel(n_groups=9, grouping="correlation", random_state=sd).fit(Xtr, ytr, Xval, yval)
        rows.append({"ebm": _auc(yte, ebm.predict_proba(Xte)[:, 1]),
                     "peam_semantic": _auc(yte, pm.predict_proba(Xte)[:, 1]),
                     "peam_corr": _auc(yte, pc.predict_proba(Xte)[:, 1])})
    return pd.DataFrame(rows).mean().to_dict()


# --------------------------------------------------------------- gas-drift
def do_gas_drift(seeds):
    d = fetch_openml("gas-drift", version=1, as_frame=False, parser="liac-arff", data_home=CACHE)
    X = np.asarray(d.data, float)
    ym = np.asarray(d.target).astype(int)
    y = (ym == 1).astype(int)  # class 1 vs rest (representative; all tasks saturate)
    groups = [np.arange(8 * i, 8 * i + 8) for i in range(16)]  # 16 sensors x 8 descriptors
    rows = []
    for sd in seeds:
        Xtr, Xval, Xte, ytr, yval, yte = _split(X, y, sd)
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")).fit(Xtr, ytr)
        ebm = _ebm(Xtr, ytr, sd)
        pm = PatternExpertAdditiveModel(groups=groups, random_state=sd).fit(Xtr, ytr, Xval, yval)
        rows.append({"logreg": _auc(yte, lr.predict_proba(Xte)[:, 1]),
                     "ebm": _auc(yte, ebm.predict_proba(Xte)[:, 1]),
                     "peam_semantic": _auc(yte, pm.predict_proba(Xte)[:, 1])})
    return pd.DataFrame(rows).mean().to_dict()


# ------------------------------------------------------------------- HIGGS
def do_higgs(seeds, n_sub=40000):
    d = fetch_openml("higgs", version=2, as_frame=False, parser="liac-arff", data_home=CACHE)
    X_all = np.asarray(d.data, float)
    y_all = np.asarray(d.target).astype(int)
    ok = ~np.isnan(X_all).any(axis=1)
    X_all, y_all = X_all[ok], y_all[ok]
    low = list(range(21))
    obj = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5, 6, 7, 8]),
           np.array([9, 10, 11, 12]), np.array([13, 14, 15, 16]), np.array([17, 18, 19, 20])]
    rows = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        idx = rng.choice(len(y_all), size=min(n_sub, len(y_all)), replace=False)
        X, y = X_all[idx], y_all[idx]
        Xtr, Xval, Xte, ytr, yval, yte = _split(X, y, sd)
        ebm_low = _ebm(Xtr[:, low], ytr, sd)
        xgb_low = _xgb(Xtr[:, low], ytr, sd)
        pm = PatternExpertAdditiveModel(groups=obj, random_state=sd).fit(Xtr[:, low], ytr, Xval[:, low], yval)
        ebm_all = _ebm(Xtr, ytr, sd)  # includes the 7 cross-object mass features
        rows.append({"ebm_lowlevel": _auc(yte, ebm_low.predict_proba(Xte[:, low])[:, 1]),
                     "peam_perobject": _auc(yte, pm.predict_proba(Xte[:, low])[:, 1]),
                     "xgb_lowlevel": _auc(yte, xgb_low.predict_proba(Xte[:, low])[:, 1]),
                     "ebm_with_masses": _auc(yte, ebm_all.predict_proba(Xte)[:, 1])})
    return pd.DataFrame(rows).mean().to_dict()


# ----------------------------------------------------------------- MADELON
def do_madelon(seeds):
    d = fetch_openml("madelon", version=1, as_frame=False, parser="liac-arff", data_home=CACHE)
    X = np.asarray(d.data, float)
    y = (np.asarray(d.target) == np.unique(d.target)[1]).astype(int)
    rows = []
    for sd in seeds:
        Xtr, Xval, Xte, ytr, yval, yte = _split(X, y, sd)
        ebm = _ebm(Xtr, ytr, sd)
        xgb = _xgb(Xtr, ytr, sd)
        pc = PatternExpertAdditiveModel(n_groups=20, grouping="correlation", random_state=sd).fit(Xtr, ytr, Xval, yval)
        # shared feature selection (fair interpretable-tier comparison)
        sel = SelectFromModel(ExtraTreesClassifier(n_estimators=400, n_jobs=-1, random_state=sd),
                              threshold="2*mean").fit(Xtr, ytr)
        m = sel.get_support()
        Xtr_s, Xval_s, Xte_s = Xtr[:, m], Xval[:, m], Xte[:, m]
        ebm_s = _ebm(Xtr_s, ytr, sd)
        xgb_s = _xgb(Xtr_s, ytr, sd)
        k = int(m.sum())
        pm_s = PatternExpertAdditiveModel(groups=[np.arange(k)], random_state=sd).fit(Xtr_s, ytr, Xval_s, yval)
        rows.append({"ebm_full": _auc(yte, ebm.predict_proba(Xte)[:, 1]),
                     "xgb_full": _auc(yte, xgb.predict_proba(Xte)[:, 1]),
                     "peam_corr_full": _auc(yte, pc.predict_proba(Xte)[:, 1]),
                     "ebm_fs": _auc(yte, ebm_s.predict_proba(Xte_s)[:, 1]),
                     "xgb_fs": _auc(yte, xgb_s.predict_proba(Xte_s)[:, 1]),
                     "peam_fs": _auc(yte, pm_s.predict_proba(Xte_s)[:, 1])})
    return pd.DataFrame(rows).mean().to_dict()


DATASETS = {"satimage": do_satimage, "gas-drift": do_gas_drift, "higgs": do_higgs, "madelon": do_madelon}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for name in args.datasets:
        print(f"[{name}] running...", flush=True)
        out[name] = DATASETS[name](args.seeds)
        print(f"[{name}] {', '.join(f'{k}={v:.4f}' for k, v in out[name].items())}", flush=True)

    lines = ["# Real-data hunt: where does the localized-high-order regime occur?",
             "",
             f"Seeds: {args.seeds}. ROC-AUC, mean over seeds. `peam_*` are interpretable; "
             "`xgb_*` is the black-box reference.", ""]
    notes = {
        "satimage": "Engineered remote-sensing features (9 pixels x 4 bands). Low-order: "
                    "**EBM wins**, semantic grouping ~ correlation grouping. Not in the regime.",
        "gas-drift": "16 sensors x 8 descriptors. Near-separable: even logistic regression is "
                     "~0.99. Saturated -> no interaction headroom.",
        "higgs": "Low-level = per-object kinematics; `ebm_with_masses` adds the 7 cross-object "
                 "invariant masses. The high-order signal is **cross-object (delocalized)**: "
                 "per-object PEAM loses to EBM, and masses lift EBM far above low-level. The "
                 "contrast case -- high-order but not localized.",
        "madelon": "5 informative features (a localized high-order interaction) + 15 correlated "
                   "redundant + 480 noise. **PEAM is the best interpretable model**: beats EBM "
                   "on full features and under shared feature selection, and beats full-feature "
                   "black boxes; only a black-box + feature-selection pipeline (`xgb_fs`) edges "
                   "it, without interpretability.",
    }
    for name in args.datasets:
        lines.append(f"## {name}")
        lines.append("")
        row = out[name]
        lines.append("| model | ROC-AUC |")
        lines.append("|:--|--:|")
        for k, v in row.items():
            lines.append(f"| {k} | {v:.4f} |")
        lines.append("")
        if name in notes:
            lines.append(notes[name])
            lines.append("")
    md = "\n".join(lines)
    outpath = os.path.join(ROOT, "results", "real_data_hunt.md")
    with open(outpath, "w") as f:
        f.write(md)
    print("\nwrote", outpath)
    print("\n" + md)


if __name__ == "__main__":
    main()
