"""Proof-of-signal for PEAM.

Question: does a pattern-expert additive model with log-odds reason codes
(a) stay close to black-box accuracy, (b) match the glass-box SOTA (EBM), and
(c) produce faithful reason codes?

Per dataset x seed (stratified 60/20/20 train/val/test):
  compare  logreg | ebm | gbm(black) | ensemble(black) | peam | peam_naive | peam_random
  metrics  ROC-AUC, PR-AUC, Brier, ECE
  plus     reason-code fidelity (AUC drop when the top group's evidence is zeroed).

Run:  ue-py experiments/run_proof_of_signal.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)
from peam import metrics, datasets  # noqa: E402
from peam.model import PatternExpertAdditiveModel, _sigmoid  # noqa: E402

try:
    from interpret.glassbox import ExplainableBoostingClassifier
    HAS_EBM = True
except Exception:  # noqa: BLE001
    HAS_EBM = False

METHODS = ["logreg", "ebm", "gbm", "ensemble", "peam", "peam_naive", "peam_random"]
METRIC_COLS = ["roc_auc", "pr_auc", "brier", "ece"]


def fit_all(X_tr, y_tr, X_val, y_val, seed):
    out = {}

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    lr.fit(X_tr, y_tr)
    out["logreg"] = lambda X, m=lr: m.predict_proba(X)[:, 1]

    if HAS_EBM:
        ebm = ExplainableBoostingClassifier(random_state=seed)
        ebm.fit(X_tr, y_tr)
        out["ebm"] = lambda X, m=ebm: m.predict_proba(X)[:, 1]

    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=seed)
    gbm.fit(X_tr, y_tr)
    out["gbm"] = lambda X, m=gbm: m.predict_proba(X)[:, 1]

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=seed).fit(X_tr, y_tr)
    et = ExtraTreesClassifier(n_estimators=200, class_weight="balanced_subsample",
                              n_jobs=-1, random_state=seed).fit(X_tr, y_tr)
    out["ensemble"] = lambda X, a=gbm, b=rf, c=et: np.mean(
        [a.predict_proba(X)[:, 1], b.predict_proba(X)[:, 1], c.predict_proba(X)[:, 1]], axis=0)

    peam = PatternExpertAdditiveModel(n_groups=5, combiner="logistic", random_state=seed
                                      ).fit(X_tr, y_tr, X_val, y_val)
    out["peam"] = lambda X, m=peam: m.predict_proba(X)[:, 1]

    peam_n = PatternExpertAdditiveModel(n_groups=5, combiner="naive", random_state=seed
                                        ).fit(X_tr, y_tr, X_val, y_val)
    out["peam_naive"] = lambda X, m=peam_n: m.predict_proba(X)[:, 1]

    peam_r = PatternExpertAdditiveModel(n_groups=5, grouping="random", combiner="logistic",
                                        random_state=seed).fit(X_tr, y_tr, X_val, y_val)
    out["peam_random"] = lambda X, m=peam_r: m.predict_proba(X)[:, 1]

    return out, peam


def fidelity(peam, X_val, X_te, y_te):
    """AUC drop when the globally most-important group's evidence is zeroed."""
    imp = peam.group_importance(X_val)
    top = int(np.argmax(imp))
    E = peam._evidence(X_te)
    full = metrics.roc_auc(y_te, _sigmoid(peam.b0_ + E @ peam.beta_))
    beta2 = peam.beta_.copy()
    beta2[top] = 0.0
    ablated = metrics.roc_auc(y_te, _sigmoid(peam.b0_ + E @ beta2))
    return full - ablated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=datasets.DEFAULT_DATASETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    outdir = os.path.join(os.path.dirname(HERE), "results")
    os.makedirs(outdir, exist_ok=True)
    data_home = os.path.join(os.path.dirname(HERE), "data")

    rows, fid = [], []
    for name in args.datasets:
        try:
            ds = datasets.load(name, data_home=data_home)
        except Exception as e:  # noqa: BLE001
            print(f"!! skip {name}: {e}", flush=True)
            continue
        print(f"[{name}] n={ds.n} feat={ds.n_features} pos={ds.pos_rate:.4f}", flush=True)
        for sd in args.seeds:
            Xtr, Xtmp, ytr, ytmp = train_test_split(ds.X, ds.y, test_size=0.4,
                                                    stratify=ds.y, random_state=sd)
            Xval, Xte, yval, yte = train_test_split(Xtmp, ytmp, test_size=0.5,
                                                    stratify=ytmp, random_state=sd)
            preds, peam = fit_all(Xtr, ytr, Xval, yval, sd)
            for m, fn in preds.items():
                rows.append({"dataset": name, "seed": sd, "method": m, **metrics.all_metrics(yte, fn(Xte))})
            d = fidelity(peam, Xval, Xte, yte)
            fid.append({"dataset": name, "seed": sd, "auc_drop_top_group": d})
            auc_peam = metrics.roc_auc(yte, preds["peam"](Xte))
            auc_ens = metrics.roc_auc(yte, preds["ensemble"](Xte))
            print(f"  seed {sd}: peam AUC={auc_peam:.4f} | ensemble={auc_ens:.4f} | "
                  f"gap={auc_ens-auc_peam:+.4f} | top-group AUC drop={d:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "proof_of_signal.csv"), index=False)
    present = [m for m in METHODS if m in df["method"].unique()]
    mean_tbl = df.groupby("method")[METRIC_COLS].mean().reindex(present)

    rank_frames = []
    for _, g in df.groupby(["dataset", "seed"]):
        g = g.set_index("method")
        rank_frames.append(pd.DataFrame({c: g[c].rank(ascending=c in ("brier", "ece")) for c in METRIC_COLS}))
    avg_rank = pd.concat(rank_frames).groupby(level=0).mean().reindex(present)

    fid_df = pd.DataFrame(fid)
    lines = ["# PEAM proof-of-signal results", "",
             f"Datasets: {df['dataset'].nunique()} | seeds: {sorted(df['seed'].unique())} | EBM available: {HAS_EBM}",
             "", "## Mean metric across datasets x seeds", "", mean_tbl.round(4).to_markdown(),
             "", "## Average rank (1 = best)", "", avg_rank.round(2).to_markdown(),
             "", "## Reason-code fidelity (AUC drop when top group's evidence zeroed; higher = codes identify real drivers)",
             "", f"mean AUC drop = {fid_df['auc_drop_top_group'].mean():.4f} "
             f"(std {fid_df['auc_drop_top_group'].std():.4f})",
             "", "## Per-dataset ROC-AUC (mean over seeds)", "",
             df.groupby(["dataset", "method"])["roc_auc"].mean().unstack("method")[present].round(4).to_markdown()]
    md = "\n".join(lines)
    with open(os.path.join(outdir, "RESULTS.md"), "w") as f:
        f.write(md)
    fid_df.to_csv(os.path.join(outdir, "fidelity.csv"), index=False)
    print("\n" + md)


if __name__ == "__main__":
    main()
