# PEAM — Pattern-Expert Additive Model with Log-Odds Reason Codes

An intrinsically interpretable ensemble for imbalanced binary classification.
Features are partitioned into coherent **groups**; one calibrated expert is
trained per group; the experts are combined in **log-odds (evidence) space** so
that every prediction decomposes into additive, signed, per-group **reason
codes** — exactly, with no post-hoc approximation (unlike SHAP/LIME).

## The model

Group features into `K` patterns, train a calibrated expert per group, then:

```
logit(p) = b0 + Σ_k  β_k · [ logit(expert_k(x)) − logit(base_rate) ]
```

Each term `r_k = β_k · (logit(expert_k) − logit(base_rate))` is the reason code
for group *k*: the decision log-odds is exactly `b0 + Σ_k r_k`. Groups can be
switched **on/off** at deployment (portability across settings that lack some
feature groups), and the combiner weights `β_k` are transparent.

```python
from peam import PatternExpertAdditiveModel

m = PatternExpertAdditiveModel(n_groups=5, combiner="logistic").fit(X_tr, y_tr, X_val, y_val)
p  = m.predict_proba(X_test)[:, 1]
rc = m.reason_codes(X_test)        # (n, K) additive per-group log-odds contributions
imp = m.group_importance(X_val)    # global group importances
```

## What the experiments test

PEAM is compared against the interpretable floor (logistic regression), the
glass-box SOTA (**EBM** / Explainable Boosting Machine), and black-box ceilings
(gradient boosting, and a tree ensemble), on ROC-AUC, PR-AUC, Brier and ECE —
plus two ablations (naive β=1 combiner, random grouping) and a **reason-code
fidelity** check (AUC drop when the top group's evidence is removed). The
honest question is the *accuracy–interpretability trade-off*: how small is the
gap to the black box, and does PEAM match EBM while offering modular group-level
reason codes?

## Findings (honest summary)

Across 8 domain-diverse public imbalanced datasets × 3 seeds
([`results/RESULTS.md`](results/RESULTS.md)):

- ✅ PEAM produces **exact** additive reason codes (they sum to the decision
  log-odds), is **very well calibrated** (ECE 0.0144 — essentially tied with EBM,
  the best), beats the logistic-regression interpretable floor (+0.02 ROC-AUC),
  and its reason codes are **faithful** (removing the top group's evidence drops
  AUC ~0.14 on average).
- ❌ On **accuracy it is dominated by EBM**, which is also intrinsically
  interpretable yet more accurate (ROC-AUC 0.909 vs 0.888; PR-AUC 0.549 vs 0.493)
  at equal calibration.
- The gap is the **price of modularity**: forcing additivity across independent
  group-experts discards cross-group interactions. Accuracy falls monotonically
  as the number of groups grows, and at `n_groups=1` PEAM matches/beats EBM
  ([`results/ngroups_tradeoff.md`](results/ngroups_tradeoff.md)). Correlation-based
  grouping only marginally beats random grouping.

**Takeaway (general tabular data):** PEAM's distinctive value over EBM is
**modularity/portability** (group-experts can be switched on/off; any model per
group) and group-level reason codes, bought at a measurable, dataset-dependent
accuracy cost. As a general-purpose interpretable model judged purely on
accuracy it does not beat EBM.

### A regime where PEAM genuinely wins: localized high-order interactions

There is a specific, identifiable regime where PEAM is not merely competitive
but the **single best model — beating even black-box gradient boosting — while
staying interpretable**. EBM is additive + pairwise (GA2M) and *cannot
represent* interactions of order >= 3. PEAM keeps a flexible gradient-boosted
expert per group, so it captures arbitrary-order interactions **within** a group
while remaining additive **across** groups. On data whose signal is high-order
interactions *localized inside feature groups*, PEAM's structural prior is
exactly right (see [`results/interaction_locality.md`](results/interaction_locality.md)
and [`results/interaction_locality_robustness.md`](results/interaction_locality_robustness.md)):

- ROC-AUC at interaction order 3 (4 groups, 10% positives): **PEAM 0.795 vs EBM
  0.668 vs black-box GBM 0.731**. The win over a *strong, tuned* HistGBM/XGBoost
  holds from n=2k to n=80k.
- At order 1 (linear) and order 2 (pairwise) PEAM is dominated (by logistic
  regression and EBM respectively) — the honest other side of the frontier.
- **Caveat:** the win requires the right grouping. Pure high-order interactions
  induce zero pairwise correlation, so correlation-based grouping cannot find
  them — a domain grouping or an interaction-aware discovery method is needed.
- On the first real dataset tried (satimage, with its genuine pixel/band
  grouping) EBM still wins: its signal is additive + pairwise, so it is *not* in
  this regime. PEAM is a **targeted tool** for localized-high-order structure,
  not a general SOTA; whether and where this regime occurs in real data is an
  open, ongoing search.

## Reproduce

```bash
pip install -r requirements.txt
python experiments/run_proof_of_signal.py --seeds 0 1 2
```

Datasets are public and fetched automatically (no credentials) from the
`imbalanced-learn` benchmark collection, spanning domains (medical, biology,
chemistry, physics, remote-sensing, social, environmental) and imbalance ratios.

## Layout

```
src/peam/     grouping.py, experts.py, model.py (PEAM), metrics.py, datasets.py
experiments/  run_proof_of_signal.py
results/      generated tables (CSV + Markdown)
```

## License

MIT — see [LICENSE](LICENSE).
