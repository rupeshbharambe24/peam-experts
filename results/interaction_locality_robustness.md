# When does PEAM genuinely win? Robustness + first real-data probe

This extends `interaction_locality.md` with (a) a fairness/scale stress test of
the headline claim and (b) a first real-data probe. Short version: the win over
the glass-box SOTA is **structural and robust**, the win over a *black box* in
the localized-high-order regime **survives strong tuning and large samples**,
and whether the regime occurs in the wild is a genuine open question — the first
real dataset tried does **not** exhibit it.

## 1. The headline: localized high-order interactions (synthetic)

ROC-AUC by interaction order, 4 groups x 4 feats + 6 noise, 10% positives, mean
over 3 seeds (from `interaction_locality.md`):

| order | logreg | ebm | peam_true | gbm (black box) |
|------:|-------:|----:|----------:|----------------:|
| 1 (linear)   | 0.936 | 0.930 | 0.868 | 0.918 |
| 2 (pairwise) | 0.515 | 0.908 | 0.856 | 0.890 |
| 3 (localized high-order) | 0.485 | 0.668 | **0.795** | 0.731 |
| 4 (localized high-order) | 0.522 | 0.679 | **0.685** | 0.637 |

At order >= 3, EBM (additive + pairwise = GA2M) **cannot represent** the signal
and saturates; PEAM with the true grouping is the single best model — beating
even the black box — while staying interpretable (exact additive group reason
codes). At order 1-2 PEAM is *dominated* (linear -> logreg; pairwise -> EBM):
the honest other side of the frontier.

## 2. Is "PEAM beats the black box" a tuning / small-n artifact? No.

Order 3, PEAM (true groups) vs a strong early-stopped HistGBM (max_iter=2000,
lr=0.03, 63 leaves) and XGBoost (1500 trees, depth 6), mean over 3 seeds:

| n      | peam_true | ebm | HistGBM (strong) | XGBoost |
|-------:|----------:|----:|-----------------:|--------:|
| 2,000  | 0.651 | 0.636 | 0.603 | 0.622 |
| 8,000  | **0.795** | 0.668 | 0.733 | 0.717 |
| 30,000 | **0.884** | 0.725 | 0.856 | 0.845 |
| 80,000 | **0.905** | 0.746 | 0.889 | 0.886 |

PEAM leads at every sample size. The black box closes the gap as `n` grows (it
eventually learns the block structure from data) but never overtakes the model
that is *told* the structure; EBM never catches up because order-3 terms are
outside its hypothesis class. The advantage is a correct **structural prior**,
not extra capacity — PEAM's within-group learner is itself a small HistGBM.

## 3. The catch: the win needs the *right* grouping

Correlation-based grouping collapses in this regime (`peam_corr`/`peam_random`
in `interaction_locality.md`): a pure order-r product interaction induces **zero
pairwise correlation**, so correlation clustering cannot recover interaction
groups. The advantage therefore requires a grouping from domain knowledge or an
*interaction-aware* discovery method — plain correlation is not enough.

## 4. First real-data probe (satimage) — honest negative

Statlog satimage (6,435 x 36, 9.7% positive) has a genuine semantic grouping:
36 features = a 3x3 pixel neighbourhood x 4 spectral bands (pixel-major). Mean
ROC-AUC / PR-AUC over 3 seeds:

| model         | ROC-AUC | PR-AUC |
|:--------------|--------:|-------:|
| ebm           | **0.961** | **0.788** |
| peam_pixel9   | 0.948 | 0.725 |
| peam_band4    | 0.948 | 0.746 |
| peam_corr4    | 0.949 | 0.749 |
| peam_corr9    | 0.941 | 0.718 |

EBM wins: satimage's signal is captured by additive + pairwise structure, so it
is **not** in the localized-high-order regime. Semantic groupings beat
correlation grouping slightly but do not close the gap. This is expected — it is
*why* GA2M/EBM is so effective on typical tabular data — and it sharpens the
honest claim: PEAM is a **targeted tool** for a specific, identifiable regime,
not a general-purpose SOTA. Finding real datasets that live in that regime (and
come with a usable grouping) is the open, ongoing hunt.
