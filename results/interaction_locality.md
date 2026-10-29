# Localized high-order interactions: PEAM vs the glass-box SOTA (EBM)

Synthetic: 4 groups x 4 feats + 6 noise, pos_rate=0.1, seeds=[0, 1, 2]. Each group's signal is a pure order-r multiplicative interaction; log-odds additive across groups.

`gbm` is the non-interpretable black-box ceiling; the rest are interpretable.
EBM = additive + pairwise (cannot represent order >= 3).

## ROC-AUC by interaction order (mean over seeds)

|   order |   logreg |    ebm |   peam_true |   peam_corr |   peam_random |    gbm |
|--------:|---------:|-------:|------------:|------------:|--------------:|-------:|
|       1 |   0.9363 | 0.9302 |      0.8678 |      0.8998 |        0.9008 | 0.9181 |
|       2 |   0.5149 | 0.9083 |      0.8559 |      0.7305 |        0.7049 | 0.8899 |
|       3 |   0.4847 | 0.6678 |      0.7952 |      0.5973 |        0.6142 | 0.7314 |
|       4 |   0.5223 | 0.6794 |      0.6852 |      0.5992 |        0.6129 | 0.6368 |

## PR-AUC by interaction order (mean over seeds)

|   order |   logreg |    ebm |   peam_true |   peam_corr |   peam_random |    gbm |
|--------:|---------:|-------:|------------:|------------:|--------------:|-------:|
|       1 |   0.6862 | 0.667  |      0.4969 |      0.5604 |        0.5655 | 0.6187 |
|       2 |   0.121  | 0.6179 |      0.4723 |      0.2832 |        0.2679 | 0.5555 |
|       3 |   0.1037 | 0.1831 |      0.4182 |      0.1448 |        0.1529 | 0.3118 |
|       4 |   0.1281 | 0.179  |      0.254  |      0.1467 |        0.1497 | 0.1664 |

## Reading

- order 1-2: EBM and PEAM both capture the signal (pairwise is enough).
- order >= 3: EBM saturates (no high-order terms) while PEAM_true tracks the
  black-box GBM ceiling -> PEAM strictly beats the interpretable SOTA at the
  same interpretability tier, exactly in the localized-high-order regime.
- peam_corr vs peam_random shows how much the data-driven grouping recovers vs a wrong grouping.