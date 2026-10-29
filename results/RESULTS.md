# PEAM proof-of-signal results

Datasets: 8 | seeds: [np.int64(0), np.int64(1), np.int64(2)] | EBM available: True

## Mean metric across datasets x seeds

| method      |   roc_auc |   pr_auc |   brier |    ece |
|:------------|----------:|---------:|--------:|-------:|
| logreg      |    0.8671 |   0.4147 |  0.1072 | 0.178  |
| ebm         |    0.9089 |   0.5488 |  0.0231 | 0.0142 |
| gbm         |    0.9    |   0.5273 |  0.026  | 0.0258 |
| ensemble    |    0.9086 |   0.5501 |  0.0234 | 0.0175 |
| peam        |    0.8876 |   0.4933 |  0.0251 | 0.0144 |
| peam_naive  |    0.8833 |   0.4861 |  0.0313 | 0.0252 |
| peam_random |    0.8834 |   0.4841 |  0.0249 | 0.0151 |

## Average rank (1 = best)

| method      |   roc_auc |   pr_auc |   brier |   ece |
|:------------|----------:|---------:|--------:|------:|
| logreg      |      4.92 |     5.25 |    7    |  7    |
| ebm         |      2.62 |     2.21 |    1.96 |  2.42 |
| gbm         |      3.46 |     3.29 |    4.33 |  5.25 |
| ensemble    |      2.83 |     2.79 |    2.75 |  3.79 |
| peam        |      4.27 |     4.44 |    3.27 |  2.35 |
| peam_naive  |      5.31 |     5.21 |    5.08 |  4.58 |
| peam_random |      4.58 |     4.81 |    3.6  |  2.6  |

## Reason-code fidelity (AUC drop when top group's evidence zeroed; higher = codes identify real drivers)

mean AUC drop = 0.1427 (std 0.1872)

## Per-dataset ROC-AUC (mean over seeds)

| dataset      |   logreg |    ebm |    gbm |   ensemble |   peam |   peam_naive |   peam_random |
|:-------------|---------:|-------:|-------:|-----------:|-------:|-------------:|--------------:|
| abalone_19   |   0.7978 | 0.7644 | 0.7436 |     0.7131 | 0.7163 |       0.7079 |        0.6676 |
| mammography  |   0.9109 | 0.9537 | 0.9294 |     0.9307 | 0.9315 |       0.9272 |        0.9261 |
| oil          |   0.9167 | 0.937  | 0.8928 |     0.9579 | 0.9227 |       0.9181 |        0.8951 |
| ozone_level  |   0.8521 | 0.8709 | 0.8934 |     0.8764 | 0.8487 |       0.8438 |        0.8853 |
| protein_homo |   0.9936 | 0.9951 | 0.9933 |     0.9936 | 0.9947 |       0.9917 |        0.9927 |
| satimage     |   0.7639 | 0.9609 | 0.9661 |     0.9676 | 0.9468 |       0.9472 |        0.9581 |
| us_crime     |   0.9044 | 0.9257 | 0.913  |     0.9238 | 0.9034 |       0.8959 |        0.9246 |
| wine_quality |   0.7971 | 0.8635 | 0.8687 |     0.9056 | 0.8371 |       0.8348 |        0.818  |