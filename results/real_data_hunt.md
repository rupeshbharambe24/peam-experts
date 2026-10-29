# Real-data hunt: where does the localized-high-order regime occur?

Seeds: [0, 1, 2]. ROC-AUC, mean over seeds. `peam_*` are interpretable; `xgb_*` is the black-box reference.

## satimage

| model | ROC-AUC |
|:--|--:|
| ebm | 0.9609 |
| peam_semantic | 0.9484 |
| peam_corr | 0.9411 |

Engineered remote-sensing features (9 pixels x 4 bands). Low-order: **EBM wins**, semantic grouping ~ correlation grouping. Not in the regime.

## gas-drift

| model | ROC-AUC |
|:--|--:|
| logreg | 0.9989 |
| ebm | 0.9997 |
| peam_semantic | 0.9994 |

16 sensors x 8 descriptors. Near-separable: even logistic regression is ~0.99. Saturated -> no interaction headroom.

## higgs

| model | ROC-AUC |
|:--|--:|
| ebm_lowlevel | 0.6743 |
| peam_perobject | 0.6077 |
| xgb_lowlevel | 0.6802 |
| ebm_with_masses | 0.7959 |

Low-level = per-object kinematics; `ebm_with_masses` adds the 7 cross-object invariant masses. The high-order signal is **cross-object (delocalized)**: per-object PEAM loses to EBM, and masses lift EBM far above low-level. The contrast case -- high-order but not localized.

## madelon

| model | ROC-AUC |
|:--|--:|
| ebm_full | 0.7998 |
| xgb_full | 0.8565 |
| peam_corr_full | 0.9025 |
| ebm_fs | 0.8850 |
| xgb_fs | 0.9273 |
| peam_fs | 0.9115 |

5 informative features (a localized high-order interaction) + 15 correlated redundant + 480 noise. **PEAM is the best interpretable model**: beats EBM on full features and under shared feature selection, and beats full-feature black boxes; only a black-box + feature-selection pipeline (`xgb_fs`) edges it, without interpretability.
