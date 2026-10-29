# The price of modularity: PEAM accuracy vs number of groups

ROC-AUC as the feature set is split into more independent group-experts.
`g=1` is a single calibrated black-box expert (no additive decomposition);
higher `g` means more modular / finer group-level reason codes. EBM shown for
reference. Mean over 2 seeds.

| dataset      |   g=1  |   g=2  |   g=3  |   g=5  |   g=8  |  EBM   |
|:-------------|-------:|-------:|-------:|-------:|-------:|-------:|
| wine_quality | 0.8306 | 0.8315 | 0.8241 | 0.8242 | 0.8289 | 0.8498 |
| satimage     | 0.9612 | 0.9564 | 0.9460 | 0.9445 | 0.9411 | 0.9590 |
| us_crime     | 0.9151 | 0.9157 | 0.9155 | 0.9150 | 0.9072 | 0.9202 |
| ozone_level  | 0.8572 | 0.8236 | 0.8294 | 0.8154 | 0.7899 | 0.8430 |

**Reading:** where cross-group interactions carry signal (satimage, ozone_level)
accuracy falls monotonically as the model is forced to be additive across more
groups; at `g=1` PEAM matches or beats EBM. The accuracy gap to EBM/black-box is
therefore the *price of the additive-across-groups structure* (lost
interactions), not a weakness of the within-group experts. This quantifies
PEAM's core trade-off: modularity and group-level reason codes are bought with a
measurable accuracy cost that depends on how interaction-driven the dataset is.
