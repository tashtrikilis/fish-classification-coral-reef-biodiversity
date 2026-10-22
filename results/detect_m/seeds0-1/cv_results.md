# Cross-validation: baseline vs copy-paste (detection, pooled over 5 test folds, seeds 0 + 1, averaged)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.372 [0.149–0.567] | 0.411 [0.206–0.601] | +0.040 [-0.071 to +0.181] | 0.471 |
| Forcepsfish | 93 | R | 0.280 [0.093–0.456] | 0.371 [0.198–0.538] | +0.091 [-0.027 to +0.270] | 0.171 |
| Orangespine Unicornfish | 77 | AP50 | 0.184 [0.074–0.299] | 0.256 [0.156–0.358] | +0.073 [+0.012 to +0.152] | 0.023 |
| Orangespine Unicornfish | 77 | R | 0.136 [0.067–0.300] | 0.247 [0.181–0.397] | +0.110 [+0.042 to +0.192] | 0.007 |
| Convict Tang | 33 | AP50 | 0.451 [0.149–0.739] | 0.604 [0.295–0.849] | +0.153 [+0.054 to +0.260] | 0.009 |
| Convict Tang | 33 | R | 0.409 [0.158–0.732] | 0.591 [0.273–0.846] | +0.182 [+0.000 to +0.348] | 0.053 |
| **Focus species (mean)** | 203 | AP50 | 0.335 [0.206–0.459] | 0.424 [0.298–0.547] | +0.089 [+0.041 to +0.144] | 0.001 |
| **Focus species (mean)** | 203 | R | 0.275 [0.159–0.414] | 0.403 [0.278–0.532] | +0.128 [+0.058 to +0.209] | 0.001 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.694 [0.637–0.744] | 0.687 [0.626–0.741] | -0.007 [-0.019 to +0.005] | 0.331 |
| Any fish (class-agnostic) | 4098 | R | 0.625 [0.586–0.665] | 0.606 [0.567–0.647] | -0.019 [-0.035 to -0.002] | 0.022 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.302 [0.275–0.358] | 0.315 [0.289–0.374] | +0.013 [-0.002 to +0.033] | 0.086 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.279 [0.245–0.327] | 0.289 [0.259–0.346] | +0.010 [-0.005 to +0.040] | 0.174 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.372 | 0.411 | +0.040 |
| Forcepsfish | AP50-95 | 0.224 | 0.289 | +0.065 |
| Forcepsfish | P | 0.634 | 0.538 | -0.096 |
| Forcepsfish | R | 0.280 | 0.371 | +0.091 |
| Forcepsfish | F1 | 0.388 | 0.438 | +0.049 |
| Orangespine Unicornfish | AP50 | 0.184 | 0.256 | +0.073 |
| Orangespine Unicornfish | AP50-95 | 0.115 | 0.165 | +0.050 |
| Orangespine Unicornfish | P | 0.319 | 0.481 | +0.162 |
| Orangespine Unicornfish | R | 0.136 | 0.247 | +0.110 |
| Orangespine Unicornfish | F1 | 0.190 | 0.326 | +0.136 |
| Convict Tang | AP50 | 0.451 | 0.604 | +0.153 |
| Convict Tang | AP50-95 | 0.372 | 0.493 | +0.121 |
| Convict Tang | P | 0.627 | 0.600 | -0.027 |
| Convict Tang | R | 0.409 | 0.591 | +0.182 |
| Convict Tang | F1 | 0.495 | 0.595 | +0.100 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.4578 | 0.4638 | 0.25 | 0.2083 |
| 0 | Orangespine Unicornfish | 6 | 0.3611 | 0.3683 | 0.3333 | 0.3333 |
| 0 | Convict Tang | 5 | 0.9717 | 0.9299 | 1.0 | 0.9 |
| 1 | Forcepsfish | 14 | 0.3978 | 0.4101 | 0.3571 | 0.3929 |
| 1 | Orangespine Unicornfish | 15 | 0.1681 | 0.2784 | 0.0667 | 0.2 |
| 1 | Convict Tang | 3 | 0.9579 | 1.0 | 0.8333 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4807 | 0.4689 | 0.1875 | 0.375 |
| 2 | Orangespine Unicornfish | 5 | 0.2942 | 0.3921 | 0.6 | 0.6 |
| 2 | Convict Tang | 9 | 0.6046 | 0.7041 | 0.3889 | 0.7222 |
| 3 | Forcepsfish | 16 | 0.6023 | 0.4406 | 0.6875 | 0.5625 |
| 3 | Orangespine Unicornfish | 21 | 0.1567 | 0.2937 | 0.0714 | 0.2619 |
| 3 | Convict Tang | 7 | 0.0097 | 0.0906 | 0.0 | 0.0 |
| 4 | Forcepsfish | 23 | 0.2112 | 0.5829 | 0.0435 | 0.3913 |
| 4 | Orangespine Unicornfish | 30 | 0.3432 | 0.2765 | 0.1 | 0.1833 |
| 4 | Convict Tang | 9 | 0.3475 | 0.5856 | 0.2778 | 0.6111 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
