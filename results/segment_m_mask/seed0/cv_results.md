# Cross-validation: baseline vs copy-paste (mask AP, pooled over 5 test folds, yolo26m-seg, mask AP; seed 0)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference. Recall is at confidence 0.25. Primary endpoint: 'Focus species (mean)' AP50.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.387 [0.167–0.576] | 0.557 [0.352–0.720] | +0.171 [+0.046 to +0.349] | 0.002 |
| Forcepsfish | 93 | R | 0.290 [0.120–0.462] | 0.462 [0.300–0.614] | +0.172 [+0.021 to +0.362] | 0.031 |
| Orangespine Unicornfish | 77 | AP50 | 0.159 [0.045–0.247] | 0.242 [0.097–0.350] | +0.083 [+0.003 to +0.210] | 0.041 |
| Orangespine Unicornfish | 77 | R | 0.130 [0.032–0.186] | 0.273 [0.174–0.394] | +0.143 [+0.056 to +0.295] | 0.001 |
| Convict Tang | 33 | AP50 | 0.349 [0.073–0.586] | 0.524 [0.219–0.826] | +0.175 [+0.023 to +0.401] | 0.034 |
| Convict Tang | 33 | R | 0.273 [0.057–0.500] | 0.545 [0.250–0.875] | +0.273 [+0.109 to +0.549] | 0.006 |
| **Focus species (mean)** | 203 | AP50 | 0.298 [0.176–0.405] | 0.441 [0.320–0.565] | +0.143 [+0.070 to +0.251] | 0.000 |
| **Focus species (mean)** | 203 | R | 0.231 [0.127–0.328] | 0.427 [0.312–0.555] | +0.196 [+0.125 to +0.321] | 0.000 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.691 [0.631–0.745] | 0.694 [0.643–0.743] | +0.002 [-0.016 to +0.027] | 0.741 |
| Any fish (class-agnostic) | 4098 | R | 0.602 [0.554–0.650] | 0.619 [0.572–0.666] | +0.018 [-0.003 to +0.037] | 0.094 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.273 [0.245–0.321] | 0.313 [0.292–0.366] | +0.040 [+0.016 to +0.078] | 0.001 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.253 [0.224–0.290] | 0.293 [0.268–0.339] | +0.039 [+0.018 to +0.072] | 0.000 |

A difference counts as a finding only if its 95% interval excludes 0. The focus-species mean is the single pre-specified test; treat the per-species rows as supporting detail.

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.387 | 0.557 | +0.171 |
| Forcepsfish | AP50-95 | 0.228 | 0.375 | +0.147 |
| Forcepsfish | P | 0.628 | 0.768 | +0.140 |
| Forcepsfish | R | 0.290 | 0.462 | +0.172 |
| Forcepsfish | F1 | 0.397 | 0.577 | +0.180 |
| Orangespine Unicornfish | AP50 | 0.159 | 0.242 | +0.083 |
| Orangespine Unicornfish | AP50-95 | 0.116 | 0.166 | +0.050 |
| Orangespine Unicornfish | P | 0.303 | 0.356 | +0.053 |
| Orangespine Unicornfish | R | 0.130 | 0.273 | +0.143 |
| Orangespine Unicornfish | F1 | 0.182 | 0.309 | +0.127 |
| Convict Tang | AP50 | 0.349 | 0.524 | +0.175 |
| Convict Tang | AP50-95 | 0.295 | 0.430 | +0.135 |
| Convict Tang | P | 0.391 | 0.486 | +0.095 |
| Convict Tang | R | 0.273 | 0.545 | +0.273 |
| Convict Tang | F1 | 0.321 | 0.514 | +0.193 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.6255 | 0.6146 | 0.2917 | 0.375 |
| 0 | Orangespine Unicornfish | 6 | 0.1058 | 0.3651 | 0.0 | 0.3333 |
| 0 | Convict Tang | 5 | 0.8074 | 1.0 | 0.8 | 1.0 |
| 1 | Forcepsfish | 14 | 0.4103 | 0.4349 | 0.3571 | 0.3571 |
| 1 | Orangespine Unicornfish | 15 | 0.0878 | 0.0591 | 0.0 | 0.0667 |
| 1 | Convict Tang | 3 | 0.3 | 1.0 | 0.0 | 1.0 |
| 2 | Forcepsfish | 16 | 0.3788 | 0.4032 | 0.3125 | 0.375 |
| 2 | Orangespine Unicornfish | 5 | 0.2254 | 0.4583 | 0.2 | 0.8 |
| 2 | Convict Tang | 9 | 0.5657 | 0.6621 | 0.3333 | 0.5556 |
| 3 | Forcepsfish | 16 | 0.4953 | 0.5922 | 0.5625 | 0.5 |
| 3 | Orangespine Unicornfish | 21 | 0.1861 | 0.3339 | 0.1429 | 0.3333 |
| 3 | Convict Tang | 7 | 0.0349 | 0.2327 | 0.0 | 0.0 |
| 4 | Forcepsfish | 23 | 0.2055 | 0.7384 | 0.0435 | 0.6522 |
| 4 | Orangespine Unicornfish | 30 | 0.258 | 0.342 | 0.2 | 0.2333 |
| 4 | Convict Tang | 9 | 0.1787 | 0.5908 | 0.2222 | 0.5556 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
