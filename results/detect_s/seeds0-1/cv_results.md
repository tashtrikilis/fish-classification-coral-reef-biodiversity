# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26s detection, box AP; seeds 0 + 1, averaged)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference. Recall is at confidence 0.25. Primary endpoint: 'Focus species (mean)' AP50.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.302 [0.118–0.473] | 0.375 [0.183–0.548] | +0.073 [-0.004 to +0.158] | 0.061 |
| Forcepsfish | 93 | R | 0.253 [0.104–0.396] | 0.333 [0.167–0.500] | +0.081 [+0.022 to +0.173] | 0.007 |
| Orangespine Unicornfish | 77 | AP50 | 0.100 [0.031–0.216] | 0.143 [0.060–0.241] | +0.043 [-0.028 to +0.087] | 0.168 |
| Orangespine Unicornfish | 77 | R | 0.097 [0.030–0.250] | 0.123 [0.054–0.229] | +0.026 [-0.057 to +0.069] | 0.413 |
| Convict Tang | 33 | AP50 | 0.400 [0.143–0.644] | 0.554 [0.340–0.730] | +0.154 [+0.011 to +0.288] | 0.037 |
| Convict Tang | 33 | R | 0.364 [0.154–0.600] | 0.561 [0.379–0.828] | +0.197 [+0.071 to +0.357] | 0.011 |
| **Focus species (mean)** | 203 | AP50 | 0.267 [0.155–0.385] | 0.357 [0.253–0.461] | +0.090 [+0.026 to +0.150] | 0.013 |
| **Focus species (mean)** | 203 | R | 0.238 [0.140–0.355] | 0.339 [0.243–0.458] | +0.101 [+0.037 to +0.167] | 0.004 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.670 [0.613–0.721] | 0.667 [0.601–0.725] | -0.003 [-0.016 to +0.012] | 0.814 |
| Any fish (class-agnostic) | 4098 | R | 0.575 [0.536–0.617] | 0.576 [0.536–0.618] | +0.001 [-0.012 to +0.014] | 0.956 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.262 [0.241–0.308] | 0.272 [0.244–0.328] | +0.009 [-0.015 to +0.042] | 0.468 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.248 [0.223–0.289] | 0.251 [0.225–0.293] | +0.003 [-0.028 to +0.032] | 0.615 |

A difference counts as a finding only if its 95% interval excludes 0. The focus-species mean is the single pre-specified test; treat the per-species rows as supporting detail.

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.302 | 0.375 | +0.073 |
| Forcepsfish | AP50-95 | 0.170 | 0.238 | +0.069 |
| Forcepsfish | P | 0.567 | 0.522 | -0.045 |
| Forcepsfish | R | 0.253 | 0.333 | +0.081 |
| Forcepsfish | F1 | 0.349 | 0.407 | +0.057 |
| Orangespine Unicornfish | AP50 | 0.100 | 0.143 | +0.043 |
| Orangespine Unicornfish | AP50-95 | 0.061 | 0.087 | +0.026 |
| Orangespine Unicornfish | P | 0.279 | 0.333 | +0.054 |
| Orangespine Unicornfish | R | 0.097 | 0.123 | +0.026 |
| Orangespine Unicornfish | F1 | 0.142 | 0.180 | +0.038 |
| Convict Tang | AP50 | 0.400 | 0.554 | +0.154 |
| Convict Tang | AP50-95 | 0.334 | 0.454 | +0.120 |
| Convict Tang | P | 0.576 | 0.497 | -0.080 |
| Convict Tang | R | 0.364 | 0.561 | +0.197 |
| Convict Tang | F1 | 0.444 | 0.527 | +0.083 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.3705 | 0.4242 | 0.1458 | 0.2083 |
| 0 | Orangespine Unicornfish | 6 | 0.4129 | 0.3374 | 0.4167 | 0.25 |
| 0 | Convict Tang | 5 | 0.7709 | 0.8995 | 0.8 | 0.9 |
| 1 | Forcepsfish | 14 | 0.3141 | 0.2944 | 0.25 | 0.25 |
| 1 | Orangespine Unicornfish | 15 | 0.1176 | 0.1131 | 0.0667 | 0.1 |
| 1 | Convict Tang | 3 | 0.6012 | 0.7584 | 0.5 | 1.0 |
| 2 | Forcepsfish | 16 | 0.344 | 0.4611 | 0.2188 | 0.4062 |
| 2 | Orangespine Unicornfish | 5 | 0.1294 | 0.1922 | 0.2 | 0.3 |
| 2 | Convict Tang | 9 | 0.4987 | 0.5875 | 0.3889 | 0.4444 |
| 3 | Forcepsfish | 16 | 0.4712 | 0.496 | 0.5312 | 0.5625 |
| 3 | Orangespine Unicornfish | 21 | 0.0424 | 0.0815 | 0.0238 | 0.0476 |
| 3 | Convict Tang | 7 | 0.0352 | 0.4192 | 0.0 | 0.2857 |
| 4 | Forcepsfish | 23 | 0.2576 | 0.4651 | 0.1957 | 0.3043 |
| 4 | Orangespine Unicornfish | 30 | 0.2521 | 0.2923 | 0.0833 | 0.1333 |
| 4 | Convict Tang | 9 | 0.5129 | 0.5413 | 0.3333 | 0.5556 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
