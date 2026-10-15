# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26s-seg, box AP; seed 1)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference. Recall is at confidence 0.25. Primary endpoint: 'Focus species (mean)' AP50.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.264 [0.110–0.413] | 0.437 [0.236–0.601] | +0.173 [+0.032 to +0.294] | 0.018 |
| Forcepsfish | 93 | R | 0.215 [0.105–0.319] | 0.376 [0.222–0.514] | +0.161 [+0.055 to +0.289] | 0.004 |
| Orangespine Unicornfish | 77 | AP50 | 0.108 [0.017–0.241] | 0.106 [0.040–0.267] | -0.002 [-0.116 to +0.171] | 0.987 |
| Orangespine Unicornfish | 77 | R | 0.104 [0.017–0.180] | 0.091 [0.026–0.280] | -0.013 [-0.113 to +0.158] | 0.970 |
| Convict Tang | 33 | AP50 | 0.408 [0.107–0.681] | 0.623 [0.463–0.774] | +0.215 [+0.011 to +0.443] | 0.038 |
| Convict Tang | 33 | R | 0.273 [0.071–0.556] | 0.727 [0.615–0.895] | +0.455 [+0.333 to +0.560] | 0.000 |
| **Focus species (mean)** | 203 | AP50 | 0.260 [0.131–0.371] | 0.389 [0.300–0.485] | +0.129 [+0.052 to +0.226] | 0.001 |
| **Focus species (mean)** | 203 | R | 0.197 [0.103–0.307] | 0.398 [0.328–0.497] | +0.201 [+0.140 to +0.279] | 0.000 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.672 [0.619–0.723] | 0.663 [0.606–0.717] | -0.010 [-0.028 to +0.007] | 0.292 |
| Any fish (class-agnostic) | 4098 | R | 0.589 [0.549–0.629] | 0.591 [0.549–0.634] | +0.003 [-0.019 to +0.022] | 0.793 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.261 [0.231–0.304] | 0.286 [0.260–0.333] | +0.024 [+0.008 to +0.049] | 0.008 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.238 [0.205–0.272] | 0.281 [0.250–0.316] | +0.043 [+0.025 to +0.067] | 0.000 |

A difference counts as a finding only if its 95% interval excludes 0. The focus-species mean is the single pre-specified test; treat the per-species rows as supporting detail.

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.264 | 0.437 | +0.173 |
| Forcepsfish | AP50-95 | 0.130 | 0.302 | +0.172 |
| Forcepsfish | P | 0.455 | 0.530 | +0.076 |
| Forcepsfish | R | 0.215 | 0.376 | +0.161 |
| Forcepsfish | F1 | 0.292 | 0.440 | +0.148 |
| Orangespine Unicornfish | AP50 | 0.108 | 0.106 | -0.002 |
| Orangespine Unicornfish | AP50-95 | 0.057 | 0.063 | +0.006 |
| Orangespine Unicornfish | P | 0.296 | 0.304 | +0.008 |
| Orangespine Unicornfish | R | 0.104 | 0.091 | -0.013 |
| Orangespine Unicornfish | F1 | 0.154 | 0.140 | -0.014 |
| Convict Tang | AP50 | 0.408 | 0.623 | +0.215 |
| Convict Tang | AP50-95 | 0.339 | 0.491 | +0.152 |
| Convict Tang | P | 0.692 | 0.471 | -0.222 |
| Convict Tang | R | 0.273 | 0.727 | +0.455 |
| Convict Tang | F1 | 0.391 | 0.571 | +0.180 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.4722 | 0.3814 | 0.25 | 0.2917 |
| 0 | Orangespine Unicornfish | 6 | 0.2909 | 0.3993 | 0.1667 | 0.3333 |
| 0 | Convict Tang | 5 | 0.8727 | 0.8515 | 0.8 | 1.0 |
| 1 | Forcepsfish | 14 | 0.2311 | 0.3908 | 0.1429 | 0.3571 |
| 1 | Orangespine Unicornfish | 15 | 0.0388 | 0.1384 | 0.0667 | 0.1333 |
| 1 | Convict Tang | 3 | 0.8076 | 0.8076 | 0.6667 | 1.0 |
| 2 | Forcepsfish | 16 | 0.2963 | 0.4178 | 0.1875 | 0.25 |
| 2 | Orangespine Unicornfish | 5 | 0.1364 | 0.419 | 0.2 | 0.4 |
| 2 | Convict Tang | 9 | 0.527 | 0.5731 | 0.2222 | 0.6667 |
| 3 | Forcepsfish | 16 | 0.3695 | 0.5189 | 0.4375 | 0.5 |
| 3 | Orangespine Unicornfish | 21 | 0.019 | 0.0242 | 0.0 | 0.0476 |
| 3 | Convict Tang | 7 | 0.0334 | 0.4655 | 0.0 | 0.5714 |
| 4 | Forcepsfish | 23 | 0.1661 | 0.5928 | 0.087 | 0.4783 |
| 4 | Orangespine Unicornfish | 30 | 0.3252 | 0.1066 | 0.1667 | 0.0 |
| 4 | Convict Tang | 9 | 0.3094 | 0.703 | 0.1111 | 0.6667 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
