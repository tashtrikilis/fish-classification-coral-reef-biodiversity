# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26s detection, box AP; seed 1)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.278 [0.096–0.453] | 0.363 [0.164–0.559] | +0.085 [+0.014 to +0.184] | 0.018 |
| Forcepsfish | 93 | R | 0.247 [0.105–0.382] | 0.344 [0.164–0.521] | +0.097 [+0.029 to +0.196] | 0.004 |
| Orangespine Unicornfish | 77 | AP50 | 0.076 [0.030–0.155] | 0.154 [0.058–0.243] | +0.078 [-0.001 to +0.136] | 0.054 |
| Orangespine Unicornfish | 77 | R | 0.052 [0.000–0.116] | 0.117 [0.043–0.169] | +0.065 [-0.028 to +0.109] | 0.158 |
| Convict Tang | 33 | AP50 | 0.360 [0.139–0.578] | 0.464 [0.149–0.692] | +0.104 [-0.083 to +0.275] | 0.274 |
| Convict Tang | 33 | R | 0.333 [0.143–0.571] | 0.485 [0.192–0.769] | +0.152 [+0.000 to +0.276] | 0.092 |
| **Focus species (mean)** | 203 | AP50 | 0.238 [0.142–0.341] | 0.327 [0.199–0.443] | +0.089 [+0.013 to +0.164] | 0.031 |
| **Focus species (mean)** | 203 | R | 0.211 [0.123–0.307] | 0.315 [0.186–0.437] | +0.104 [+0.032 to +0.165] | 0.010 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.674 [0.616–0.723] | 0.663 [0.596–0.722] | -0.010 [-0.028 to +0.008] | 0.306 |
| Any fish (class-agnostic) | 4098 | R | 0.575 [0.535–0.617] | 0.569 [0.525–0.614] | -0.006 [-0.030 to +0.018] | 0.616 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.252 [0.229–0.295] | 0.266 [0.233–0.339] | +0.014 [-0.013 to +0.069] | 0.260 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.243 [0.220–0.282] | 0.253 [0.219–0.314] | +0.010 [-0.033 to +0.061] | 0.492 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.278 | 0.363 | +0.085 |
| Forcepsfish | AP50-95 | 0.158 | 0.231 | +0.073 |
| Forcepsfish | P | 0.575 | 0.552 | -0.023 |
| Forcepsfish | R | 0.247 | 0.344 | +0.097 |
| Forcepsfish | F1 | 0.346 | 0.424 | +0.078 |
| Orangespine Unicornfish | AP50 | 0.076 | 0.154 | +0.078 |
| Orangespine Unicornfish | AP50-95 | 0.049 | 0.094 | +0.045 |
| Orangespine Unicornfish | P | 0.235 | 0.333 | +0.098 |
| Orangespine Unicornfish | R | 0.052 | 0.117 | +0.065 |
| Orangespine Unicornfish | F1 | 0.085 | 0.173 | +0.088 |
| Convict Tang | AP50 | 0.360 | 0.464 | +0.104 |
| Convict Tang | AP50-95 | 0.294 | 0.384 | +0.090 |
| Convict Tang | P | 0.611 | 0.410 | -0.201 |
| Convict Tang | R | 0.333 | 0.485 | +0.152 |
| Convict Tang | F1 | 0.431 | 0.444 | +0.013 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.3406 | 0.3839 | 0.1667 | 0.25 |
| 0 | Orangespine Unicornfish | 6 | 0.2435 | 0.3552 | 0.1667 | 0.1667 |
| 0 | Convict Tang | 5 | 0.6953 | 0.7989 | 0.6 | 0.8 |
| 1 | Forcepsfish | 14 | 0.348 | 0.3247 | 0.2857 | 0.2857 |
| 1 | Orangespine Unicornfish | 15 | 0.134 | 0.1093 | 0.0667 | 0.0667 |
| 1 | Convict Tang | 3 | 0.5341 | 0.9158 | 0.6667 | 1.0 |
| 2 | Forcepsfish | 16 | 0.296 | 0.4797 | 0.1875 | 0.4375 |
| 2 | Orangespine Unicornfish | 5 | 0.0557 | 0.1559 | 0.0 | 0.2 |
| 2 | Convict Tang | 9 | 0.4144 | 0.6106 | 0.3333 | 0.5556 |
| 3 | Forcepsfish | 16 | 0.4533 | 0.4576 | 0.5 | 0.5625 |
| 3 | Orangespine Unicornfish | 21 | 0.06 | 0.0939 | 0.0 | 0.0476 |
| 3 | Convict Tang | 7 | 0.0691 | 0.0331 | 0.0 | 0.0 |
| 4 | Forcepsfish | 23 | 0.2157 | 0.4658 | 0.1739 | 0.2609 |
| 4 | Orangespine Unicornfish | 30 | 0.1874 | 0.2967 | 0.0667 | 0.1667 |
| 4 | Convict Tang | 9 | 0.5149 | 0.44 | 0.3333 | 0.4444 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
