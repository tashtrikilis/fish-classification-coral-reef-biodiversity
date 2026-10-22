# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26m-seg, box AP; seeds 0 + 1, averaged)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.404 [0.212–0.564] | 0.512 [0.311–0.681] | +0.108 [+0.010 to +0.222] | 0.029 |
| Forcepsfish | 93 | R | 0.296 [0.151–0.424] | 0.435 [0.263–0.601] | +0.140 [+0.030 to +0.281] | 0.012 |
| Orangespine Unicornfish | 77 | AP50 | 0.144 [0.044–0.243] | 0.232 [0.112–0.346] | +0.088 [+0.011 to +0.206] | 0.013 |
| Orangespine Unicornfish | 77 | R | 0.149 [0.040–0.231] | 0.253 [0.167–0.395] | +0.104 [+0.009 to +0.255] | 0.027 |
| Convict Tang | 33 | AP50 | 0.361 [0.087–0.616] | 0.610 [0.341–0.830] | +0.249 [+0.149 to +0.390] | 0.000 |
| Convict Tang | 33 | R | 0.318 [0.091–0.556] | 0.636 [0.404–0.885] | +0.318 [+0.241 to +0.467] | 0.000 |
| **Focus species (mean)** | 203 | AP50 | 0.303 [0.187–0.408] | 0.451 [0.336–0.559] | +0.148 [+0.095 to +0.218] | 0.000 |
| **Focus species (mean)** | 203 | R | 0.254 [0.150–0.350] | 0.442 [0.335–0.561] | +0.187 [+0.141 to +0.265] | 0.000 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.683 [0.624–0.737] | 0.693 [0.637–0.742] | +0.010 [-0.007 to +0.026] | 0.291 |
| Any fish (class-agnostic) | 4098 | R | 0.611 [0.567–0.656] | 0.620 [0.575–0.666] | +0.009 [-0.007 to +0.026] | 0.273 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.279 [0.253–0.326] | 0.321 [0.299–0.371] | +0.043 [+0.026 to +0.067] | 0.000 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.268 [0.241–0.307] | 0.301 [0.276–0.346] | +0.033 [+0.019 to +0.053] | 0.000 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.404 | 0.512 | +0.108 |
| Forcepsfish | AP50-95 | 0.255 | 0.358 | +0.103 |
| Forcepsfish | P | 0.647 | 0.690 | +0.043 |
| Forcepsfish | R | 0.296 | 0.435 | +0.140 |
| Forcepsfish | F1 | 0.406 | 0.534 | +0.128 |
| Orangespine Unicornfish | AP50 | 0.144 | 0.232 | +0.088 |
| Orangespine Unicornfish | AP50-95 | 0.086 | 0.139 | +0.053 |
| Orangespine Unicornfish | P | 0.306 | 0.392 | +0.086 |
| Orangespine Unicornfish | R | 0.149 | 0.253 | +0.104 |
| Orangespine Unicornfish | F1 | 0.200 | 0.306 | +0.106 |
| Convict Tang | AP50 | 0.361 | 0.610 | +0.249 |
| Convict Tang | AP50-95 | 0.307 | 0.514 | +0.207 |
| Convict Tang | P | 0.418 | 0.510 | +0.092 |
| Convict Tang | R | 0.318 | 0.636 | +0.318 |
| Convict Tang | F1 | 0.361 | 0.565 | +0.204 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.6011 | 0.5565 | 0.3125 | 0.2917 |
| 0 | Orangespine Unicornfish | 6 | 0.3371 | 0.4239 | 0.25 | 0.4167 |
| 0 | Convict Tang | 5 | 0.8261 | 0.9717 | 0.8 | 1.0 |
| 1 | Forcepsfish | 14 | 0.3774 | 0.4322 | 0.3571 | 0.3929 |
| 1 | Orangespine Unicornfish | 15 | 0.1028 | 0.116 | 0.0 | 0.1 |
| 1 | Convict Tang | 3 | 0.5827 | 0.9171 | 0.5 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4338 | 0.4735 | 0.25 | 0.375 |
| 2 | Orangespine Unicornfish | 5 | 0.1417 | 0.4288 | 0.2 | 0.7 |
| 2 | Convict Tang | 9 | 0.5427 | 0.7374 | 0.3889 | 0.6667 |
| 3 | Forcepsfish | 16 | 0.3668 | 0.5372 | 0.4375 | 0.5312 |
| 3 | Orangespine Unicornfish | 21 | 0.1142 | 0.3122 | 0.0714 | 0.2619 |
| 3 | Convict Tang | 7 | 0.0417 | 0.3717 | 0.0 | 0.2143 |
| 4 | Forcepsfish | 23 | 0.3782 | 0.6689 | 0.1739 | 0.587 |
| 4 | Orangespine Unicornfish | 30 | 0.2694 | 0.3151 | 0.25 | 0.2167 |
| 4 | Convict Tang | 9 | 0.2574 | 0.6027 | 0.1667 | 0.6111 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
