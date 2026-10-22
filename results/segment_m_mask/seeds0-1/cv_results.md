# Cross-validation: baseline vs copy-paste (mask AP, pooled over 5 test folds, yolo26m-seg, mask AP; seeds 0 + 1, averaged)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.418 [0.223–0.577] | 0.529 [0.318–0.698] | +0.110 [+0.020 to +0.220] | 0.014 |
| Forcepsfish | 93 | R | 0.296 [0.151–0.424] | 0.441 [0.267–0.604] | +0.145 [+0.042 to +0.286] | 0.004 |
| Orangespine Unicornfish | 77 | AP50 | 0.147 [0.044–0.248] | 0.234 [0.112–0.348] | +0.088 [+0.011 to +0.206] | 0.013 |
| Orangespine Unicornfish | 77 | R | 0.149 [0.040–0.231] | 0.253 [0.167–0.395] | +0.104 [+0.009 to +0.255] | 0.027 |
| Convict Tang | 33 | AP50 | 0.360 [0.087–0.613] | 0.610 [0.341–0.830] | +0.250 [+0.150 to +0.390] | 0.000 |
| Convict Tang | 33 | R | 0.318 [0.091–0.556] | 0.636 [0.404–0.885] | +0.318 [+0.241 to +0.467] | 0.000 |
| **Focus species (mean)** | 203 | AP50 | 0.308 [0.192–0.413] | 0.458 [0.341–0.566] | +0.149 [+0.099 to +0.217] | 0.000 |
| **Focus species (mean)** | 203 | R | 0.254 [0.150–0.350] | 0.443 [0.337–0.562] | +0.189 [+0.145 to +0.265] | 0.000 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.689 [0.629–0.741] | 0.698 [0.643–0.748] | +0.009 [-0.006 to +0.027] | 0.246 |
| Any fish (class-agnostic) | 4098 | R | 0.613 [0.570–0.658] | 0.623 [0.579–0.669] | +0.010 [-0.007 to +0.027] | 0.231 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.280 [0.254–0.328] | 0.325 [0.303–0.375] | +0.045 [+0.029 to +0.068] | 0.000 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.268 [0.241–0.307] | 0.302 [0.277–0.347] | +0.034 [+0.021 to +0.054] | 0.000 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.418 | 0.529 | +0.110 |
| Forcepsfish | AP50-95 | 0.259 | 0.361 | +0.102 |
| Forcepsfish | P | 0.647 | 0.698 | +0.051 |
| Forcepsfish | R | 0.296 | 0.441 | +0.145 |
| Forcepsfish | F1 | 0.406 | 0.540 | +0.134 |
| Orangespine Unicornfish | AP50 | 0.147 | 0.234 | +0.088 |
| Orangespine Unicornfish | AP50-95 | 0.105 | 0.166 | +0.060 |
| Orangespine Unicornfish | P | 0.306 | 0.392 | +0.086 |
| Orangespine Unicornfish | R | 0.149 | 0.253 | +0.104 |
| Orangespine Unicornfish | F1 | 0.200 | 0.306 | +0.106 |
| Convict Tang | AP50 | 0.360 | 0.610 | +0.250 |
| Convict Tang | AP50-95 | 0.305 | 0.493 | +0.188 |
| Convict Tang | P | 0.418 | 0.510 | +0.092 |
| Convict Tang | R | 0.318 | 0.636 | +0.318 |
| Convict Tang | F1 | 0.361 | 0.565 | +0.204 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.6311 | 0.6035 | 0.3125 | 0.3125 |
| 0 | Orangespine Unicornfish | 6 | 0.3371 | 0.4239 | 0.25 | 0.4167 |
| 0 | Convict Tang | 5 | 0.8261 | 0.9717 | 0.8 | 1.0 |
| 1 | Forcepsfish | 14 | 0.3774 | 0.4322 | 0.3571 | 0.3929 |
| 1 | Orangespine Unicornfish | 15 | 0.1028 | 0.116 | 0.0 | 0.1 |
| 1 | Convict Tang | 3 | 0.5827 | 0.9171 | 0.5 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4338 | 0.4735 | 0.25 | 0.375 |
| 2 | Orangespine Unicornfish | 5 | 0.1417 | 0.4288 | 0.2 | 0.7 |
| 2 | Convict Tang | 9 | 0.5382 | 0.7374 | 0.3889 | 0.6667 |
| 3 | Forcepsfish | 16 | 0.394 | 0.5643 | 0.4375 | 0.5312 |
| 3 | Orangespine Unicornfish | 21 | 0.1142 | 0.3122 | 0.0714 | 0.2619 |
| 3 | Convict Tang | 7 | 0.0417 | 0.3717 | 0.0 | 0.2143 |
| 4 | Forcepsfish | 23 | 0.3853 | 0.6789 | 0.1739 | 0.587 |
| 4 | Orangespine Unicornfish | 30 | 0.2772 | 0.3233 | 0.25 | 0.2167 |
| 4 | Convict Tang | 9 | 0.2574 | 0.6027 | 0.1667 | 0.6111 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
