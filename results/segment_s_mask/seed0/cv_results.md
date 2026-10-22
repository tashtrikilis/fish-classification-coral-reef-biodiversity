# Cross-validation: baseline vs copy-paste (mask AP, pooled over 5 test folds, yolo26s-seg, mask AP; seed 0)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.283 [0.106–0.472] | 0.362 [0.202–0.511] | +0.079 [-0.027 to +0.175] | 0.144 |
| Forcepsfish | 93 | R | 0.215 [0.072–0.328] | 0.323 [0.203–0.430] | +0.108 [+0.050 to +0.190] | 0.000 |
| Orangespine Unicornfish | 77 | AP50 | 0.093 [0.015–0.214] | 0.094 [0.025–0.189] | +0.000 [-0.090 to +0.060] | 0.941 |
| Orangespine Unicornfish | 77 | R | 0.091 [0.000–0.213] | 0.078 [0.018–0.154] | -0.013 [-0.081 to +0.042] | 0.799 |
| Convict Tang | 33 | AP50 | 0.269 [0.015–0.564] | 0.479 [0.185–0.722] | +0.210 [-0.056 to +0.395] | 0.125 |
| Convict Tang | 33 | R | 0.273 [0.000–0.500] | 0.455 [0.237–0.714] | +0.182 [+0.045 to +0.375] | 0.027 |
| **Focus species (mean)** | 203 | AP50 | 0.215 [0.107–0.357] | 0.312 [0.192–0.422] | +0.096 [-0.009 to +0.174] | 0.075 |
| **Focus species (mean)** | 203 | R | 0.193 [0.092–0.299] | 0.285 [0.186–0.391] | +0.092 [+0.030 to +0.165] | 0.001 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.671 [0.608–0.727] | 0.662 [0.599–0.717] | -0.009 [-0.024 to +0.005] | 0.217 |
| Any fish (class-agnostic) | 4098 | R | 0.581 [0.536–0.626] | 0.586 [0.546–0.630] | +0.006 [-0.013 to +0.024] | 0.571 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.246 [0.217–0.304] | 0.263 [0.236–0.311] | +0.017 [-0.015 to +0.039] | 0.297 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.226 [0.194–0.265] | 0.259 [0.228–0.295] | +0.033 [+0.009 to +0.058] | 0.006 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.283 | 0.362 | +0.079 |
| Forcepsfish | AP50-95 | 0.161 | 0.224 | +0.062 |
| Forcepsfish | P | 0.571 | 0.517 | -0.054 |
| Forcepsfish | R | 0.215 | 0.323 | +0.108 |
| Forcepsfish | F1 | 0.312 | 0.397 | +0.085 |
| Orangespine Unicornfish | AP50 | 0.093 | 0.094 | +0.000 |
| Orangespine Unicornfish | AP50-95 | 0.066 | 0.066 | -0.000 |
| Orangespine Unicornfish | P | 0.233 | 0.222 | -0.011 |
| Orangespine Unicornfish | R | 0.091 | 0.078 | -0.013 |
| Orangespine Unicornfish | F1 | 0.131 | 0.115 | -0.015 |
| Convict Tang | AP50 | 0.269 | 0.479 | +0.210 |
| Convict Tang | AP50-95 | 0.226 | 0.393 | +0.167 |
| Convict Tang | P | 0.562 | 0.405 | -0.157 |
| Convict Tang | R | 0.273 | 0.455 | +0.182 |
| Convict Tang | F1 | 0.367 | 0.429 | +0.061 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.5208 | 0.5209 | 0.2917 | 0.375 |
| 0 | Orangespine Unicornfish | 6 | 0.3442 | 0.1654 | 0.3333 | 0.1667 |
| 0 | Convict Tang | 5 | 0.715 | 1.0 | 0.6 | 1.0 |
| 1 | Forcepsfish | 14 | 0.3061 | 0.3278 | 0.2857 | 0.2857 |
| 1 | Orangespine Unicornfish | 15 | 0.033 | 0.032 | 0.0 | 0.0 |
| 1 | Convict Tang | 3 | 0.7896 | 0.5933 | 0.6667 | 0.6667 |
| 2 | Forcepsfish | 16 | 0.3282 | 0.3503 | 0.1875 | 0.3125 |
| 2 | Orangespine Unicornfish | 5 | 0.3681 | 0.2735 | 0.4 | 0.4 |
| 2 | Convict Tang | 9 | 0.793 | 0.4211 | 0.4444 | 0.4444 |
| 3 | Forcepsfish | 16 | 0.2389 | 0.3185 | 0.3125 | 0.375 |
| 3 | Orangespine Unicornfish | 21 | 0.0147 | 0.0349 | 0.0 | 0.0 |
| 3 | Convict Tang | 7 | 0.0133 | 0.4984 | 0.0 | 0.1429 |
| 4 | Forcepsfish | 23 | 0.2078 | 0.3916 | 0.0435 | 0.2609 |
| 4 | Orangespine Unicornfish | 30 | 0.1473 | 0.2049 | 0.1 | 0.1 |
| 4 | Convict Tang | 9 | 0.0485 | 0.4673 | 0.0 | 0.3333 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
