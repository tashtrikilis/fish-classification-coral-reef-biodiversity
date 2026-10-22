# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26l detection, box AP; seed 0)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.392 [0.202–0.573] | 0.513 [0.324–0.680] | +0.121 [+0.019 to +0.218] | 0.017 |
| Forcepsfish | 93 | R | 0.258 [0.132–0.415] | 0.462 [0.291–0.630] | +0.204 [+0.097 to +0.302] | 0.000 |
| Orangespine Unicornfish | 77 | AP50 | 0.276 [0.111–0.385] | 0.205 [0.062–0.378] | -0.070 [-0.190 to +0.119] | 0.336 |
| Orangespine Unicornfish | 77 | R | 0.273 [0.111–0.368] | 0.195 [0.053–0.378] | -0.078 [-0.183 to +0.154] | 0.384 |
| Convict Tang | 33 | AP50 | 0.447 [0.135–0.698] | 0.491 [0.195–0.751] | +0.044 [-0.126 to +0.211] | 0.548 |
| Convict Tang | 33 | R | 0.364 [0.111–0.600] | 0.545 [0.259–0.864] | +0.182 [+0.000 to +0.444] | 0.085 |
| **Focus species (mean)** | 203 | AP50 | 0.372 [0.236–0.489] | 0.403 [0.261–0.548] | +0.031 [-0.036 to +0.123] | 0.428 |
| **Focus species (mean)** | 203 | R | 0.298 [0.184–0.397] | 0.401 [0.262–0.556] | +0.103 [+0.021 to +0.249] | 0.010 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.690 [0.634–0.743] | 0.673 [0.615–0.727] | -0.017 [-0.034 to +0.001] | 0.066 |
| Any fish (class-agnostic) | 4098 | R | 0.610 [0.575–0.649] | 0.595 [0.554–0.638] | -0.015 [-0.034 to +0.005] | 0.145 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.311 [0.282–0.364] | 0.317 [0.286–0.372] | +0.006 [-0.016 to +0.033] | 0.561 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.274 [0.247–0.314] | 0.295 [0.260–0.346] | +0.022 [-0.006 to +0.057] | 0.139 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.392 | 0.513 | +0.121 |
| Forcepsfish | AP50-95 | 0.258 | 0.319 | +0.061 |
| Forcepsfish | P | 0.571 | 0.741 | +0.170 |
| Forcepsfish | R | 0.258 | 0.462 | +0.204 |
| Forcepsfish | F1 | 0.356 | 0.570 | +0.214 |
| Orangespine Unicornfish | AP50 | 0.276 | 0.205 | -0.070 |
| Orangespine Unicornfish | AP50-95 | 0.167 | 0.124 | -0.043 |
| Orangespine Unicornfish | P | 0.656 | 0.500 | -0.156 |
| Orangespine Unicornfish | R | 0.273 | 0.195 | -0.078 |
| Orangespine Unicornfish | F1 | 0.385 | 0.280 | -0.105 |
| Convict Tang | AP50 | 0.447 | 0.491 | +0.044 |
| Convict Tang | AP50-95 | 0.374 | 0.412 | +0.038 |
| Convict Tang | P | 0.667 | 0.429 | -0.238 |
| Convict Tang | R | 0.364 | 0.545 | +0.182 |
| Convict Tang | F1 | 0.471 | 0.480 | +0.009 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.5405 | 0.6196 | 0.1667 | 0.5417 |
| 0 | Orangespine Unicornfish | 6 | 0.4045 | 0.5046 | 0.3333 | 0.5 |
| 0 | Convict Tang | 5 | 0.7378 | 0.934 | 0.6 | 1.0 |
| 1 | Forcepsfish | 14 | 0.4366 | 0.4069 | 0.4286 | 0.4286 |
| 1 | Orangespine Unicornfish | 15 | 0.0677 | 0.2958 | 0.0 | 0.2 |
| 1 | Convict Tang | 3 | 0.8076 | 0.8653 | 0.6667 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4229 | 0.4818 | 0.25 | 0.4375 |
| 2 | Orangespine Unicornfish | 5 | 0.6919 | 0.3347 | 0.8 | 0.4 |
| 2 | Convict Tang | 9 | 0.5529 | 0.5594 | 0.4444 | 0.5556 |
| 3 | Forcepsfish | 16 | 0.3854 | 0.5047 | 0.3125 | 0.5 |
| 3 | Orangespine Unicornfish | 21 | 0.2777 | 0.046 | 0.2381 | 0.0 |
| 3 | Convict Tang | 7 | 0.0 | 0.0421 | 0.0 | 0.0 |
| 4 | Forcepsfish | 23 | 0.4103 | 0.6245 | 0.2174 | 0.3913 |
| 4 | Orangespine Unicornfish | 30 | 0.4002 | 0.3382 | 0.3333 | 0.2333 |
| 4 | Convict Tang | 9 | 0.5776 | 0.5191 | 0.3333 | 0.5556 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
