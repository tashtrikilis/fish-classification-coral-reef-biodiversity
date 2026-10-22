# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26s-seg, box AP; seed 0)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference. Recall is at confidence 0.25. Primary endpoint: 'Focus species (mean)' AP50.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.257 [0.093–0.438] | 0.335 [0.191–0.471] | +0.078 [-0.052 to +0.194] | 0.238 |
| Forcepsfish | 93 | R | 0.215 [0.072–0.328] | 0.312 [0.197–0.415] | +0.097 [+0.028 to +0.190] | 0.010 |
| Orangespine Unicornfish | 77 | AP50 | 0.092 [0.014–0.213] | 0.087 [0.024–0.176] | -0.005 [-0.090 to +0.049] | 0.942 |
| Orangespine Unicornfish | 77 | R | 0.091 [0.000–0.213] | 0.078 [0.018–0.154] | -0.013 [-0.081 to +0.042] | 0.799 |
| Convict Tang | 33 | AP50 | 0.269 [0.015–0.564] | 0.479 [0.185–0.722] | +0.210 [-0.056 to +0.395] | 0.125 |
| Convict Tang | 33 | R | 0.273 [0.000–0.500] | 0.455 [0.237–0.714] | +0.182 [+0.045 to +0.375] | 0.027 |
| **Focus species (mean)** | 203 | AP50 | 0.206 [0.101–0.345] | 0.301 [0.185–0.408] | +0.094 [-0.010 to +0.173] | 0.088 |
| **Focus species (mean)** | 203 | R | 0.193 [0.092–0.299] | 0.281 [0.183–0.386] | +0.089 [+0.025 to +0.162] | 0.003 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.663 [0.601–0.719] | 0.653 [0.590–0.708] | -0.010 [-0.025 to +0.004] | 0.159 |
| Any fish (class-agnostic) | 4098 | R | 0.577 [0.532–0.623] | 0.582 [0.542–0.626] | +0.005 [-0.013 to +0.024] | 0.575 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.243 [0.215–0.300] | 0.259 [0.233–0.307] | +0.016 [-0.015 to +0.038] | 0.325 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.225 [0.194–0.264] | 0.258 [0.227–0.294] | +0.033 [+0.008 to +0.058] | 0.007 |

A difference counts as a finding only if its 95% interval excludes 0. The focus-species mean is the single pre-specified test; treat the per-species rows as supporting detail.

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.257 | 0.335 | +0.078 |
| Forcepsfish | AP50-95 | 0.142 | 0.211 | +0.069 |
| Forcepsfish | P | 0.571 | 0.500 | -0.071 |
| Forcepsfish | R | 0.215 | 0.312 | +0.097 |
| Forcepsfish | F1 | 0.312 | 0.384 | +0.072 |
| Orangespine Unicornfish | AP50 | 0.092 | 0.087 | -0.005 |
| Orangespine Unicornfish | AP50-95 | 0.060 | 0.055 | -0.004 |
| Orangespine Unicornfish | P | 0.233 | 0.222 | -0.011 |
| Orangespine Unicornfish | R | 0.091 | 0.078 | -0.013 |
| Orangespine Unicornfish | F1 | 0.131 | 0.115 | -0.015 |
| Convict Tang | AP50 | 0.269 | 0.479 | +0.210 |
| Convict Tang | AP50-95 | 0.219 | 0.387 | +0.169 |
| Convict Tang | P | 0.562 | 0.405 | -0.157 |
| Convict Tang | R | 0.273 | 0.455 | +0.182 |
| Convict Tang | F1 | 0.367 | 0.429 | +0.061 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.5141 | 0.4401 | 0.2917 | 0.3333 |
| 0 | Orangespine Unicornfish | 6 | 0.3442 | 0.1654 | 0.3333 | 0.1667 |
| 0 | Convict Tang | 5 | 0.715 | 1.0 | 0.6 | 1.0 |
| 1 | Forcepsfish | 14 | 0.3061 | 0.3278 | 0.2857 | 0.2857 |
| 1 | Orangespine Unicornfish | 15 | 0.0295 | 0.0292 | 0.0 | 0.0 |
| 1 | Convict Tang | 3 | 0.7896 | 0.5933 | 0.6667 | 0.6667 |
| 2 | Forcepsfish | 16 | 0.2637 | 0.3503 | 0.1875 | 0.3125 |
| 2 | Orangespine Unicornfish | 5 | 0.3681 | 0.2735 | 0.4 | 0.4 |
| 2 | Convict Tang | 9 | 0.793 | 0.4211 | 0.4444 | 0.4444 |
| 3 | Forcepsfish | 16 | 0.2041 | 0.2969 | 0.3125 | 0.375 |
| 3 | Orangespine Unicornfish | 21 | 0.0147 | 0.0349 | 0.0 | 0.0 |
| 3 | Convict Tang | 7 | 0.0133 | 0.4984 | 0.0 | 0.1429 |
| 4 | Forcepsfish | 23 | 0.1783 | 0.3736 | 0.0435 | 0.2609 |
| 4 | Orangespine Unicornfish | 30 | 0.1473 | 0.1839 | 0.1 | 0.1 |
| 4 | Convict Tang | 9 | 0.0485 | 0.4673 | 0.0 | 0.3333 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
