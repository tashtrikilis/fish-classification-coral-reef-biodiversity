# Cross-validation: baseline vs copy-paste (detection, pooled over 5 test folds, seed 1)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference. Recall is at confidence 0.25. Primary endpoint: 'Focus species (mean)' AP50.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.453 [0.192–0.651] | 0.417 [0.218–0.628] | -0.036 [-0.224 to +0.194] | 0.852 |
| Forcepsfish | 93 | R | 0.344 [0.113–0.525] | 0.366 [0.197–0.571] | +0.022 [-0.188 to +0.309] | 0.850 |
| Orangespine Unicornfish | 77 | AP50 | 0.171 [0.028–0.308] | 0.266 [0.131–0.395] | +0.095 [+0.015 to +0.232] | 0.023 |
| Orangespine Unicornfish | 77 | R | 0.156 [0.035–0.318] | 0.247 [0.156–0.429] | +0.091 [+0.017 to +0.222] | 0.036 |
| Convict Tang | 33 | AP50 | 0.461 [0.147–0.741] | 0.603 [0.229–0.882] | +0.142 [+0.050 to +0.238] | 0.004 |
| Convict Tang | 33 | R | 0.364 [0.100–0.682] | 0.606 [0.250–0.885] | +0.242 [+0.051 to +0.438] | 0.019 |
| **Focus species (mean)** | 203 | AP50 | 0.362 [0.208–0.497] | 0.429 [0.272–0.580] | +0.067 [-0.001 to +0.159] | 0.056 |
| **Focus species (mean)** | 203 | R | 0.288 [0.151–0.430] | 0.406 [0.250–0.554] | +0.118 [+0.020 to +0.249] | 0.012 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.692 [0.635–0.744] | 0.696 [0.636–0.751] | +0.004 [-0.011 to +0.018] | 0.626 |
| Any fish (class-agnostic) | 4098 | R | 0.617 [0.578–0.658] | 0.613 [0.573–0.655] | -0.005 [-0.023 to +0.014] | 0.603 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.305 [0.270–0.361] | 0.318 [0.290–0.387] | +0.012 [-0.005 to +0.060] | 0.137 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.279 [0.240–0.325] | 0.299 [0.264–0.386] | +0.021 [-0.004 to +0.094] | 0.094 |

A difference counts as a finding only if its 95% interval excludes 0. The focus-species mean is the single pre-specified test; treat the per-species rows as supporting detail.

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.453 | 0.417 | -0.036 |
| Forcepsfish | AP50-95 | 0.269 | 0.295 | +0.026 |
| Forcepsfish | P | 0.780 | 0.596 | -0.184 |
| Forcepsfish | R | 0.344 | 0.366 | +0.022 |
| Forcepsfish | F1 | 0.478 | 0.453 | -0.024 |
| Orangespine Unicornfish | AP50 | 0.171 | 0.266 | +0.095 |
| Orangespine Unicornfish | AP50-95 | 0.107 | 0.175 | +0.068 |
| Orangespine Unicornfish | P | 0.316 | 0.487 | +0.171 |
| Orangespine Unicornfish | R | 0.156 | 0.247 | +0.091 |
| Orangespine Unicornfish | F1 | 0.209 | 0.328 | +0.119 |
| Convict Tang | AP50 | 0.461 | 0.603 | +0.142 |
| Convict Tang | AP50-95 | 0.378 | 0.489 | +0.112 |
| Convict Tang | P | 0.571 | 0.606 | +0.035 |
| Convict Tang | R | 0.364 | 0.606 | +0.242 |
| Convict Tang | F1 | 0.444 | 0.606 | +0.162 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.6746 | 0.428 | 0.4583 | 0.1667 |
| 0 | Orangespine Unicornfish | 6 | 0.2359 | 0.3231 | 0.1667 | 0.3333 |
| 0 | Convict Tang | 5 | 0.9434 | 1.0 | 1.0 | 1.0 |
| 1 | Forcepsfish | 14 | 0.3765 | 0.4066 | 0.3571 | 0.4286 |
| 1 | Orangespine Unicornfish | 15 | 0.0924 | 0.3327 | 0.0667 | 0.2667 |
| 1 | Convict Tang | 3 | 1.0 | 1.0 | 0.6667 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4182 | 0.4644 | 0.1875 | 0.3125 |
| 2 | Orangespine Unicornfish | 5 | 0.3338 | 0.5432 | 0.8 | 0.6 |
| 2 | Convict Tang | 9 | 0.5545 | 0.7506 | 0.3333 | 0.7778 |
| 3 | Forcepsfish | 16 | 0.6331 | 0.2881 | 0.6875 | 0.375 |
| 3 | Orangespine Unicornfish | 21 | 0.0891 | 0.1774 | 0.0 | 0.1429 |
| 3 | Convict Tang | 7 | 0.0 | 0.0376 | 0.0 | 0.0 |
| 4 | Forcepsfish | 23 | 0.2814 | 0.6727 | 0.087 | 0.5652 |
| 4 | Orangespine Unicornfish | 30 | 0.402 | 0.3409 | 0.2 | 0.2333 |
| 4 | Convict Tang | 9 | 0.455 | 0.5624 | 0.2222 | 0.5556 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
