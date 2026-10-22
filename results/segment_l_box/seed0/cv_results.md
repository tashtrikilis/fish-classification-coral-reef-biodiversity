# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26l-seg, box AP; seed 0)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.356 [0.169–0.521] | 0.456 [0.240–0.643] | +0.100 [-0.031 to +0.237] | 0.130 |
| Forcepsfish | 93 | R | 0.269 [0.127–0.407] | 0.419 [0.259–0.566] | +0.151 [+0.047 to +0.266] | 0.005 |
| Orangespine Unicornfish | 77 | AP50 | 0.124 [0.038–0.263] | 0.263 [0.053–0.445] | +0.140 [-0.069 to +0.267] | 0.265 |
| Orangespine Unicornfish | 77 | R | 0.130 [0.051–0.255] | 0.260 [0.053–0.394] | +0.130 [-0.091 to +0.240] | 0.296 |
| Convict Tang | 33 | AP50 | 0.469 [0.104–0.722] | 0.573 [0.290–0.798] | +0.104 [-0.065 to +0.304] | 0.197 |
| Convict Tang | 33 | R | 0.364 [0.067–0.596] | 0.576 [0.304–0.870] | +0.212 [+0.068 to +0.460] | 0.011 |
| **Focus species (mean)** | 203 | AP50 | 0.316 [0.177–0.444] | 0.431 [0.282–0.555] | +0.115 [+0.011 to +0.195] | 0.031 |
| **Focus species (mean)** | 203 | R | 0.254 [0.139–0.366] | 0.418 [0.279–0.548] | +0.164 [+0.061 to +0.267] | 0.004 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.667 [0.617–0.716] | 0.671 [0.614–0.726] | +0.004 [-0.016 to +0.022] | 0.630 |
| Any fish (class-agnostic) | 4098 | R | 0.611 [0.568–0.657] | 0.620 [0.579–0.661] | +0.009 [-0.023 to +0.038] | 0.560 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.280 [0.247–0.336] | 0.301 [0.266–0.372] | +0.021 [-0.002 to +0.054] | 0.066 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.266 [0.230–0.307] | 0.310 [0.272–0.373] | +0.044 [+0.011 to +0.100] | 0.014 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.356 | 0.456 | +0.100 |
| Forcepsfish | AP50-95 | 0.226 | 0.314 | +0.088 |
| Forcepsfish | P | 0.510 | 0.582 | +0.072 |
| Forcepsfish | R | 0.269 | 0.419 | +0.151 |
| Forcepsfish | F1 | 0.352 | 0.488 | +0.135 |
| Orangespine Unicornfish | AP50 | 0.124 | 0.263 | +0.140 |
| Orangespine Unicornfish | AP50-95 | 0.075 | 0.190 | +0.115 |
| Orangespine Unicornfish | P | 0.270 | 0.465 | +0.195 |
| Orangespine Unicornfish | R | 0.130 | 0.260 | +0.130 |
| Orangespine Unicornfish | F1 | 0.175 | 0.333 | +0.158 |
| Convict Tang | AP50 | 0.469 | 0.573 | +0.104 |
| Convict Tang | AP50-95 | 0.383 | 0.501 | +0.118 |
| Convict Tang | P | 0.632 | 0.731 | +0.099 |
| Convict Tang | R | 0.364 | 0.576 | +0.212 |
| Convict Tang | F1 | 0.462 | 0.644 | +0.183 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.3502 | 0.5198 | 0.25 | 0.2917 |
| 0 | Orangespine Unicornfish | 6 | 0.3337 | 0.4491 | 0.3333 | 0.5 |
| 0 | Convict Tang | 5 | 0.8868 | 0.9038 | 0.8 | 1.0 |
| 1 | Forcepsfish | 14 | 0.4504 | 0.4338 | 0.3571 | 0.3571 |
| 1 | Orangespine Unicornfish | 15 | 0.0303 | 0.0179 | 0.0667 | 0.0 |
| 1 | Convict Tang | 3 | 0.3894 | 0.8342 | 0.3333 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4219 | 0.4787 | 0.375 | 0.375 |
| 2 | Orangespine Unicornfish | 5 | 0.576 | 0.2619 | 0.4 | 0.2 |
| 2 | Convict Tang | 9 | 0.724 | 0.6434 | 0.5556 | 0.5556 |
| 3 | Forcepsfish | 16 | 0.33 | 0.5204 | 0.3125 | 0.5625 |
| 3 | Orangespine Unicornfish | 21 | 0.0623 | 0.2074 | 0.0476 | 0.1429 |
| 3 | Convict Tang | 7 | 0.0355 | 0.2465 | 0.0 | 0.1429 |
| 4 | Forcepsfish | 23 | 0.4268 | 0.5604 | 0.1304 | 0.5217 |
| 4 | Orangespine Unicornfish | 30 | 0.2104 | 0.4873 | 0.1333 | 0.4333 |
| 4 | Convict Tang | 9 | 0.4542 | 0.5371 | 0.2222 | 0.5556 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
