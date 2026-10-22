# Cross-validation: baseline vs copy-paste (detection, pooled over 5 test folds)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference. Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.291 [0.082–0.517] | 0.405 [0.168–0.604] | +0.115 [-0.006 to +0.230] | 0.065 |
| Forcepsfish | 93 | R | 0.215 [0.029–0.430] | 0.376 [0.171–0.555] | +0.161 [+0.010 to +0.298] | 0.048 |
| Orangespine Unicornfish | 77 | AP50 | 0.197 [0.087–0.338] | 0.247 [0.108–0.397] | +0.050 [-0.067 to +0.162] | 0.398 |
| Orangespine Unicornfish | 77 | R | 0.117 [0.017–0.312] | 0.247 [0.138–0.400] | +0.130 [+0.000 to +0.200] | 0.064 |
| Convict Tang | 33 | AP50 | 0.440 [0.148–0.749] | 0.605 [0.355–0.824] | +0.165 [+0.026 to +0.346] | 0.030 |
| Convict Tang | 33 | R | 0.455 [0.187–0.800] | 0.576 [0.292–0.818] | +0.121 [-0.054 to +0.308] | 0.230 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.695 [0.636–0.746] | 0.678 [0.617–0.732] | -0.017 [-0.035 to +0.001] | 0.078 |
| Any fish (class-agnostic) | 4098 | R | 0.632 [0.588–0.675] | 0.600 [0.561–0.642] | -0.033 [-0.057 to -0.009] | 0.005 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.298 [0.273–0.367] | 0.311 [0.283–0.366] | +0.013 [-0.026 to +0.036] | 0.576 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.279 [0.242–0.344] | 0.278 [0.247–0.317] | -0.002 [-0.055 to +0.028] | 0.802 |

A difference counts as a finding only if its 95% interval excludes 0. With several rows tested, treat a single p near 0.05 with caution.

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.241 | 0.4997 | 0.0417 | 0.25 |
| 0 | Orangespine Unicornfish | 6 | 0.4864 | 0.4134 | 0.5 | 0.3333 |
| 0 | Convict Tang | 5 | 1.0 | 0.8597 | 1.0 | 0.8 |
| 1 | Forcepsfish | 14 | 0.4192 | 0.4136 | 0.3571 | 0.3571 |
| 1 | Orangespine Unicornfish | 15 | 0.2437 | 0.224 | 0.0667 | 0.1333 |
| 1 | Convict Tang | 3 | 0.9158 | 1.0 | 1.0 | 1.0 |
| 2 | Forcepsfish | 16 | 0.5432 | 0.4735 | 0.1875 | 0.4375 |
| 2 | Orangespine Unicornfish | 5 | 0.2545 | 0.2409 | 0.4 | 0.6 |
| 2 | Convict Tang | 9 | 0.6547 | 0.6576 | 0.4444 | 0.6667 |
| 3 | Forcepsfish | 16 | 0.5715 | 0.593 | 0.6875 | 0.75 |
| 3 | Orangespine Unicornfish | 21 | 0.2243 | 0.41 | 0.1429 | 0.381 |
| 3 | Convict Tang | 7 | 0.0193 | 0.1437 | 0.0 | 0.0 |
| 4 | Forcepsfish | 23 | 0.1411 | 0.493 | 0.0 | 0.2174 |
| 4 | Orangespine Unicornfish | 30 | 0.2845 | 0.212 | 0.0 | 0.1333 |
| 4 | Convict Tang | 9 | 0.2399 | 0.6089 | 0.3333 | 0.6667 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
