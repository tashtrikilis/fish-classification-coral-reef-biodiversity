# Cross-validation: baseline vs copy-paste (mask AP, pooled over 5 test folds, yolo26l-seg, mask AP; seed 1)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference. Recall is at confidence 0.25. Primary endpoint: 'Focus species (mean)' AP50.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.405 [0.187–0.587] | 0.545 [0.330–0.713] | +0.141 [+0.026 to +0.280] | 0.016 |
| Forcepsfish | 93 | R | 0.301 [0.140–0.435] | 0.495 [0.310–0.653] | +0.194 [+0.110 to +0.305] | 0.000 |
| Orangespine Unicornfish | 77 | AP50 | 0.325 [0.097–0.433] | 0.224 [0.049–0.350] | -0.101 [-0.171 to +0.042] | 0.132 |
| Orangespine Unicornfish | 77 | R | 0.299 [0.107–0.398] | 0.247 [0.054–0.392] | -0.052 [-0.141 to +0.111] | 0.412 |
| Convict Tang | 33 | AP50 | 0.458 [0.130–0.691] | 0.586 [0.377–0.811] | +0.128 [-0.034 to +0.348] | 0.162 |
| Convict Tang | 33 | R | 0.424 [0.130–0.689] | 0.576 [0.439–0.792] | +0.152 [-0.083 to +0.450] | 0.243 |
| **Focus species (mean)** | 203 | AP50 | 0.396 [0.251–0.501] | 0.452 [0.326–0.560] | +0.056 [-0.002 to +0.140] | 0.057 |
| **Focus species (mean)** | 203 | R | 0.341 [0.205–0.445] | 0.439 [0.330–0.549] | +0.098 [+0.021 to +0.205] | 0.013 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.684 [0.620–0.743] | 0.688 [0.631–0.744] | +0.004 [-0.009 to +0.022] | 0.373 |
| Any fish (class-agnostic) | 4098 | R | 0.620 [0.576–0.666] | 0.632 [0.589–0.677] | +0.012 [-0.008 to +0.033] | 0.249 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.322 [0.288–0.376] | 0.359 [0.317–0.414] | +0.037 [+0.004 to +0.069] | 0.024 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.306 [0.268–0.354] | 0.318 [0.279–0.357] | +0.012 [-0.032 to +0.041] | 0.539 |

A difference counts as a finding only if its 95% interval excludes 0. The focus-species mean is the single pre-specified test; treat the per-species rows as supporting detail.

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.405 | 0.545 | +0.141 |
| Forcepsfish | AP50-95 | 0.241 | 0.374 | +0.133 |
| Forcepsfish | P | 0.583 | 0.708 | +0.124 |
| Forcepsfish | R | 0.301 | 0.495 | +0.194 |
| Forcepsfish | F1 | 0.397 | 0.582 | +0.185 |
| Orangespine Unicornfish | AP50 | 0.325 | 0.224 | -0.101 |
| Orangespine Unicornfish | AP50-95 | 0.226 | 0.158 | -0.068 |
| Orangespine Unicornfish | P | 0.479 | 0.475 | -0.004 |
| Orangespine Unicornfish | R | 0.299 | 0.247 | -0.052 |
| Orangespine Unicornfish | F1 | 0.368 | 0.325 | -0.043 |
| Convict Tang | AP50 | 0.458 | 0.586 | +0.128 |
| Convict Tang | AP50-95 | 0.380 | 0.487 | +0.108 |
| Convict Tang | P | 0.667 | 0.633 | -0.033 |
| Convict Tang | R | 0.424 | 0.576 | +0.152 |
| Convict Tang | F1 | 0.519 | 0.603 | +0.085 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.5524 | 0.6396 | 0.3333 | 0.4583 |
| 0 | Orangespine Unicornfish | 6 | 0.3584 | 0.3903 | 0.3333 | 0.5 |
| 0 | Convict Tang | 5 | 0.7737 | 0.9434 | 1.0 | 0.8 |
| 1 | Forcepsfish | 14 | 0.3899 | 0.4335 | 0.2857 | 0.4286 |
| 1 | Orangespine Unicornfish | 15 | 0.0953 | 0.0579 | 0.0667 | 0.0 |
| 1 | Convict Tang | 3 | 0.8076 | 1.0 | 0.3333 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4615 | 0.503 | 0.3125 | 0.3125 |
| 2 | Orangespine Unicornfish | 5 | 0.4135 | 0.5767 | 0.4 | 0.6 |
| 2 | Convict Tang | 9 | 0.6244 | 0.5143 | 0.5556 | 0.4444 |
| 3 | Forcepsfish | 16 | 0.4929 | 0.5729 | 0.5 | 0.625 |
| 3 | Orangespine Unicornfish | 21 | 0.3815 | 0.1498 | 0.2381 | 0.0952 |
| 3 | Convict Tang | 7 | 0.0 | 0.6276 | 0.0 | 0.4286 |
| 4 | Forcepsfish | 23 | 0.3022 | 0.7641 | 0.1304 | 0.6087 |
| 4 | Orangespine Unicornfish | 30 | 0.4807 | 0.3259 | 0.4333 | 0.3667 |
| 4 | Convict Tang | 9 | 0.5098 | 0.497 | 0.3333 | 0.5556 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
