# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26s-seg, box AP; seeds 0 + 1, averaged)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference. Recall is at confidence 0.25. Primary endpoint: 'Focus species (mean)' AP50.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.260 [0.111–0.408] | 0.386 [0.219–0.532] | +0.126 [+0.009 to +0.224] | 0.029 |
| Forcepsfish | 93 | R | 0.215 [0.102–0.309] | 0.344 [0.216–0.453] | +0.129 [+0.059 to +0.227] | 0.000 |
| Orangespine Unicornfish | 77 | AP50 | 0.100 [0.018–0.208] | 0.097 [0.039–0.206] | -0.004 [-0.058 to +0.076] | 0.999 |
| Orangespine Unicornfish | 77 | R | 0.097 [0.016–0.186] | 0.084 [0.041–0.210] | -0.013 [-0.063 to +0.085] | 0.866 |
| Convict Tang | 33 | AP50 | 0.338 [0.065–0.607] | 0.551 [0.330–0.746] | +0.212 [+0.016 to +0.373] | 0.032 |
| Convict Tang | 33 | R | 0.273 [0.042–0.514] | 0.591 [0.428–0.806] | +0.318 [+0.239 to +0.429] | 0.000 |
| **Focus species (mean)** | 203 | AP50 | 0.233 [0.123–0.348] | 0.345 [0.245–0.443] | +0.112 [+0.040 to +0.176] | 0.004 |
| **Focus species (mean)** | 203 | R | 0.195 [0.102–0.293] | 0.340 [0.263–0.437] | +0.145 [+0.108 to +0.195] | 0.000 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.668 [0.610–0.720] | 0.658 [0.599–0.712] | -0.010 [-0.023 to +0.004] | 0.135 |
| Any fish (class-agnostic) | 4098 | R | 0.583 [0.542–0.625] | 0.587 [0.547–0.630] | +0.004 [-0.008 to +0.016] | 0.556 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.252 [0.225–0.299] | 0.272 [0.247–0.318] | +0.020 [+0.004 to +0.037] | 0.017 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.232 [0.201–0.266] | 0.270 [0.240–0.302] | +0.038 [+0.026 to +0.053] | 0.000 |

A difference counts as a finding only if its 95% interval excludes 0. The focus-species mean is the single pre-specified test; treat the per-species rows as supporting detail.

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.260 | 0.386 | +0.126 |
| Forcepsfish | AP50-95 | 0.136 | 0.257 | +0.121 |
| Forcepsfish | P | 0.513 | 0.515 | +0.002 |
| Forcepsfish | R | 0.215 | 0.344 | +0.129 |
| Forcepsfish | F1 | 0.302 | 0.412 | +0.110 |
| Orangespine Unicornfish | AP50 | 0.100 | 0.097 | -0.004 |
| Orangespine Unicornfish | AP50-95 | 0.058 | 0.059 | +0.001 |
| Orangespine Unicornfish | P | 0.265 | 0.263 | -0.002 |
| Orangespine Unicornfish | R | 0.097 | 0.084 | -0.013 |
| Orangespine Unicornfish | F1 | 0.142 | 0.128 | -0.015 |
| Convict Tang | AP50 | 0.338 | 0.551 | +0.212 |
| Convict Tang | AP50-95 | 0.279 | 0.439 | +0.161 |
| Convict Tang | P | 0.627 | 0.438 | -0.189 |
| Convict Tang | R | 0.273 | 0.591 | +0.318 |
| Convict Tang | F1 | 0.379 | 0.500 | +0.121 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.4932 | 0.4107 | 0.2708 | 0.3125 |
| 0 | Orangespine Unicornfish | 6 | 0.3176 | 0.2824 | 0.25 | 0.25 |
| 0 | Convict Tang | 5 | 0.7938 | 0.9257 | 0.7 | 1.0 |
| 1 | Forcepsfish | 14 | 0.2686 | 0.3593 | 0.2143 | 0.3214 |
| 1 | Orangespine Unicornfish | 15 | 0.0341 | 0.0838 | 0.0333 | 0.0667 |
| 1 | Convict Tang | 3 | 0.7986 | 0.7005 | 0.6667 | 0.8333 |
| 2 | Forcepsfish | 16 | 0.28 | 0.384 | 0.1875 | 0.2812 |
| 2 | Orangespine Unicornfish | 5 | 0.2523 | 0.3462 | 0.3 | 0.4 |
| 2 | Convict Tang | 9 | 0.66 | 0.4971 | 0.3333 | 0.5556 |
| 3 | Forcepsfish | 16 | 0.2868 | 0.4079 | 0.375 | 0.4375 |
| 3 | Orangespine Unicornfish | 21 | 0.0168 | 0.0295 | 0.0 | 0.0238 |
| 3 | Convict Tang | 7 | 0.0233 | 0.482 | 0.0 | 0.3571 |
| 4 | Forcepsfish | 23 | 0.1722 | 0.4832 | 0.0652 | 0.3696 |
| 4 | Orangespine Unicornfish | 30 | 0.2363 | 0.1453 | 0.1333 | 0.05 |
| 4 | Convict Tang | 9 | 0.179 | 0.5851 | 0.0556 | 0.5 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
