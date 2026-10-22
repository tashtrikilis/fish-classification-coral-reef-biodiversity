# Cross-validation: baseline vs copy-paste (mask AP, pooled over 5 test folds, yolo26l-seg, mask AP; seed 0)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference. Recall is at confidence 0.25. Primary endpoint: 'Focus species (mean)' AP50.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.383 [0.178–0.573] | 0.474 [0.250–0.658] | +0.091 [-0.035 to +0.224] | 0.168 |
| Forcepsfish | 93 | R | 0.290 [0.130–0.446] | 0.430 [0.264–0.575] | +0.140 [+0.029 to +0.259] | 0.022 |
| Orangespine Unicornfish | 77 | AP50 | 0.125 [0.038–0.265] | 0.264 [0.054–0.446] | +0.139 [-0.069 to +0.264] | 0.262 |
| Orangespine Unicornfish | 77 | R | 0.130 [0.051–0.255] | 0.260 [0.053–0.394] | +0.130 [-0.091 to +0.240] | 0.296 |
| Convict Tang | 33 | AP50 | 0.469 [0.104–0.722] | 0.573 [0.290–0.798] | +0.104 [-0.065 to +0.304] | 0.197 |
| Convict Tang | 33 | R | 0.364 [0.067–0.596] | 0.576 [0.304–0.870] | +0.212 [+0.068 to +0.460] | 0.011 |
| **Focus species (mean)** | 203 | AP50 | 0.326 [0.185–0.457] | 0.437 [0.288–0.561] | +0.111 [+0.005 to +0.193] | 0.039 |
| **Focus species (mean)** | 203 | R | 0.261 [0.143–0.376] | 0.422 [0.282–0.549] | +0.161 [+0.054 to +0.264] | 0.006 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.675 [0.623–0.725] | 0.679 [0.619–0.733] | +0.004 [-0.017 to +0.021] | 0.703 |
| Any fish (class-agnostic) | 4098 | R | 0.614 [0.570–0.661] | 0.624 [0.583–0.665] | +0.010 [-0.023 to +0.039] | 0.539 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.283 [0.249–0.340] | 0.305 [0.269–0.376] | +0.022 [-0.002 to +0.055] | 0.060 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.268 [0.231–0.310] | 0.312 [0.274–0.375] | +0.044 [+0.010 to +0.101] | 0.014 |

A difference counts as a finding only if its 95% interval excludes 0. The focus-species mean is the single pre-specified test; treat the per-species rows as supporting detail.

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.383 | 0.474 | +0.091 |
| Forcepsfish | AP50-95 | 0.246 | 0.316 | +0.070 |
| Forcepsfish | P | 0.551 | 0.597 | +0.046 |
| Forcepsfish | R | 0.290 | 0.430 | +0.140 |
| Forcepsfish | F1 | 0.380 | 0.500 | +0.120 |
| Orangespine Unicornfish | AP50 | 0.125 | 0.264 | +0.139 |
| Orangespine Unicornfish | AP50-95 | 0.092 | 0.185 | +0.094 |
| Orangespine Unicornfish | P | 0.270 | 0.465 | +0.195 |
| Orangespine Unicornfish | R | 0.130 | 0.260 | +0.130 |
| Orangespine Unicornfish | F1 | 0.175 | 0.333 | +0.158 |
| Convict Tang | AP50 | 0.469 | 0.573 | +0.104 |
| Convict Tang | AP50-95 | 0.380 | 0.482 | +0.102 |
| Convict Tang | P | 0.632 | 0.731 | +0.099 |
| Convict Tang | R | 0.364 | 0.576 | +0.212 |
| Convict Tang | F1 | 0.462 | 0.644 | +0.183 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.4071 | 0.6235 | 0.2917 | 0.3333 |
| 0 | Orangespine Unicornfish | 6 | 0.3337 | 0.4491 | 0.3333 | 0.5 |
| 0 | Convict Tang | 5 | 0.8868 | 0.9038 | 0.8 | 1.0 |
| 1 | Forcepsfish | 14 | 0.4504 | 0.4338 | 0.3571 | 0.3571 |
| 1 | Orangespine Unicornfish | 15 | 0.0303 | 0.0219 | 0.0667 | 0.0 |
| 1 | Convict Tang | 3 | 0.3894 | 0.8342 | 0.3333 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4727 | 0.4787 | 0.4375 | 0.375 |
| 2 | Orangespine Unicornfish | 5 | 0.576 | 0.2619 | 0.4 | 0.2 |
| 2 | Convict Tang | 9 | 0.724 | 0.6434 | 0.5556 | 0.5556 |
| 3 | Forcepsfish | 16 | 0.3435 | 0.5204 | 0.3125 | 0.5625 |
| 3 | Orangespine Unicornfish | 21 | 0.0623 | 0.2074 | 0.0476 | 0.1429 |
| 3 | Convict Tang | 7 | 0.0355 | 0.2465 | 0.0 | 0.1429 |
| 4 | Forcepsfish | 23 | 0.4294 | 0.5673 | 0.1304 | 0.5217 |
| 4 | Orangespine Unicornfish | 30 | 0.2142 | 0.4873 | 0.1333 | 0.4333 |
| 4 | Convict Tang | 9 | 0.4542 | 0.5371 | 0.2222 | 0.5556 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
