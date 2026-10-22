# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26l-seg, box AP; seed 1)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.380 [0.178–0.554] | 0.543 [0.328–0.711] | +0.163 [+0.050 to +0.292] | 0.004 |
| Forcepsfish | 93 | R | 0.290 [0.138–0.416] | 0.495 [0.310–0.653] | +0.204 [+0.119 to +0.313] | 0.000 |
| Orangespine Unicornfish | 77 | AP50 | 0.323 [0.094–0.432] | 0.224 [0.049–0.350] | -0.099 [-0.171 to +0.046] | 0.146 |
| Orangespine Unicornfish | 77 | R | 0.299 [0.107–0.398] | 0.247 [0.054–0.392] | -0.052 [-0.141 to +0.111] | 0.412 |
| Convict Tang | 33 | AP50 | 0.458 [0.130–0.691] | 0.586 [0.377–0.811] | +0.128 [-0.034 to +0.348] | 0.162 |
| Convict Tang | 33 | R | 0.424 [0.130–0.689] | 0.576 [0.439–0.792] | +0.152 [-0.083 to +0.450] | 0.243 |
| **Focus species (mean)** | 203 | AP50 | 0.387 [0.244–0.489] | 0.451 [0.325–0.560] | +0.064 [+0.004 to +0.149] | 0.038 |
| **Focus species (mean)** | 203 | R | 0.338 [0.202–0.440] | 0.439 [0.330–0.549] | +0.101 [+0.025 to +0.208] | 0.011 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.676 [0.614–0.736] | 0.684 [0.627–0.738] | +0.008 [-0.008 to +0.023] | 0.356 |
| Any fish (class-agnostic) | 4098 | R | 0.616 [0.572–0.662] | 0.629 [0.586–0.674] | +0.013 [-0.007 to +0.035] | 0.213 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.318 [0.285–0.370] | 0.355 [0.315–0.410] | +0.037 [+0.005 to +0.070] | 0.021 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.304 [0.267–0.351] | 0.317 [0.279–0.356] | +0.013 [-0.032 to +0.042] | 0.522 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.380 | 0.543 | +0.163 |
| Forcepsfish | AP50-95 | 0.232 | 0.359 | +0.127 |
| Forcepsfish | P | 0.562 | 0.708 | +0.145 |
| Forcepsfish | R | 0.290 | 0.495 | +0.204 |
| Forcepsfish | F1 | 0.383 | 0.582 | +0.199 |
| Orangespine Unicornfish | AP50 | 0.323 | 0.224 | -0.099 |
| Orangespine Unicornfish | AP50-95 | 0.200 | 0.140 | -0.060 |
| Orangespine Unicornfish | P | 0.479 | 0.475 | -0.004 |
| Orangespine Unicornfish | R | 0.299 | 0.247 | -0.052 |
| Orangespine Unicornfish | F1 | 0.368 | 0.325 | -0.043 |
| Convict Tang | AP50 | 0.458 | 0.586 | +0.128 |
| Convict Tang | AP50-95 | 0.368 | 0.488 | +0.120 |
| Convict Tang | P | 0.667 | 0.633 | -0.033 |
| Convict Tang | R | 0.424 | 0.576 | +0.152 |
| Convict Tang | F1 | 0.519 | 0.603 | +0.085 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.4859 | 0.6396 | 0.2917 | 0.4583 |
| 0 | Orangespine Unicornfish | 6 | 0.3584 | 0.3903 | 0.3333 | 0.5 |
| 0 | Convict Tang | 5 | 0.7737 | 0.9434 | 1.0 | 0.8 |
| 1 | Forcepsfish | 14 | 0.3899 | 0.4335 | 0.2857 | 0.4286 |
| 1 | Orangespine Unicornfish | 15 | 0.0751 | 0.0579 | 0.0667 | 0.0 |
| 1 | Convict Tang | 3 | 0.8076 | 1.0 | 0.3333 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4083 | 0.503 | 0.3125 | 0.3125 |
| 2 | Orangespine Unicornfish | 5 | 0.4135 | 0.5767 | 0.4 | 0.6 |
| 2 | Convict Tang | 9 | 0.6244 | 0.5143 | 0.5556 | 0.4444 |
| 3 | Forcepsfish | 16 | 0.4796 | 0.5681 | 0.5 | 0.625 |
| 3 | Orangespine Unicornfish | 21 | 0.3815 | 0.1498 | 0.2381 | 0.0952 |
| 3 | Convict Tang | 7 | 0.0 | 0.6276 | 0.0 | 0.4286 |
| 4 | Forcepsfish | 23 | 0.3022 | 0.759 | 0.1304 | 0.6087 |
| 4 | Orangespine Unicornfish | 30 | 0.4807 | 0.3259 | 0.4333 | 0.3667 |
| 4 | Convict Tang | 9 | 0.5098 | 0.497 | 0.3333 | 0.5556 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
