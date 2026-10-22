# Cross-validation: baseline vs copy-paste (mask AP, pooled over 5 test folds, yolo26s-seg, mask AP; seeds 0 + 1, averaged)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.281 [0.122–0.442] | 0.412 [0.230–0.563] | +0.131 [+0.029 to +0.217] | 0.006 |
| Forcepsfish | 93 | R | 0.220 [0.104–0.318] | 0.355 [0.221–0.465] | +0.134 [+0.066 to +0.221] | 0.000 |
| Orangespine Unicornfish | 77 | AP50 | 0.101 [0.018–0.209] | 0.103 [0.041–0.212] | +0.002 [-0.050 to +0.085] | 0.861 |
| Orangespine Unicornfish | 77 | R | 0.097 [0.016–0.186] | 0.084 [0.041–0.210] | -0.013 [-0.063 to +0.085] | 0.866 |
| Convict Tang | 33 | AP50 | 0.335 [0.065–0.600] | 0.551 [0.330–0.746] | +0.215 [+0.025 to +0.373] | 0.031 |
| Convict Tang | 33 | R | 0.273 [0.042–0.514] | 0.591 [0.428–0.806] | +0.318 [+0.239 to +0.429] | 0.000 |
| **Focus species (mean)** | 203 | AP50 | 0.239 [0.128–0.357] | 0.355 [0.252–0.456] | +0.116 [+0.049 to +0.176] | 0.000 |
| **Focus species (mean)** | 203 | R | 0.197 [0.105–0.296] | 0.343 [0.265–0.441] | +0.147 [+0.110 to +0.195] | 0.000 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.676 [0.618–0.727] | 0.667 [0.607–0.720] | -0.009 [-0.022 to +0.004] | 0.157 |
| Any fish (class-agnostic) | 4098 | R | 0.586 [0.545–0.629] | 0.591 [0.551–0.633] | +0.004 [-0.007 to +0.016] | 0.509 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.255 [0.227–0.303] | 0.277 [0.252–0.324] | +0.023 [+0.007 to +0.039] | 0.006 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.232 [0.201–0.267] | 0.272 [0.242–0.305] | +0.039 [+0.027 to +0.053] | 0.000 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.281 | 0.412 | +0.131 |
| Forcepsfish | AP50-95 | 0.157 | 0.263 | +0.106 |
| Forcepsfish | P | 0.524 | 0.531 | +0.007 |
| Forcepsfish | R | 0.220 | 0.355 | +0.134 |
| Forcepsfish | F1 | 0.310 | 0.425 | +0.116 |
| Orangespine Unicornfish | AP50 | 0.101 | 0.103 | +0.002 |
| Orangespine Unicornfish | AP50-95 | 0.071 | 0.069 | -0.002 |
| Orangespine Unicornfish | P | 0.265 | 0.263 | -0.002 |
| Orangespine Unicornfish | R | 0.097 | 0.084 | -0.013 |
| Orangespine Unicornfish | F1 | 0.142 | 0.128 | -0.015 |
| Convict Tang | AP50 | 0.335 | 0.551 | +0.215 |
| Convict Tang | AP50-95 | 0.280 | 0.439 | +0.159 |
| Convict Tang | P | 0.627 | 0.438 | -0.189 |
| Convict Tang | R | 0.273 | 0.591 | +0.318 |
| Convict Tang | F1 | 0.379 | 0.500 | +0.121 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.4965 | 0.4632 | 0.2708 | 0.3333 |
| 0 | Orangespine Unicornfish | 6 | 0.3176 | 0.2824 | 0.25 | 0.25 |
| 0 | Convict Tang | 5 | 0.7938 | 0.9257 | 0.7 | 1.0 |
| 1 | Forcepsfish | 14 | 0.2686 | 0.3593 | 0.2143 | 0.3214 |
| 1 | Orangespine Unicornfish | 15 | 0.0359 | 0.1046 | 0.0333 | 0.0667 |
| 1 | Convict Tang | 3 | 0.7986 | 0.7005 | 0.6667 | 0.8333 |
| 2 | Forcepsfish | 16 | 0.3122 | 0.384 | 0.2188 | 0.2812 |
| 2 | Orangespine Unicornfish | 5 | 0.2523 | 0.3462 | 0.3 | 0.4 |
| 2 | Convict Tang | 9 | 0.6469 | 0.4971 | 0.3333 | 0.5556 |
| 3 | Forcepsfish | 16 | 0.32 | 0.4415 | 0.375 | 0.4688 |
| 3 | Orangespine Unicornfish | 21 | 0.0168 | 0.0295 | 0.0 | 0.0238 |
| 3 | Convict Tang | 7 | 0.0233 | 0.482 | 0.0 | 0.3571 |
| 4 | Forcepsfish | 23 | 0.1938 | 0.5128 | 0.0652 | 0.3696 |
| 4 | Orangespine Unicornfish | 30 | 0.2387 | 0.1558 | 0.1333 | 0.05 |
| 4 | Convict Tang | 9 | 0.179 | 0.5851 | 0.0556 | 0.5 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
