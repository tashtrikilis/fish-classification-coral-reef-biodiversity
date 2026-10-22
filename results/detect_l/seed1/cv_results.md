# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26l detection, box AP; seed 1)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.365 [0.202–0.528] | 0.538 [0.332–0.709] | +0.173 [+0.053 to +0.285] | 0.009 |
| Forcepsfish | 93 | R | 0.312 [0.164–0.457] | 0.495 [0.302–0.675] | +0.183 [+0.043 to +0.324] | 0.019 |
| Orangespine Unicornfish | 77 | AP50 | 0.258 [0.115–0.438] | 0.226 [0.068–0.433] | -0.032 [-0.165 to +0.123] | 0.591 |
| Orangespine Unicornfish | 77 | R | 0.195 [0.049–0.429] | 0.234 [0.054–0.486] | +0.039 [-0.192 to +0.250] | 0.677 |
| Convict Tang | 33 | AP50 | 0.498 [0.165–0.773] | 0.669 [0.451–0.861] | +0.171 [+0.039 to +0.367] | 0.012 |
| Convict Tang | 33 | R | 0.394 [0.129–0.697] | 0.606 [0.364–0.895] | +0.212 [+0.115 to +0.400] | 0.003 |
| **Focus species (mean)** | 203 | AP50 | 0.374 [0.262–0.489] | 0.478 [0.375–0.595] | +0.104 [+0.026 to +0.194] | 0.017 |
| **Focus species (mean)** | 203 | R | 0.300 [0.206–0.428] | 0.445 [0.349–0.576] | +0.145 [+0.049 to +0.252] | 0.011 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.671 [0.604–0.733] | 0.671 [0.602–0.732] | +0.000 [-0.017 to +0.016] | 0.884 |
| Any fish (class-agnostic) | 4098 | R | 0.621 [0.578–0.666] | 0.607 [0.570–0.646] | -0.014 [-0.036 to +0.008] | 0.197 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.301 [0.270–0.356] | 0.342 [0.303–0.403] | +0.041 [+0.000 to +0.072] | 0.047 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.293 [0.252–0.358] | 0.302 [0.274–0.343] | +0.010 [-0.043 to +0.042] | 0.634 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.365 | 0.538 | +0.173 |
| Forcepsfish | AP50-95 | 0.250 | 0.372 | +0.123 |
| Forcepsfish | P | 0.580 | 0.708 | +0.128 |
| Forcepsfish | R | 0.312 | 0.495 | +0.183 |
| Forcepsfish | F1 | 0.406 | 0.582 | +0.177 |
| Orangespine Unicornfish | AP50 | 0.258 | 0.226 | -0.032 |
| Orangespine Unicornfish | AP50-95 | 0.148 | 0.126 | -0.022 |
| Orangespine Unicornfish | P | 0.536 | 0.486 | -0.049 |
| Orangespine Unicornfish | R | 0.195 | 0.234 | +0.039 |
| Orangespine Unicornfish | F1 | 0.286 | 0.316 | +0.030 |
| Convict Tang | AP50 | 0.498 | 0.669 | +0.171 |
| Convict Tang | AP50-95 | 0.405 | 0.540 | +0.135 |
| Convict Tang | P | 0.765 | 0.690 | -0.075 |
| Convict Tang | R | 0.394 | 0.606 | +0.212 |
| Convict Tang | F1 | 0.520 | 0.645 | +0.125 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.471 | 0.5815 | 0.2917 | 0.2917 |
| 0 | Orangespine Unicornfish | 6 | 0.5503 | 0.3317 | 0.6667 | 0.0 |
| 0 | Convict Tang | 5 | 0.8845 | 1.0 | 0.8 | 1.0 |
| 1 | Forcepsfish | 14 | 0.4099 | 0.3991 | 0.4286 | 0.4286 |
| 1 | Orangespine Unicornfish | 15 | 0.1625 | 0.2427 | 0.0667 | 0.2 |
| 1 | Convict Tang | 3 | 1.0 | 0.9158 | 1.0 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4488 | 0.4639 | 0.1875 | 0.4375 |
| 2 | Orangespine Unicornfish | 5 | 0.2967 | 0.6676 | 0.2 | 0.8 |
| 2 | Convict Tang | 9 | 0.6965 | 0.6144 | 0.4444 | 0.5556 |
| 3 | Forcepsfish | 16 | 0.3191 | 0.5546 | 0.4375 | 0.625 |
| 3 | Orangespine Unicornfish | 21 | 0.4006 | 0.3459 | 0.381 | 0.4762 |
| 3 | Convict Tang | 7 | 0.0074 | 0.398 | 0.0 | 0.1429 |
| 4 | Forcepsfish | 23 | 0.4274 | 0.7658 | 0.2609 | 0.6957 |
| 4 | Orangespine Unicornfish | 30 | 0.2561 | 0.1853 | 0.0333 | 0.0333 |
| 4 | Convict Tang | 9 | 0.6271 | 0.7587 | 0.2222 | 0.6667 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
