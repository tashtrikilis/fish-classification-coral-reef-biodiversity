# Cross-validation: baseline vs copy-paste (mask AP, pooled over 5 test folds, yolo26s-seg, mask AP; seed 1)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.279 [0.117–0.442] | 0.463 [0.246–0.635] | +0.183 [+0.039 to +0.302] | 0.015 |
| Forcepsfish | 93 | R | 0.226 [0.109–0.337] | 0.387 [0.225–0.529] | +0.161 [+0.048 to +0.281] | 0.008 |
| Orangespine Unicornfish | 77 | AP50 | 0.109 [0.017–0.243] | 0.112 [0.042–0.291] | +0.003 [-0.116 to +0.190] | 0.906 |
| Orangespine Unicornfish | 77 | R | 0.104 [0.017–0.180] | 0.091 [0.026–0.280] | -0.013 [-0.113 to +0.158] | 0.970 |
| Convict Tang | 33 | AP50 | 0.401 [0.107–0.673] | 0.623 [0.463–0.774] | +0.221 [+0.028 to +0.443] | 0.028 |
| Convict Tang | 33 | R | 0.273 [0.071–0.556] | 0.727 [0.615–0.895] | +0.455 [+0.333 to +0.560] | 0.000 |
| **Focus species (mean)** | 203 | AP50 | 0.263 [0.135–0.377] | 0.399 [0.309–0.501] | +0.136 [+0.062 to +0.230] | 0.001 |
| **Focus species (mean)** | 203 | R | 0.201 [0.107–0.310] | 0.402 [0.331–0.500] | +0.201 [+0.139 to +0.279] | 0.000 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.681 [0.627–0.730] | 0.672 [0.614–0.724] | -0.009 [-0.027 to +0.008] | 0.297 |
| Any fish (class-agnostic) | 4098 | R | 0.592 [0.552–0.633] | 0.595 [0.553–0.638] | +0.003 [-0.017 to +0.022] | 0.748 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.264 [0.233–0.307] | 0.292 [0.264–0.341] | +0.028 [+0.011 to +0.053] | 0.001 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.239 [0.205–0.273] | 0.284 [0.252–0.319] | +0.045 [+0.026 to +0.068] | 0.000 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.279 | 0.463 | +0.183 |
| Forcepsfish | AP50-95 | 0.153 | 0.301 | +0.149 |
| Forcepsfish | P | 0.477 | 0.545 | +0.068 |
| Forcepsfish | R | 0.226 | 0.387 | +0.161 |
| Forcepsfish | F1 | 0.307 | 0.453 | +0.146 |
| Orangespine Unicornfish | AP50 | 0.109 | 0.112 | +0.003 |
| Orangespine Unicornfish | AP50-95 | 0.076 | 0.072 | -0.004 |
| Orangespine Unicornfish | P | 0.296 | 0.304 | +0.008 |
| Orangespine Unicornfish | R | 0.104 | 0.091 | -0.013 |
| Orangespine Unicornfish | F1 | 0.154 | 0.140 | -0.014 |
| Convict Tang | AP50 | 0.401 | 0.623 | +0.221 |
| Convict Tang | AP50-95 | 0.333 | 0.484 | +0.151 |
| Convict Tang | P | 0.692 | 0.471 | -0.222 |
| Convict Tang | R | 0.273 | 0.727 | +0.455 |
| Convict Tang | F1 | 0.391 | 0.571 | +0.180 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.4722 | 0.4055 | 0.25 | 0.2917 |
| 0 | Orangespine Unicornfish | 6 | 0.2909 | 0.3993 | 0.1667 | 0.3333 |
| 0 | Convict Tang | 5 | 0.8727 | 0.8515 | 0.8 | 1.0 |
| 1 | Forcepsfish | 14 | 0.2311 | 0.3908 | 0.1429 | 0.3571 |
| 1 | Orangespine Unicornfish | 15 | 0.0388 | 0.1771 | 0.0667 | 0.1333 |
| 1 | Convict Tang | 3 | 0.8076 | 0.8076 | 0.6667 | 1.0 |
| 2 | Forcepsfish | 16 | 0.2963 | 0.4178 | 0.25 | 0.25 |
| 2 | Orangespine Unicornfish | 5 | 0.1364 | 0.419 | 0.2 | 0.4 |
| 2 | Convict Tang | 9 | 0.5007 | 0.5731 | 0.2222 | 0.6667 |
| 3 | Forcepsfish | 16 | 0.4011 | 0.5644 | 0.4375 | 0.5625 |
| 3 | Orangespine Unicornfish | 21 | 0.019 | 0.0242 | 0.0 | 0.0476 |
| 3 | Convict Tang | 7 | 0.0334 | 0.4655 | 0.0 | 0.5714 |
| 4 | Forcepsfish | 23 | 0.1798 | 0.634 | 0.087 | 0.4783 |
| 4 | Orangespine Unicornfish | 30 | 0.33 | 0.1066 | 0.1667 | 0.0 |
| 4 | Convict Tang | 9 | 0.3094 | 0.703 | 0.1111 | 0.6667 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
