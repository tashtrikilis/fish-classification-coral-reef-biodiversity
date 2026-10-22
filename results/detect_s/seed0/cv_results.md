# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26s detection, box AP; seed 0)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.327 [0.138–0.497] | 0.387 [0.201–0.544] | +0.061 [-0.037 to +0.153] | 0.208 |
| Forcepsfish | 93 | R | 0.258 [0.098–0.411] | 0.323 [0.167–0.480] | +0.065 [+0.000 to +0.169] | 0.089 |
| Orangespine Unicornfish | 77 | AP50 | 0.123 [0.028–0.286] | 0.132 [0.051–0.250] | +0.009 [-0.085 to +0.069] | 0.715 |
| Orangespine Unicornfish | 77 | R | 0.143 [0.043–0.394] | 0.130 [0.054–0.318] | -0.013 [-0.113 to +0.048] | 0.934 |
| Convict Tang | 33 | AP50 | 0.440 [0.149–0.727] | 0.644 [0.478–0.826] | +0.204 [-0.018 to +0.448] | 0.075 |
| Convict Tang | 33 | R | 0.394 [0.148–0.682] | 0.636 [0.447–0.895] | +0.242 [+0.000 to +0.533] | 0.079 |
| **Focus species (mean)** | 203 | AP50 | 0.297 [0.167–0.437] | 0.388 [0.297–0.488] | +0.091 [+0.002 to +0.179] | 0.044 |
| **Focus species (mean)** | 203 | R | 0.265 [0.149–0.424] | 0.363 [0.275–0.488] | +0.098 [-0.000 to +0.203] | 0.051 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.667 [0.605–0.721] | 0.672 [0.605–0.729] | +0.004 [-0.012 to +0.024] | 0.508 |
| Any fish (class-agnostic) | 4098 | R | 0.575 [0.530–0.622] | 0.583 [0.543–0.627] | +0.008 [-0.009 to +0.024] | 0.363 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.272 [0.249–0.325] | 0.277 [0.248–0.330] | +0.005 [-0.033 to +0.031] | 0.943 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.252 [0.221–0.298] | 0.249 [0.224–0.288] | -0.003 [-0.029 to +0.023] | 0.872 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.327 | 0.387 | +0.061 |
| Forcepsfish | AP50-95 | 0.181 | 0.245 | +0.064 |
| Forcepsfish | P | 0.558 | 0.492 | -0.066 |
| Forcepsfish | R | 0.258 | 0.323 | +0.065 |
| Forcepsfish | F1 | 0.353 | 0.390 | +0.037 |
| Orangespine Unicornfish | AP50 | 0.123 | 0.132 | +0.009 |
| Orangespine Unicornfish | AP50-95 | 0.072 | 0.079 | +0.007 |
| Orangespine Unicornfish | P | 0.324 | 0.333 | +0.010 |
| Orangespine Unicornfish | R | 0.143 | 0.130 | -0.013 |
| Orangespine Unicornfish | F1 | 0.198 | 0.187 | -0.011 |
| Convict Tang | AP50 | 0.440 | 0.644 | +0.204 |
| Convict Tang | AP50-95 | 0.373 | 0.523 | +0.150 |
| Convict Tang | P | 0.542 | 0.583 | +0.042 |
| Convict Tang | R | 0.394 | 0.636 | +0.242 |
| Convict Tang | F1 | 0.456 | 0.609 | +0.153 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.4004 | 0.4645 | 0.125 | 0.1667 |
| 0 | Orangespine Unicornfish | 6 | 0.5823 | 0.3197 | 0.6667 | 0.3333 |
| 0 | Convict Tang | 5 | 0.8465 | 1.0 | 1.0 | 1.0 |
| 1 | Forcepsfish | 14 | 0.2801 | 0.2641 | 0.2143 | 0.2143 |
| 1 | Orangespine Unicornfish | 15 | 0.1013 | 0.117 | 0.0667 | 0.1333 |
| 1 | Convict Tang | 3 | 0.6683 | 0.601 | 0.3333 | 1.0 |
| 2 | Forcepsfish | 16 | 0.392 | 0.4424 | 0.25 | 0.375 |
| 2 | Orangespine Unicornfish | 5 | 0.2031 | 0.2284 | 0.4 | 0.4 |
| 2 | Convict Tang | 9 | 0.583 | 0.5645 | 0.4444 | 0.3333 |
| 3 | Forcepsfish | 16 | 0.4891 | 0.5344 | 0.5625 | 0.5625 |
| 3 | Orangespine Unicornfish | 21 | 0.0247 | 0.069 | 0.0476 | 0.0476 |
| 3 | Convict Tang | 7 | 0.0013 | 0.8053 | 0.0 | 0.5714 |
| 4 | Forcepsfish | 23 | 0.2995 | 0.4643 | 0.2174 | 0.3478 |
| 4 | Orangespine Unicornfish | 30 | 0.3169 | 0.2878 | 0.1 | 0.1 |
| 4 | Convict Tang | 9 | 0.5109 | 0.6427 | 0.3333 | 0.6667 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
