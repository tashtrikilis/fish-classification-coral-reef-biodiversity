# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26l detection, box AP; seeds 0 + 1, averaged)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.379 [0.203–0.548] | 0.526 [0.329–0.685] | +0.147 [+0.062 to +0.229] | 0.001 |
| Forcepsfish | 93 | R | 0.285 [0.151–0.429] | 0.478 [0.306–0.640] | +0.194 [+0.105 to +0.287] | 0.000 |
| Orangespine Unicornfish | 77 | AP50 | 0.267 [0.134–0.385] | 0.216 [0.105–0.368] | -0.051 [-0.128 to +0.082] | 0.340 |
| Orangespine Unicornfish | 77 | R | 0.234 [0.131–0.355] | 0.214 [0.128–0.381] | -0.019 [-0.099 to +0.147] | 0.673 |
| Convict Tang | 33 | AP50 | 0.473 [0.150–0.727] | 0.580 [0.330–0.797] | +0.107 [-0.004 to +0.252] | 0.060 |
| Convict Tang | 33 | R | 0.379 [0.125–0.633] | 0.576 [0.315–0.875] | +0.197 [+0.088 to +0.400] | 0.000 |
| **Focus species (mean)** | 203 | AP50 | 0.373 [0.257–0.480] | 0.440 [0.328–0.558] | +0.068 [+0.015 to +0.138] | 0.011 |
| **Focus species (mean)** | 203 | R | 0.299 [0.205–0.396] | 0.423 [0.317–0.556] | +0.124 [+0.066 to +0.228] | 0.000 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.681 [0.621–0.736] | 0.672 [0.610–0.729] | -0.008 [-0.019 to +0.001] | 0.068 |
| Any fish (class-agnostic) | 4098 | R | 0.615 [0.578–0.657] | 0.601 [0.563–0.641] | -0.015 [-0.027 to -0.002] | 0.017 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.306 [0.278–0.355] | 0.329 [0.297–0.384] | +0.023 [+0.001 to +0.045] | 0.037 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.283 [0.252–0.329] | 0.299 [0.268–0.343] | +0.016 [-0.012 to +0.041] | 0.234 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.379 | 0.526 | +0.147 |
| Forcepsfish | AP50-95 | 0.254 | 0.345 | +0.092 |
| Forcepsfish | P | 0.576 | 0.725 | +0.149 |
| Forcepsfish | R | 0.285 | 0.478 | +0.194 |
| Forcepsfish | F1 | 0.381 | 0.576 | +0.195 |
| Orangespine Unicornfish | AP50 | 0.267 | 0.216 | -0.051 |
| Orangespine Unicornfish | AP50-95 | 0.157 | 0.125 | -0.033 |
| Orangespine Unicornfish | P | 0.596 | 0.493 | -0.103 |
| Orangespine Unicornfish | R | 0.234 | 0.214 | -0.019 |
| Orangespine Unicornfish | F1 | 0.336 | 0.298 | -0.037 |
| Convict Tang | AP50 | 0.473 | 0.580 | +0.107 |
| Convict Tang | AP50-95 | 0.389 | 0.476 | +0.087 |
| Convict Tang | P | 0.716 | 0.559 | -0.157 |
| Convict Tang | R | 0.379 | 0.576 | +0.197 |
| Convict Tang | F1 | 0.495 | 0.563 | +0.067 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.5057 | 0.6006 | 0.2292 | 0.4167 |
| 0 | Orangespine Unicornfish | 6 | 0.4774 | 0.4181 | 0.5 | 0.25 |
| 0 | Convict Tang | 5 | 0.8112 | 0.967 | 0.7 | 1.0 |
| 1 | Forcepsfish | 14 | 0.4232 | 0.403 | 0.4286 | 0.4286 |
| 1 | Orangespine Unicornfish | 15 | 0.1151 | 0.2693 | 0.0333 | 0.2 |
| 1 | Convict Tang | 3 | 0.9038 | 0.8906 | 0.8333 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4359 | 0.4729 | 0.2188 | 0.4375 |
| 2 | Orangespine Unicornfish | 5 | 0.4943 | 0.5012 | 0.5 | 0.6 |
| 2 | Convict Tang | 9 | 0.6247 | 0.5869 | 0.4444 | 0.5556 |
| 3 | Forcepsfish | 16 | 0.3522 | 0.5297 | 0.375 | 0.5625 |
| 3 | Orangespine Unicornfish | 21 | 0.3392 | 0.196 | 0.3095 | 0.2381 |
| 3 | Convict Tang | 7 | 0.0037 | 0.2201 | 0.0 | 0.0714 |
| 4 | Forcepsfish | 23 | 0.4188 | 0.6951 | 0.2391 | 0.5435 |
| 4 | Orangespine Unicornfish | 30 | 0.3282 | 0.2617 | 0.1833 | 0.1333 |
| 4 | Convict Tang | 9 | 0.6023 | 0.6389 | 0.2778 | 0.6111 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
