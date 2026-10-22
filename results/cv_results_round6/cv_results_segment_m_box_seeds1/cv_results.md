# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26m-seg, box AP; seed 1)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference. Recall is at confidence 0.25. Primary endpoint: 'Focus species (mean)' AP50.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.437 [0.225–0.626] | 0.483 [0.275–0.666] | +0.046 [-0.117 to +0.179] | 0.470 |
| Forcepsfish | 93 | R | 0.301 [0.132–0.467] | 0.409 [0.212–0.609] | +0.108 [-0.056 to +0.250] | 0.192 |
| Orangespine Unicornfish | 77 | AP50 | 0.131 [0.020–0.266] | 0.226 [0.096–0.367] | +0.094 [+0.005 to +0.224] | 0.034 |
| Orangespine Unicornfish | 77 | R | 0.169 [0.000–0.303] | 0.234 [0.139–0.429] | +0.065 [-0.052 to +0.259] | 0.374 |
| Convict Tang | 33 | AP50 | 0.373 [0.057–0.664] | 0.696 [0.439–0.859] | +0.323 [+0.141 to +0.490] | 0.003 |
| Convict Tang | 33 | R | 0.364 [0.083–0.667] | 0.727 [0.552–0.900] | +0.364 [+0.227 to +0.500] | 0.003 |
| **Focus species (mean)** | 203 | AP50 | 0.314 [0.189–0.440] | 0.468 [0.351–0.576] | +0.155 [+0.070 to +0.234] | 0.002 |
| **Focus species (mean)** | 203 | R | 0.278 [0.145–0.403] | 0.457 [0.357–0.577] | +0.179 [+0.102 to +0.265] | 0.001 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.683 [0.622–0.737] | 0.698 [0.637–0.750] | +0.016 [-0.004 to +0.032] | 0.102 |
| Any fish (class-agnostic) | 4098 | R | 0.623 [0.579–0.668] | 0.624 [0.581–0.671] | +0.001 [-0.019 to +0.022] | 0.912 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.285 [0.258–0.337] | 0.333 [0.305–0.388] | +0.048 [+0.025 to +0.076] | 0.002 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.282 [0.251–0.329] | 0.310 [0.279–0.356] | +0.028 [+0.006 to +0.052] | 0.017 |

A difference counts as a finding only if its 95% interval excludes 0. The focus-species mean is the single pre-specified test; treat the per-species rows as supporting detail.

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.437 | 0.483 | +0.046 |
| Forcepsfish | AP50-95 | 0.283 | 0.344 | +0.061 |
| Forcepsfish | P | 0.667 | 0.613 | -0.054 |
| Forcepsfish | R | 0.301 | 0.409 | +0.108 |
| Forcepsfish | F1 | 0.415 | 0.490 | +0.076 |
| Orangespine Unicornfish | AP50 | 0.131 | 0.226 | +0.094 |
| Orangespine Unicornfish | AP50-95 | 0.072 | 0.141 | +0.069 |
| Orangespine Unicornfish | P | 0.310 | 0.429 | +0.119 |
| Orangespine Unicornfish | R | 0.169 | 0.234 | +0.065 |
| Orangespine Unicornfish | F1 | 0.218 | 0.303 | +0.084 |
| Convict Tang | AP50 | 0.373 | 0.696 | +0.323 |
| Convict Tang | AP50-95 | 0.309 | 0.582 | +0.273 |
| Convict Tang | P | 0.444 | 0.533 | +0.089 |
| Convict Tang | R | 0.364 | 0.727 | +0.364 |
| Convict Tang | F1 | 0.400 | 0.615 | +0.215 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.5941 | 0.5315 | 0.3333 | 0.2083 |
| 0 | Orangespine Unicornfish | 6 | 0.5683 | 0.4827 | 0.5 | 0.5 |
| 0 | Convict Tang | 5 | 0.8449 | 0.9434 | 0.8 | 1.0 |
| 1 | Forcepsfish | 14 | 0.3445 | 0.4294 | 0.3571 | 0.4286 |
| 1 | Orangespine Unicornfish | 15 | 0.1179 | 0.1729 | 0.0 | 0.1333 |
| 1 | Convict Tang | 3 | 0.8653 | 0.8342 | 1.0 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4889 | 0.5439 | 0.1875 | 0.375 |
| 2 | Orangespine Unicornfish | 5 | 0.0581 | 0.3993 | 0.2 | 0.6 |
| 2 | Convict Tang | 9 | 0.5197 | 0.8127 | 0.4444 | 0.7778 |
| 3 | Forcepsfish | 16 | 0.2749 | 0.5182 | 0.3125 | 0.5625 |
| 3 | Orangespine Unicornfish | 21 | 0.0423 | 0.2906 | 0.0 | 0.1905 |
| 3 | Convict Tang | 7 | 0.0485 | 0.5108 | 0.0 | 0.4286 |
| 4 | Forcepsfish | 23 | 0.565 | 0.612 | 0.3043 | 0.5217 |
| 4 | Orangespine Unicornfish | 30 | 0.2858 | 0.3046 | 0.3 | 0.2 |
| 4 | Convict Tang | 9 | 0.3361 | 0.6146 | 0.1111 | 0.6667 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
