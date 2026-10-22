# Cross-validation: baseline vs copy-paste (mask AP, pooled over 5 test folds, yolo26l-seg, mask AP; seeds 0 + 1, averaged)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference (not adjusted for multiple comparisons). Recall is at confidence 0.25.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.394 [0.195–0.562] | 0.510 [0.293–0.682] | +0.116 [+0.032 to +0.215] | 0.008 |
| Forcepsfish | 93 | R | 0.296 [0.140–0.431] | 0.462 [0.293–0.608] | +0.167 [+0.080 to +0.273] | 0.001 |
| Orangespine Unicornfish | 77 | AP50 | 0.225 [0.079–0.329] | 0.244 [0.058–0.388] | +0.019 [-0.072 to +0.085] | 0.743 |
| Orangespine Unicornfish | 77 | R | 0.214 [0.092–0.292] | 0.253 [0.065–0.380] | +0.039 [-0.071 to +0.113] | 0.534 |
| Convict Tang | 33 | AP50 | 0.463 [0.116–0.698] | 0.579 [0.342–0.793] | +0.116 [-0.032 to +0.289] | 0.116 |
| Convict Tang | 33 | R | 0.394 [0.094–0.641] | 0.576 [0.391–0.828] | +0.182 [+0.025 to +0.400] | 0.030 |
| **Focus species (mean)** | 203 | AP50 | 0.361 [0.222–0.470] | 0.444 [0.307–0.556] | +0.084 [+0.020 to +0.148] | 0.009 |
| **Focus species (mean)** | 203 | R | 0.301 [0.176–0.405] | 0.430 [0.307–0.546] | +0.129 [+0.051 to +0.216] | 0.005 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.680 [0.622–0.732] | 0.684 [0.625–0.738] | +0.004 [-0.006 to +0.017] | 0.355 |
| Any fish (class-agnostic) | 4098 | R | 0.617 [0.576–0.661] | 0.628 [0.587–0.670] | +0.011 [-0.006 to +0.027] | 0.192 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.302 [0.271–0.354] | 0.332 [0.297–0.392] | +0.029 [+0.008 to +0.057] | 0.003 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.287 [0.253–0.327] | 0.315 [0.277–0.360] | +0.028 [-0.001 to +0.062] | 0.057 |

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.394 | 0.510 | +0.116 |
| Forcepsfish | AP50-95 | 0.244 | 0.345 | +0.102 |
| Forcepsfish | P | 0.567 | 0.652 | +0.085 |
| Forcepsfish | R | 0.296 | 0.462 | +0.167 |
| Forcepsfish | F1 | 0.389 | 0.541 | +0.152 |
| Orangespine Unicornfish | AP50 | 0.225 | 0.244 | +0.019 |
| Orangespine Unicornfish | AP50-95 | 0.159 | 0.172 | +0.013 |
| Orangespine Unicornfish | P | 0.375 | 0.470 | +0.095 |
| Orangespine Unicornfish | R | 0.214 | 0.253 | +0.039 |
| Orangespine Unicornfish | F1 | 0.272 | 0.329 | +0.057 |
| Convict Tang | AP50 | 0.463 | 0.579 | +0.116 |
| Convict Tang | AP50-95 | 0.380 | 0.485 | +0.105 |
| Convict Tang | P | 0.649 | 0.682 | +0.033 |
| Convict Tang | R | 0.394 | 0.576 | +0.182 |
| Convict Tang | F1 | 0.490 | 0.624 | +0.134 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.4797 | 0.6315 | 0.3125 | 0.3958 |
| 0 | Orangespine Unicornfish | 6 | 0.346 | 0.4197 | 0.3333 | 0.5 |
| 0 | Convict Tang | 5 | 0.8303 | 0.9236 | 0.9 | 0.9 |
| 1 | Forcepsfish | 14 | 0.4201 | 0.4337 | 0.3214 | 0.3929 |
| 1 | Orangespine Unicornfish | 15 | 0.0628 | 0.0399 | 0.0667 | 0.0 |
| 1 | Convict Tang | 3 | 0.5985 | 0.9171 | 0.3333 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4671 | 0.4908 | 0.375 | 0.3438 |
| 2 | Orangespine Unicornfish | 5 | 0.4948 | 0.4193 | 0.4 | 0.4 |
| 2 | Convict Tang | 9 | 0.6742 | 0.5788 | 0.5556 | 0.5 |
| 3 | Forcepsfish | 16 | 0.4182 | 0.5467 | 0.4062 | 0.5938 |
| 3 | Orangespine Unicornfish | 21 | 0.2219 | 0.1786 | 0.1429 | 0.119 |
| 3 | Convict Tang | 7 | 0.0177 | 0.437 | 0.0 | 0.2857 |
| 4 | Forcepsfish | 23 | 0.3658 | 0.6657 | 0.1304 | 0.5652 |
| 4 | Orangespine Unicornfish | 30 | 0.3475 | 0.4066 | 0.2833 | 0.4 |
| 4 | Convict Tang | 9 | 0.482 | 0.517 | 0.2778 | 0.5556 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
