# Cross-validation: baseline vs copy-paste (box AP, pooled over 5 test folds, yolo26l-seg, box AP; seeds 0 + 1, averaged)

Every labelled frame in 1462 frames is scored once, by the fold model that never trained on it. Brackets: 95% interval from a paired block bootstrap (2000 resamples of 49 30-frame blocks). p: two-sided bootstrap p-value for the difference. Recall is at confidence 0.25. Primary endpoint: 'Focus species (mean)' AP50.

| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |
|---|---|---|---|---|---|---|
| Forcepsfish | 93 | AP50 | 0.368 [0.184–0.520] | 0.499 [0.286–0.671] | +0.132 [+0.051 to +0.225] | 0.002 |
| Forcepsfish | 93 | R | 0.280 [0.134–0.407] | 0.457 [0.289–0.605] | +0.177 [+0.097 to +0.278] | 0.000 |
| Orangespine Unicornfish | 77 | AP50 | 0.223 [0.079–0.327] | 0.244 [0.057–0.388] | +0.020 [-0.071 to +0.087] | 0.724 |
| Orangespine Unicornfish | 77 | R | 0.214 [0.092–0.292] | 0.253 [0.065–0.380] | +0.039 [-0.071 to +0.113] | 0.534 |
| Convict Tang | 33 | AP50 | 0.463 [0.116–0.698] | 0.579 [0.342–0.793] | +0.116 [-0.032 to +0.289] | 0.116 |
| Convict Tang | 33 | R | 0.394 [0.094–0.641] | 0.576 [0.391–0.828] | +0.182 [+0.025 to +0.400] | 0.030 |
| **Focus species (mean)** | 203 | AP50 | 0.351 [0.215–0.457] | 0.441 [0.304–0.553] | +0.089 [+0.026 to +0.153] | 0.007 |
| **Focus species (mean)** | 203 | R | 0.296 [0.170–0.398] | 0.429 [0.306–0.544] | +0.133 [+0.056 to +0.217] | 0.004 |
| Any fish (class-agnostic) | 4098 | AP50 | 0.671 [0.616–0.724] | 0.677 [0.620–0.732] | +0.006 [-0.005 to +0.018] | 0.311 |
| Any fish (class-agnostic) | 4098 | R | 0.614 [0.572–0.657] | 0.624 [0.584–0.667] | +0.011 [-0.006 to +0.027] | 0.185 |
| Macro mAP50, 20 classes with >= 20 labels |  | AP50 | 0.299 [0.269–0.350] | 0.328 [0.294–0.387] | +0.029 [+0.008 to +0.057] | 0.004 |
| Macro mAP50, 20 classes with >= 20 labels |  | R | 0.285 [0.253–0.325] | 0.313 [0.276–0.358] | +0.028 [-0.001 to +0.062] | 0.056 |

A difference counts as a finding only if its 95% interval excludes 0. The focus-species mean is the single pre-specified test; treat the per-species rows as supporting detail.

## All metrics, focus species

| Species | Metric | Baseline | Copy-paste | Change |
|---|---|---|---|---|
| Forcepsfish | AP50 | 0.368 | 0.499 | +0.132 |
| Forcepsfish | AP50-95 | 0.229 | 0.336 | +0.108 |
| Forcepsfish | P | 0.536 | 0.645 | +0.109 |
| Forcepsfish | R | 0.280 | 0.457 | +0.177 |
| Forcepsfish | F1 | 0.368 | 0.535 | +0.167 |
| Orangespine Unicornfish | AP50 | 0.223 | 0.244 | +0.020 |
| Orangespine Unicornfish | AP50-95 | 0.137 | 0.165 | +0.027 |
| Orangespine Unicornfish | P | 0.375 | 0.470 | +0.095 |
| Orangespine Unicornfish | R | 0.214 | 0.253 | +0.039 |
| Orangespine Unicornfish | F1 | 0.272 | 0.329 | +0.057 |
| Convict Tang | AP50 | 0.463 | 0.579 | +0.116 |
| Convict Tang | AP50-95 | 0.375 | 0.495 | +0.119 |
| Convict Tang | P | 0.649 | 0.682 | +0.033 |
| Convict Tang | R | 0.394 | 0.576 | +0.182 |
| Convict Tang | F1 | 0.490 | 0.624 | +0.134 |

## Per fold (AP50 / recall)

| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |
|---|---|---|---|---|---|---|
| 0 | Forcepsfish | 24 | 0.418 | 0.5797 | 0.2708 | 0.375 |
| 0 | Orangespine Unicornfish | 6 | 0.346 | 0.4197 | 0.3333 | 0.5 |
| 0 | Convict Tang | 5 | 0.8303 | 0.9236 | 0.9 | 0.9 |
| 1 | Forcepsfish | 14 | 0.4201 | 0.4337 | 0.3214 | 0.3929 |
| 1 | Orangespine Unicornfish | 15 | 0.0527 | 0.0379 | 0.0667 | 0.0 |
| 1 | Convict Tang | 3 | 0.5985 | 0.9171 | 0.3333 | 1.0 |
| 2 | Forcepsfish | 16 | 0.4151 | 0.4908 | 0.3438 | 0.3438 |
| 2 | Orangespine Unicornfish | 5 | 0.4948 | 0.4193 | 0.4 | 0.4 |
| 2 | Convict Tang | 9 | 0.6742 | 0.5788 | 0.5556 | 0.5 |
| 3 | Forcepsfish | 16 | 0.4048 | 0.5442 | 0.4062 | 0.5938 |
| 3 | Orangespine Unicornfish | 21 | 0.2219 | 0.1786 | 0.1429 | 0.119 |
| 3 | Convict Tang | 7 | 0.0177 | 0.437 | 0.0 | 0.2857 |
| 4 | Forcepsfish | 23 | 0.3645 | 0.6597 | 0.1304 | 0.5652 |
| 4 | Orangespine Unicornfish | 30 | 0.3455 | 0.4066 | 0.2833 | 0.4 |
| 4 | Convict Tang | 9 | 0.482 | 0.517 | 0.2778 | 0.5556 |

Reliable classes (>= 20 pooled test labels): Acanthuridae, Black Triggerfish, Blackfin Chromis, Brown Surgeonfish, Convict Tang, Doublebar Goatfish, Forcepsfish, Fourspot Butterflyfish, Goldring Surgeonfish, Lei Triggerfish, Lowfin Chub, Moorish Idol, Multiband Butterflyfish, Orangeband Surgeonfish, Orangespine Unicornfish, Ornate Butterfly Fish, Raccoon Butterflyfish, Redlip Parrotfish - Male, Redlip Parrotfish - Female, Yellow Tang
