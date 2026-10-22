# Confirmation run plan (written 10 Oct 2026, before any seed-1 results exist)

## Why a confirmation run

Round 4 was 5-fold cross-validation with one training seed, comparing baseline and copy-paste v2. It showed copy-paste ahead for all three focus species on both AP50 and recall (6 of 6 point estimates positive). Only two of the six 95% intervals excluded zero, and none survived a correction for six tests.

Round 4 was also exploratory. The species, the copy-paste design and the analysis were all chosen after rounds 1–3, so its result can't double as its own confirmation.

## What is run

The same 10 jobs as round 4, with **training seed 1** instead of 0:

- the same `data_cv/` folds and the same synthetic images
- detection only, yolo26m, imgsz 1280, 150 epochs, patience 50, AutoBatch
- the best epoch chosen on each fold's validation blocks
- every fold's test frames scored once by the model that never saw them

## Primary endpoint (one test, decided now)

**The focus-species mean AP50 difference, copy-paste minus baseline, for seed 1 on its own.**

- The mean is taken over Forcepsfish, Orangespine Unicornfish and Convict Tang.
- It's pooled over the 5 test folds, with a paired block bootstrap: 2000 resamples of the 49 30-frame blocks.
- It's reported in `runs_cv/cv_results_seeds1/cv_results.md`, row "Focus species (mean)", AP50.

**Decision rule:**

- **Confirmed:** the 95% interval lies entirely above 0. Copy-paste v2 improves detection of these under-represented species.
- **Not confirmed:** the interval includes 0. Report round 4 as a consistent but unconfirmed positive trend.
- **Contradicted:** the interval lies entirely below 0. Copy-paste v2 hurts these species.

For reference, round 4's seed 0 value is +0.110 (the mean of +0.115, +0.050 and +0.165).

## Reported either way (secondary, no further decisions hang on them)

- Per-species AP50 and recall differences, for seed 1 alone and for seeds 0 + 1 averaged.
- AP50-95, precision, recall and F1 for each focus species.
- Any-fish (class-agnostic) AP50 and recall, as the cost check.
- Macro mAP50 over the classes with at least 20 test labels.
- Per-fold AP50.

## What won't be done

- No further seeds, folds or settings tried in order to make the primary result significant.
- No change to the primary endpoint after the results are seen.
