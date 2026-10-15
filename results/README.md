# Cross-validation results

Outputs of `copy_paste_experiment/cv_evaluate.py`. All metrics come from the pooled out-of-fold test predictions:
every frame is scored once, by the model whose test fold contains it. CIs are 95% paired block-bootstrap intervals
(2,000 resamples of the 49 temporal blocks).

| Folder | Configuration | AP scored with |
|---|---|---|
| `detect_s/`, `detect_m/`, `detect_l/` | Detection S / M / L | box IoU |
| `segment_{s,m,l}_mask/` | Segmentation S / M / L | mask IoU (reported) |
| `segment_{s,m,l}_box/` | Segmentation S / M / L | box IoU (report Table 2, box columns) |

Each configuration has three subfolders:

- `seeds0-1/`: seeds 0 and 1 averaged (**the results reported**);
- `seed0/`, `seed1/`: each seed alone (used for the per-run comparisons).

Each subfolder contains:

- `cv_results.md`: focus species, macro mAP50, any-fish AP50 and every class, with baseline, augmented, change and 95% CI;
- `per_class_pooled.csv`: AP50, AP50-95, precision, recall and F1 for every class and both conditions;
- `per_fold.csv`: the same per test fold;
- figures produced by the evaluator.

The Holm-adjusted p-values and the other report tables are reproduced in
[`../supplementary_analysis/`](../supplementary_analysis/).
