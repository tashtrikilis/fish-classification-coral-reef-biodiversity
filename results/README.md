# Cross-validation results

Outputs of `copy_paste_experiment/cv_evaluate.py` for every model configuration. All metrics come from the pooled
out-of-fold test predictions (each frame scored by the model that never saw it), with 95% paired block-bootstrap CIs
(2,000 resamples of the 49 temporal blocks).

| Folder | Configuration | Notes |
|---|---|---|
| `cv_results_round4/` | Detection M, seed 0 | exploratory |
| `cv_results_round5/cv_results_seeds1/` | Detection M, seed 1 | |
| `cv_results_round5/cv_results_seeds0-1/` | Detection M, seeds 0+1 averaged | **reported** |
| `cv_results_round6/cv_results_<setup>_<box\|mask>_seeds0-1/` | Detection S/L, Segmentation S/M/L | **reported** (box AP for detection, mask AP for segmentation) |
| `cv_results_round6/..._seeds0/`, `..._seeds1/` | the same, one seed at a time | |

Each folder holds:

- `cv_results.md`: summary tables (focus species, any-fish, macro mAP50, per species);
- `per_class_pooled.csv`: AP50, AP50-95, P, R and F1 for every class;
- `per_fold.csv`: the same per fold;
- figures.

Holm–Bonferroni adjusted p-values for the six configurations are given in the main README and the report.
