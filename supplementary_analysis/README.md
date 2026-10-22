# Supplementary analyses

Analyses run on the **saved test predictions** of the 120 training runs. No model was retrained. They are written up in
`post_hoc_analyses.pdf`, and their tables are used in the report appendices.

| What | Scripts | Outputs |
|---|---|---|
| Bootstrap engine (COCO AP, paired block bootstrap) | `common.py`, `engine.py`, `run_boot.py` | `boot.pkl` |
| Precision / recall / F1 for the focus species | `prf.py`, `prf_sum.py` | `tables/focus_prf_*` |
| Error analysis (missed / wrong species / poorly localised), size and encounters | `errors.py`, `agg_errors.py`, `agg2.py`, `agg3.py` | `tables/err_summary_all.csv`, `size_bins.csv`, `encounters.csv`, `confusions.csv` |
| Confidence of correct detections and false positives | `conf_paired.py` | `tables/conf_*.csv`, `latex/confidence_tables.tex` |
| Detection vs segmentation, augmentation vs model capacity | `analyse_bc.py`, `components.py`, `decomp.py`, `tradeoff_mask.py` | `tables/task_comparison*.csv`, `tradeoff_*.csv` |
| Micro-averaged AP | `micro.py`, `micro_sum.py`, `micro_fig.py` | `tables/micro_*.csv`, `figures/micro_vs_macro.*` |
| Fold sizes and species per fold | `splits.py`, `splits_all.py`, `splits_fig.py`, `splits_all_fig.py` | `tables/cv_splits_*.csv` |
| Computational cost | (`log_meta.csv` from the run logs) | `tables/cost_table.csv` |
| Values used in the presentation | `pres_data.py` | `pres_data.json` |
| Supplementary figures | `figs.py` | `figures/` |

## Running

1. Download `analysis_bundle_public.tgz` from the repository's Releases page and extract it into `scripts/`, giving
   `scripts/analysis_bundle/`. It holds predictions, ground truth, `folds.csv`, run metadata and logs.
2. Run the scripts from inside `scripts/`, starting with `python run_boot.py` (CPU only; it can take a while).
   Most other scripts read `boot.pkl` or the bundle directly.

Notes:

- Detection is scored with box IoU and segmentation with mask IoU, except the error, confidence and size analyses,
  which use box IoU for all models.
- Seeds are averaged inside each bootstrap resample. CIs are 95% percentile intervals and are not adjusted for
  multiple comparisons unless stated.
