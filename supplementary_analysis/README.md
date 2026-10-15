# Analysis of the saved test predictions

Scripts that reproduce the report's tables from the saved test-fold predictions of the 120 runs. No model is retrained.
The resulting tables are in [`tables/`](tables/).

| Report | Script(s) | Table file |
|---|---|---|
| Table 1: focus species (instances, encounters, donors) | `dataset_tables.py` | `table1_focus_species.csv` |
| Table 2: baseline macro mAP50 / any-fish AP50 | from [`../results/`](../results/) (baseline columns) | |
| Table 2: baseline mAP50-95 | `baseline_map5095.py` | `table2_baseline_map50_95.csv` |
| §4.1.1: segmentation vs detection (box) | `report_tables.py` | `baseline_seg_vs_det_box.csv` |
| Table 3: baseline AP50, focus species | `report_tables.py` | `table3_baseline_focus_ap50.csv` |
| Table 4: focus-species mean, change, CI, Holm p | `report_tables.py` | `table4_focus_mean_effect.csv` |
| Table 5: per species, Holm p, runs improved | `report_tables.py` | `table5_species_effect.csv` |
| Table 6: macro, non-target and any-fish changes | `report_tables.py` | `table6_nontarget_anyfish.csv` |
| §6: gain in each configuration-seed run | `report_tables.py` | `focus_gain_per_run.csv` |
| Table 7: computational cost | `cost_table.py` | `table7_cost.csv`, `log_meta.csv` |
| Table 8: augmentation vs model capacity | `augmentation_vs_capacity.py` | `table8_augmentation_vs_capacity.csv` |
| Table 9: precision, recall, F1 | `prf.py` → `prf_sum.py` | `table9_precision_recall_f1.csv` |
| Tables 10–11: recall by size and by encounter | `errors.py` → `agg3.py` | `table10_recall_by_size.csv`, `table11_recall_by_encounter.csv` |
| §4.2.2: error breakdown (missed / wrong species / poorly localised), confusions | `errors.py` → `agg_errors.py`, `agg2.py` | `error_breakdown.csv`, `confusions.csv` |
| Table 12: donor sizes | `dataset_tables.py` | `table12_donor_sizes.csv` |
| Table 13: per-class results, medium detection | `dataset_tables.py` + `../results/detect_m/seeds0-1/per_class_pooled.csv` | `table13_class_characteristics.csv` |
| Table 14: prediction confidence | `errors.py` → `conf_paired.py` | `table14_confidence_correct.csv`, `confidence_false_positives.csv` |

Shared modules: `common.py` (loads the bundle), `engine.py` (COCO AP and the paired block bootstrap, using the same
random generator as the evaluator), `cv_evaluate.py` (matching code from the experiment).

## Running

1. Download `analysis_bundle_public.tgz` from the repository's Releases page. Extract it inside `scripts/`, so you
   have `scripts/analysis_bundle/` with the predictions, ground truth, `folds.csv`, run metadata and job logs.
2. From inside `scripts/`:

```bash
python run_boot.py                  # bootstrap AP for every run -> boot.pkl (CPU, takes a while)
python report_tables.py             # Tables 3-6 and the in-text comparisons
python baseline_map5095.py          # Table 2 mAP50-95
python augmentation_vs_capacity.py  # Table 8
python cost_table.py                # Table 7
python prf.py && python prf_sum.py  # Table 9
python errors.py                    # fish-level outcomes -> fish_level.csv, fp_level.csv
python agg3.py && python agg_errors.py && python agg2.py && python conf_paired.py   # Tables 10, 11, 14, error breakdown
python dataset_tables.py --dataset ../../dataset/kona_reef_fish_dataset --experiment ../../copy_paste_experiment   # Tables 1, 12, 13
```

`dataset_tables.py` needs the dataset zip extracted to `dataset/` at the repository root. The other scripts write their
intermediate files to `scripts/`. Detection is scored with box IoU and segmentation with mask IoU, except the
size, encounter, error and confidence analyses (Tables 10, 11 and 14), which use box IoU for all models.
