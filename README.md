# Fish Classification for Coral Reef Biodiversity Monitoring

Code, results and data for a QUT EGH490 research project (2026) on detecting and segmenting reef fish species
in fixed underwater camera footage from Kona, Hawaiʻi, using [Ultralytics YOLO26](https://docs.ultralytics.com/).

**Research questions**

1. How do model architecture (object detection vs instance segmentation) and size (small / medium / large) affect
   species-level detection and classification?
2. Does **targeted copy-paste augmentation** improve recognition of under-represented species, and at what cost to
   other species?

---

## Contents

- [Dataset](#dataset)
- [Experimental design](#experimental-design)
- [Main results](#main-results)
- [Repository structure](#repository-structure)
- [Reproducing the results](#reproducing-the-results)
- [Earlier work in this repository](#earlier-work-in-this-repository)
- [References](#references)
- [Licence](#licence)

## Dataset

**Download:** [`kona_reef_fish_dataset_v1.zip`](https://github.com/tashtrikilis/fish-classification-coral-reef-biodiversity/releases)
(GitHub Releases, about 240 MB). It's too large to store in the repository itself.

- **1,462 consecutive frames** (frames 1000–2461, 1280 × 720) from a fixed reef camera.
- **4,098 instance masks** across **40 species** (58 classes in the label list), re-annotated by hand in CVAT. The original
  Roboflow bounding-box labels contained species errors and missing fish, so they weren't used.
- Highly imbalanced: Yellow Tang is 27% of all instances, the top three species are 63%, and 20 of the 40 species have
  fewer than 20 instances.
- Labels are in YOLO-seg format (`labels/frame_<N>.txt`), with `data.yaml`, per-class counts, the fold of every frame
  (`folds.csv`) and the exact train / val / test / buffer image lists for each fold (`splits/`).

See the README inside the zip for the full description, the three corrected class names, and the licence
(images CC0 from the Coral Vision "Kona, Hawaii" dataset; annotations CC BY 4.0).

## Experimental design

| | |
|---|---|
| **Models** | YOLO26 detection and instance segmentation, sizes S / M / L (6 configurations), COCO-pretrained, `imgsz 1280` |
| **Conditions** | Baseline vs targeted copy-paste augmentation (architecture and settings identical) |
| **Focus species** | Forcepsfish (93 instances), Orangespine Unicornfish (77), Convict Tang (33) |
| **Evaluation** | 5-fold temporal block cross-validation: 49 blocks of 30 frames, a 15-frame buffer around test frames, validation used only for epoch selection. Every frame is tested once |
| **Seeds** | 2 per configuration |
| **Runs** | 6 configurations × 2 conditions × 5 folds × 2 seeds = **120 training runs** (84.6 H100 GPU-hours) |
| **Statistics** | Paired block bootstrap (2,000 resamples of the 49 blocks), Holm–Bonferroni across the 6 configurations |

**Copy-paste augmentation.** Donor fish are cut out with their masks, using only donors from each fold's own training
frames. They are randomly rotated (±20°) and flipped, shrunk to sizes drawn from that species' real training sizes,
pasted into training frames from the same camera view without overlapping existing fish, and blended with edge-only
Poisson blending. 150 synthetic images per focus species per fold were added to training.

## Main results

Focus-species mean AP50 (pooled out-of-fold test predictions, seeds averaged; detection box IoU, segmentation mask IoU):

| Model | Baseline | Augmented | Change [95% CI] | Holm p |
|---|---|---|---|---|
| Detection S | 0.267 | 0.357 | +0.090 [+0.026, +0.150] | 0.027 |
| Detection M | 0.335 | 0.424 | +0.089 [+0.041, +0.144] | 0.004 |
| Detection L | 0.373 | 0.440 | +0.068 [+0.015, +0.138] | 0.027 |
| Segmentation S | 0.239 | 0.355 | +0.116 [+0.049, +0.176] | < 0.006 |
| Segmentation M | 0.308 | 0.458 | +0.149 [+0.099, +0.217] | < 0.006 |
| Segmentation L | 0.361 | 0.444 | +0.084 [+0.020, +0.148] | 0.027 |

- **RQ1:** class-agnostic fish detection was stable across models (AP50 0.67–0.69), but species-level macro mAP50 was
  much lower (0.25–0.31). Larger models helped only modestly (e.g. Detection M 0.302 vs L 0.306), and segmentation had
  no consistent advantage over detection.
- **RQ2:** augmentation improved the focus species in all six configurations, all significant after Holm correction.
  The gains differed by species: Convict Tang 0.414 → 0.580 (improved in 12/12 runs), Forcepsfish 0.358 → 0.460,
  Orangespine Unicornfish 0.170 → 0.199.
- **Other species:** non-target AP50 and any-fish AP50 changed little (all changes within ±0.026).
- **Efficiency:** augmented Detection M (0.424) outperformed baseline Detection L (0.373) with 80% of the compute.
  Augmentation changes only the training data, so inference cost is unchanged.

Full tables are in [`results/`](results/) and [`supplementary_analysis/`](supplementary_analysis/).

## Repository structure

```
├── copy_paste_experiment/    the final experiment (cross-validation + copy-paste), run on QUT Aqua
│   ├── cv_select_donors.py        auto-select donor candidates (shorter side ≥30 px, fill ≥0.45, unoccluded)
│   ├── cv_prepare.py              build the 5 folds, buffers and per-fold synthetic copy-paste images
│   ├── train_experiment.py        train one run (task, arm, fold, seed); saves test-fold predictions
│   ├── cv_evaluate.py             pool test folds, COCO AP, paired block bootstrap, per-class / per-fold tables
│   ├── extract_for_analysis.py    export predictions + ground truth for the supplementary analysis
│   ├── prepare_experiment.py      shared helpers (label reading, transforms, edge-only Poisson paste)
│   ├── submit_cv*.sh, job_*.pbs   PBS job scripts (rounds 4–6; the 120 runs)
│   ├── cv_confirmation_plan.md    analysis plan written before the round-5 runs
│   ├── cv_grid_plan.md            analysis plan written before the round-6 runs
│   ├── experiment_donors.csv, cv_donor_candidates.csv, cv_donor_veto.txt   donor lists
│   ├── approved_acceptor_frames.csv   frames allowed as paste targets (camera view)
│   ├── labels_export/             the YOLO-seg labels used by the code (original export file names)
│   └── qa_preview/                donor galleries and synthetic-image checks
├── results/                  cross-validation outputs (cv_results.md, per-class and per-fold CSVs, figures)
│   ├── cv_results_round4/         Detection M, seed 0
│   ├── cv_results_round5/         Detection M, seed 1 and seeds 0+1
│   └── cv_results_round6/         Detection S/L and Segmentation S/M/L; box and mask; seeds 0, 1, 0+1
├── supplementary_analysis/   analyses from the saved test predictions (no retraining)
│   ├── scripts/                   bootstrap engine, error analysis, P/R/F1, micro AP, confidence, size, trade-offs
│   ├── tables/  figures/  latex/  tables and figures used in the report appendices
│   └── post_hoc_analyses.pdf      write-up of the supplementary analyses
├── report_figures/           scripts and images for the report figures
│   ├── annotation_fix/            original vs corrected labels (finds species mismatches)
│   ├── cp_steps/  blend_compare/  copy-paste pipeline and blending figures
│   ├── box_vs_mask/               detection vs segmentation example
│   ├── scripts/                   class distribution and forest-plot scripts
│   ├── figures/                   final figure files
│   └── video/                     baseline vs copy-paste test detections (mp4 + script)
├── presentation/             notebooks that make every presentation figure (light / dark themes)
├── pipeline/  analysis/  copy_paste_review/  data/  legacy/  docs/   earlier work (see below)
└── requirements.txt
```

## Reproducing the results

```bash
git clone https://github.com/tashtrikilis/fish-classification-coral-reef-biodiversity.git
cd fish-classification-coral-reef-biodiversity
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Versions used: Ultralytics 8.4.126, PyTorch 2.13.0, Python 3.13.5 (single NVIDIA H100 per run).

**Training (QUT Aqua / PBS).** Copy `copy_paste_experiment/` to the cluster. Put all the frames from the dataset zip (`images/` plus
`donor_frames/images/`, named `frame_<N>.jpg`) in one folder and point `IMAGES_DIR` in `config.sh` at it. Then set your
details in the `#PBS -M` / `#PBS -P` lines of the job scripts. Then:

```bash
cd copy_paste_experiment
bash submit_cv.sh            # Detection M, seed 0 (round 4)
bash submit_cv_confirm.sh    # Detection M, seed 1 (round 5)
bash submit_cv_grid.sh       # the other 5 configurations, seeds 0 and 1 (round 6)
```

Each submit script queues data preparation, the training jobs and the evaluation. Results are written to
`runs_cv/cv_results_*` (the same layout as [`results/`](results/)).

**Supplementary analysis (no GPU needed).** Download `analysis_bundle_public.tgz` from the
[Releases page](https://github.com/tashtrikilis/fish-classification-coral-reef-biodiversity/releases). It holds the saved
test predictions of all 120 runs, ground truth, fold assignment and run logs. Extract it into
`supplementary_analysis/scripts/`, then from that folder:

```bash
python run_boot.py        # paired block bootstrap for every run -> boot.pkl
python analyse_bc.py      # detection vs segmentation, augmentation vs capacity
python errors.py && python agg_errors.py    # error analysis
python prf.py && python prf_sum.py          # precision / recall / F1
python micro.py && python micro_sum.py      # micro-averaged AP
python figs.py            # supplementary figures
```

**Presentation figures.** The notebooks in [`presentation/`](presentation/) hold their numbers inline and run anywhere.
`presentation_figures.ipynb` also needs the images in `report_figures/` and the dataset's `images/` folder extracted to
`dataset/images/`.

## Earlier work in this repository

These folders come from the earlier stages of the project and are kept for reference. They were not used for the
final results:

- `pipeline/`, `analysis/`: the first dataset pipeline (2,817 frames, block-stratified single split) and the per-class
  diagnostics (instance count, size, family, crowding).
- `copy_paste_review/`: choosing species, blend method, donors and paste targets.
- `copy_paste_experiment/` rounds 1–3 (`submit_all.sh`, `submit_round2.sh`, `submit_hires.sh`, `compare_results.py`):
  single-split experiments that informed the cross-validation design.
- `data/`: the earlier CVAT and YOLO label exports (all 2,817 frames, original file names).
- `legacy/`: v1/v2 notebooks and job scripts.

## References

- Ultralytics YOLO26: https://docs.ultralytics.com/
- Ghiasi, G. et al. (2021). Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation. *CVPR*.
- Pérez, P., Gangnet, M., & Blake, A. (2003). Poisson Image Editing. *ACM SIGGRAPH*.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *Annals of Statistics*, 17(3).
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2).
- Wille et al. (2026). Are All Marine Species Created Equal? Performance Disparities in Underwater Object Detection. *WACV*.

## Acknowledgements

- **Footage:** the Aqualink / MEGA Lab reef camera at Kona, Hawaiʻi, via the Coral Vision dataset on Roboflow Universe.
- **Supervision:** Dr Scarlett Raine, QUT.
- **Compute:** QUT eResearch, Aqua HPC.

## Licence

Code is released under the [MIT Licence](LICENSE). The dataset release has its own licence terms (see its README).
