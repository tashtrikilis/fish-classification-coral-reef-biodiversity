# Experiment code

Trains and evaluates the 120 runs reported in the thesis on QUT's Aqua HPC (PBS Pro, one NVIDIA H100 per run):
6 YOLO26 configurations × 2 conditions (baseline, targeted copy-paste) × 5 cross-validation folds × 2 seeds.

| File | Purpose |
|---|---|
| `config.sh` | Paths shared by all job scripts: `IMAGES_DIR` (frames), `VENV`, W&B project |
| `cv_select_donors.py` | Lists donor candidates for Orangespine Unicornfish and Convict Tang (shorter side ≥30 px, ≥3 px from the frame edge, mask fill ≥0.45, no overlap with another fish) → `cv_donor_candidates.csv` |
| `cv_donor_veto.txt` | Candidates removed after visual inspection |
| `experiment_donors.csv` | The hand-picked Forcepsfish donors |
| `approved_acceptor_frames.csv` | Camera view of each frame (reef / pole), used to paste donors only into frames of the same view |
| `cv_prepare.py` | Builds the 5 temporal-block folds (train / val / test, 15-frame buffer) and, per fold, 150 synthetic copy-paste images per focus species from that fold's own donors |
| `prepare_experiment.py` | Helper functions used by `cv_prepare.py`: label reading, random transforms, placement, edge-only Poisson blending |
| `cv_check_same_det.py` | Checks that the data built with segmentation labels is identical, for detection, to the first build |
| `train_experiment.py` | Trains one run (task, condition, fold, seed) and saves its test-fold predictions |
| `cv_evaluate.py` | Pools the 5 test folds, computes COCO AP (box or mask), precision/recall and the paired block-bootstrap CIs; writes the tables in `../results/` |
| `extract_for_analysis.py` | Exports predictions, ground truth and run metadata for `../supplementary_analysis/` |
| `job_*.pbs`, `submit_cv*.sh` | PBS job scripts and the three submit scripts below |
| `labels_export/` | The YOLO-seg labels as exported from CVAT (original file names), read by the code |

## Running

1. Put every frame of the dataset release (`images/` and `donor_frames/images/`) in one folder and set
   `IMAGES_DIR` in `config.sh`. Set `#PBS -M` (email) and `#PBS -P` (project) in the `job_*.pbs` files.
2. From this folder on the Aqua login node:

```bash
bash submit_cv.sh            # step 1: Detection M, seed 0 (builds data_cv/)
bash submit_cv_confirm.sh    # step 2: Detection M, seed 1; evaluates seeds 0, 1 and 0+1
bash submit_cv_grid.sh       # step 3: Detection S/L and Segmentation S/M/L, seeds 0 and 1 (builds data_cv2/)
qsub job_extract_analysis.pbs   # optional: export everything for the supplementary analysis
```

Each submit script chains data preparation → training jobs → evaluation with PBS dependencies. Results are written to
`runs_cv/cv_results_*/` (copied into [`../results/`](../results/)).

Training settings: `imgsz 1280`, up to 150 epochs, patience 50, AutoBatch, Ultralytics defaults otherwise
(built-in `copy_paste` = 0), COCO-pretrained `yolo26{s,m,l}.pt` / `yolo26{s,m,l}-seg.pt`, Ultralytics 8.4.126.
