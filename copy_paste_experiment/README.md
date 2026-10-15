# Experiment code

Trains and evaluates the 120 runs reported in the thesis on QUT's Aqua HPC (PBS Pro, one NVIDIA H100 per run):
6 YOLO26 configurations × 2 conditions (baseline, targeted copy-paste) × 5 cross-validation folds × 2 seeds.

| File | Purpose |
|---|---|
| `config.sh` | Paths shared by all job scripts: `IMAGES_DIR` (frames), `VENV`, W&B project |
| `cv_select_donors.py` | Lists donor candidates for Orangespine Unicornfish and Convict Tang (shorter side ≥30 px, ≥3 px from the frame edge, mask fill ≥0.45, no overlap with another fish) → `cv_donor_candidates.csv` |
| `cv_donor_veto.txt` | The 7 candidates excluded at the visual quality check (see [below](#donor-and-paste-frame-selection)) |
| `experiment_donors.csv` | The 23 Forcepsfish donors from the visual review. Its 13 Moorish Idol rows are from an earlier stage and are not read by the code |
| `approved_acceptor_frames.csv` | Camera view of each frame (1 = reef view, 0 = pole / open-water view), used to paste donors only into frames of the same view |
| `cv_prepare.py` | Builds the 5 temporal-block folds (train / val / test, 15-frame buffer) and, per fold, 150 synthetic copy-paste images per focus species from that fold's own donors |
| `prepare_experiment.py` | Helper functions used by `cv_prepare.py`: label reading, random transforms, placement, edge-only Poisson blending |
| `cv_check_same_det.py` | Checks that the data built with segmentation labels is identical, for detection, to the first build |
| `train_experiment.py` | Trains one run (task, condition, fold, seed) and saves its test-fold predictions |
| `cv_evaluate.py` | Pools the 5 test folds, computes COCO AP (box or mask), precision/recall and the paired block-bootstrap CIs; writes the tables in `../results/` |
| `extract_for_analysis.py` | Exports predictions, ground truth and run metadata for `../supplementary_analysis/` |
| `job_*.pbs`, `submit_cv*.sh` | PBS job scripts and the three submit scripts below |
| `labels_export/` | The YOLO-seg labels as exported from CVAT (original file names), read by the code |

## Donor and paste-frame selection

All of the selection below was finished **before any cross-validation run**. It was based only on how each fish or
frame looks, never on model results, and was applied in the same way to every fold, model and seed. The baseline
condition does not use it.

**Donor candidates (Orangespine Unicornfish, Convict Tang).** `cv_select_donors.py` lists every labelled instance of
the species that is:

- at least 30 px on the shorter side of its box,
- at least 3 px from every frame edge,
- filling at least 0.45 of its box with its mask,
- not overlapping any other labelled fish.

This gives 30 Orangespine Unicornfish and 23 Convict Tang candidates (`cv_donor_candidates.csv`).

**Visual quality check.** The candidates were viewed as numbered crops with their mask outlines. 7 of the 53 were
excluded (3 Orangespine Unicornfish, 4 Convict Tang; listed in `cv_donor_veto.txt`), and the other 46 were kept.

**Forcepsfish donors.** These came from an earlier visual review of a gallery of labelled Forcepsfish (shorter side
≥ 20 px). 23 were kept (`experiment_donors.csv`).

**Per fold.** A donor is used in fold *k* only if its frame is a training frame of fold *k*, or lies outside frames
1000–2461. No validation or test fish is ever pasted. Usable donors per fold (folds 1–5):

| Species | Donors | Encounters | Usable per fold |
|---|---|---|---|
| Forcepsfish | 23 | ~8 | 13 / 19 / 20 / 22 / 14 |
| Orangespine Unicornfish | 27 | ~8 | 23 / 16 / 19 / 8 / 12 |
| Convict Tang | 19 | ~7 | 13 / 14 / 14 / 13 / 4 |

**Paste frames.** These are not chosen by hand. Each synthetic image uses a training frame drawn at random from the
fold's training frames with the same camera view as the donor:

- The camera pans slowly between a reef view and a paler view of the mooring pole and open water.
- One frame per minute of footage (61 samples) was labelled as reef or pole view.
- A frame counts as reef view only if the samples on both sides of it are reef view. Frames in minutes where the
  camera was turning, and frames before 1000, count as pole view.
- 871 of the 1,817 frames from 1000–2816 are reef view (`approved_acceptor_frames.csv`).

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
