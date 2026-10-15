# Copy-paste experiment: baseline vs copy-paste (Forcepsfish + Moorish Idol)

Three arms, each trained for detection and segmentation, all on the same train/val/test split:

| Arm | Training images |
|---|---|
| `baseline` | real frames only |
| `copypaste` | real frames + 300 copy-paste images, donors pasted at their real (large) size |
| `copypaste_scaled` | real frames + 300 copy-paste images, each paste shrunk to a size drawn from the real train-split sizes of that species (big and small) |

Each arm is run with **3 seeds (0, 1, 2)** for both `yolo26m` (detection) and `yolo26m-seg`, so 18 runs in total.
Run names are `<task>_<arm>` for seed 0 and `<task>_<arm>_s<seed>` otherwise, e.g. `segment_copypaste_scaled_s2`.
Everything else is identical between arms: split, model, hyperparameters. Ultralytics' own built-in
copy-paste augmentation is off in all runs. The comparison reports mean ± sd across seeds.

## Run it on Aqua

### Round 2 (seeds 1–2 + the size-matched arm) -- if round 1 has already run

Round 1's four runs (`runs/detect_baseline`, `runs/detect_copypaste`, `runs/segment_baseline`,
`runs/segment_copypaste`) are reused as seed 0. **Don't delete `runs/`.**

1. Copy the changed files over the existing folder (this doesn't touch `runs/`):
   `prepare_experiment.py`, `train_experiment.py`, `compare_results.py`, `job_train.pbs`, `submit_round2.sh`, `README.md`.
2. `cd ~/copy_paste_experiment && bash submit_round2.sh`
   This queues prepare → 14 training jobs (H100, in parallel as the queue allows) → compare.
   Prepare rebuilds `data/`; round 1's synthetic images come out byte-identical (checked), so seed 0 stays valid.

### Round 6: model sizes and segmentation (pre-registered)

The round-4/5 experiment repeated for `yolo26s` and `yolo26l` detection and `yolo26s/m/l-seg` segmentation: baseline vs
copy-paste x 5 folds x seeds 0 and 1 for each, so 100 training jobs. Plan and primary tests (five, Holm-corrected) are in
[`cv_grid_plan.md`](cv_grid_plan.md), written before the runs.

- `data_cv2/` is rebuilt with segmentation labels added. `cv_check_same_det.py` stops everything if its detection data
  is not byte-identical to `data_cv/`.
- Segmentation jobs save each prediction's mask IoU with every labelled fish, so mask AP is pooled across folds exactly
  like box AP (`cv_evaluate.py --iou-type mask`).
- Run: copy the round-6 files over, then `bash submit_cv_grid.sh`. Results go to
  `runs_cv/cv_results_<setup>_<box|mask>_seeds0-1/` (primary), plus `_seeds0` and `_seeds1`, and to W&B project
  `egh490-cv-final` (one evaluation run per set-up and IoU type).

### Round 5: confirmation with a second seed (pre-registered)

Round 4 was exploratory, so its result is re-tested with training **seed 1** on the same folds and synthetic images.
The single primary test is fixed in [`cv_confirmation_plan.md`](cv_confirmation_plan.md) *before* the runs:
the focus-species mean AP50 difference (copy-paste minus baseline), seed 1 alone. It's confirmed if its 95% interval is above 0.

Run: copy the round-5 files over, then `bash submit_cv_confirm.sh` (10 training jobs, then evaluation). Everything logs to a
fresh W&B project, `egh490-cv-final`: the seed-1 training runs (group `cv`) and three evaluation runs (seed 1, seeds 0+1, seed 0).
Each evaluation run holds the report figures (`figures/fig_ap50`, `fig_recall`, `fig_all_metrics`, `fig_per_fold`), the tables
(`results/headline`, `per_class_pooled`, `per_fold`), and every difference with its CI and p in the run summary.
The same files are in `runs_cv/cv_results_seeds1/`, `cv_results_seeds0-1/` and `cv_results/`.

### Round 4: 5-fold cross-validation (detection) — Forcepsfish, Orangespine Unicornfish, Convict Tang

Rounds 1–3 used one split, and each split's val/test held only 1–3 individual fish per focus species,
so val and test could disagree completely. Round 4 cross-validates instead: frames 1000–2461 are dealt
in 30-frame blocks into 5 folds (balanced on label counts only). Each fold is the test set once, with a
15-frame buffer between its test frames and the training frames, so **every labelled fish is scored once**,
by a model that never trained on it. Baseline vs copy-paste is compared on the pooled test predictions,
with 95% intervals from a paired block bootstrap.

Copy-paste v2 changes vs rounds 1–3:
- Donors: Forcepsfish = the hand-approved set; Orangespine Unicornfish and Convict Tang = auto-selected
  (`cv_select_donors.py`: ≥30 px, not at the frame edge, fish-shaped mask, no overlapping fish) minus your
  vetoes in `cv_donor_veto.txt`.
- Sizes drawn from each fold's real training fish (shrink only).
- 1–3 pasted fish per synthetic image, 150 images per species per fold.
- View-matched: a donor from the hazy pole/open-water view is pasted into a hazy-view frame, a reef-view
  donor into a reef-view frame.

Run: copy the round-4 files over, then `bash submit_cv.sh` (prepare → 10 training jobs → evaluate).
Results: `runs_cv/cv_results/cv_results.md` and `cv_results.png`, plus `per_class_pooled.csv`
(every class, both arms: the numbers to report per species), the class-agnostic "any fish" AP50 and the
macro mAP50 over classes with ≥20 test labels.

### Round 3: higher resolution (detection only)

Round 2 found Forcepsfish is barely detected, and its test fish are tiny (median 13 px). This round reruns
detection for all three arms × 3 seeds at `imgsz 2560`, which enlarges the 1280×720 frames 2×. Everything else
is unchanged and reuses round 2's `data/`. Output goes to `runs_i2560/`, so `runs/` is untouched.

1. Copy over the changed files: `train_experiment.py`, `compare_results.py`, `job_train.pbs`, `job_compare.pbs`, `submit_hires.sh`, `README.md`.
2. `bash submit_hires.sh` (9 jobs, then compare).
3. **Choose using `runs_i2560/comparison/comparison_val.md`**, then read `comparison.md` (test) once.
   Both include round 2's imgsz-1280 baseline as a reference column, so "did resolution alone help?" is the
   `Δ vs imgsz 1280` column.

### From scratch

1. **Copy this folder to Aqua**, e.g. next to your `aqua_yolo` folder:
   `scp -r copy_paste_experiment your_qut_id@<your Aqua login host>:~/` (the same host you normally ssh into)
2. **Point it at your frames.** Edit `IMAGES_DIR` in `config.sh` to the folder holding `frame_<N>.jpg`.
   It must contain frames **1000–2461** and the Forcepsfish donor frames **163–167 and 939–941**.
   The prepare step stops with a list of missing frames if any are absent.
3. **Log in to W&B once** on the login node: `source ~/yolo-venv/bin/activate && wandb login`
4. **Submit:** `bash submit_all.sh` (round 1: seed 0 of `baseline` and `copypaste`), then `bash submit_round2.sh`
   for the rest. Check progress with `qstat -u your_qut_id`.

If a training job hits walltime, continue it with:
`qsub -v TASK=detect,ARM=baseline,SEED=1,RESUME=1 job_train.pbs` (same TASK/ARM/SEED as the job that stopped),
then rerun the comparison afterwards with `qsub job_compare.pbs`.

**If W&B shows nothing while jobs run**, compute nodes may not have internet. Uncomment
`export WANDB_MODE=offline` in `config.sh`, resubmit, and push the logs from the login node with
`wandb sync wandb/offline-run-*`. All results are also saved locally, so nothing is lost either way.

## Where to look in W&B

Project **`egh490-copypaste-experiment`**. Runs are grouped by model type (`detect` / `segment`),
and each run is tagged with its arm and `seed0`/`seed1`/`seed2`. In the Runs table, group by **Job Type** (= arm) to see the seed means.

- **Per-epoch charts for your species** (validation set): search the workspace for `focus/`.
  - `focus/Forcepsfish/box_mAP50`, `box_R`, `box_P`, `box_F1`, `box_mAP50-95`
  - `focus/Moorish_Idol/...`
  - Segmentation runs also have `mask_...` versions.
  - Put baseline and copy-paste on the same chart by selecting both runs.
- **The final answer is the test set**, in each run's summary under `test/Forcepsfish/...` and `test/Moorish_Idol/...`.
  - Add these as columns in the Runs table to compare runs side by side.
  - `test/others_mean/...` is the mean over every class you didn't augment, to check copy-paste didn't hurt them.
  - `test/all/...` is the overall score.
- **Every class**: the `per_class_test` and `per_class_val` tables in each run.
- **Test confusion matrix and PR curves**: under `test_plots/`.
- **The `comparison` run**: created after all runs finish: mean ± sd per arm, the change from baseline (`delta/...` in its summary), and the chart.
- **Ultralytics' usual overall curves** (losses, `metrics/mAP50(B)` …) are in the same runs as normal.

## Results on disk

- **Each run**, in `runs/<run>/`:
  - `per_class_test.csv` and `per_class_val.csv`, with mask mAP columns added for segmentation.
  - `focus_metrics.json`
  - The usual Ultralytics plots, plus `eval_test/` with the test-set confusion matrix.
- **The comparison**, in `runs/comparison/`:
  - `comparison.md`: the test table (mean ± sd, n runs, Δ vs baseline), ready for the thesis.
  - `comparison.csv`: every split / metric / arm, with each seed's value.
  - `comparison.png`: bars = mean, whiskers = sd, dots = individual seeds.

## What the data step does (for the methods section)

- **Frames**: 1000–2461 from the CVAT YOLO-segmentation export in `labels_export/`.
  - 1462 frames in total: 1206 with fish, 256 labelled background.
- **Split**: 30-frame blocks (about 1 minute of footage), giving 982 train / 240 val / 240 test frames, with split seed 2.
  - Seed 2 was chosen only on label counts, to balance the focus species. It was never chosen on results.
  - Forcepsfish has 56 train / 16 val / 21 test instances.
  - Moorish Idol has 85 train / 29 val / 29 test instances.
- **Leakage guard**:
  - Blocks containing a donor frame, or within 15 frames (about 30 s) of one, are kept in train.
  - So no val/test frame is within 15 frames of any donor frame. The copy-paste arm can't benefit from having seen a test fish.
- **Donors** (`experiment_donors.csv`):
  - The donors you approved in review: 13 Forcepsfish and 13 Moorish Idol.
  - 10 more Forcepsfish from frames outside the dataset range (163–166, 939–941), which are never evaluated.
  - 3 instances in those frames that touch the frame edge were left out (frames 166, 167, 939).
  - Nothing else: Forcepsfish in frames 2448–2449 were in the review gallery and not picked, so they aren't used.
  - Donor polygons are matched to the export by box overlap (IoU 0.94–0.99).
- **Acceptors**: any training frame in an allowed camera view (`approved_acceptor_frames.csv`, so no pole or open-water view): 539 frames, whatever species they contain. About 130 different frames end up used per species. (`--acceptor-same-species` restores the old same-species rule.)
- **Augmentation**:
  - 150 synthetic images per species, one pasted fish each.
  - Donors are reused evenly, about 6–7 times each for Forcepsfish (23 donors) and about 11–12 times each for Moorish Idol (13 donors).
  - Each paste gets a random rotation within ±20° and a 50% chance of a horizontal flip. There's no vertical flip.
  - `copypaste`: no rescaling. Pastes are as big as the donors (median shorter side 63 px Forcepsfish, 82 px Moorish Idol).
  - `copypaste_scaled`: for each paste, a target size is drawn at random from that species' real TRAIN instances
    (shorter box side), and the donor is shrunk to it (`INTER_AREA`; never enlarged). The Poisson ring thins to 1 px on tiny pastes.
    Its size check (`prepare_summary.json` → `size_check`):

    | Shorter side, px | Real train | `copypaste` | `copypaste_scaled` |
    |---|---|---|---|
    | Forcepsfish median (p10–p90) | 31 (15–69) | 63 (32–92) | 31 (15–61) |
    | Moorish Idol median (p10–p90) | 30 (9–88) | 82 (60–132) | 29 (8–80) |

    Sizes come from TRAIN only, so the test set isn't used to design the augmentation.
  - Pastes never overlap a real fish's box.
- **Blend**: Poisson (`cv2.seamlessClone`, normal clone) in a thin ring of about 3 px at the fish's edge, fading to the fish's original pixels inside.
  - Plain Poisson recoloured the whole fish from its background. Bright yellow Forcepsfish went grey-blue, and the Moorish Idol's thin dorsal filament vanished while still being labelled.
  - See `data/qa_synthetic_samples.jpg`. Its outlines are drawn from the written label files, so it also checks that labels line up with the pasted fish.
- **Not changed from your v3 runs**: yolo26m(-seg), imgsz 1280, 150 epochs, patience 50, AutoBatch.
  - `--cls-pw` is 0 (off) here, unlike your v3 runs' 0.5, so that copy-paste is the only difference between arms.

**Caveats to state:**
- 3 seeds per arm gives a rough sd only. Treat a difference smaller than the spread of the seeds as no difference.
- Test fish are smaller than train fish: median 13 px for both species in test (20 of 21 Forcepsfish and 26 of 29 Moorish Idol
  under 20 px), against about 30 px in train. So `copypaste_scaled` matches the training distribution, but still pastes
  larger fish than the test set mostly contains. Val is in between (Forcepsfish 21.5 px, Moorish Idol 14 px).
- Each copy-paste arm sees 300 more images per epoch, so it also gets about 30% more gradient steps.

## `qa_preview/` (from the test run on your Mac)

- `synthetic_samples_with_label_outlines.jpg`: 24 random pastes. The pink outline is read back from the label file that was written for each one.
- `blend_comparison_poisson_vs_edge_vs_matched.jpg`: the same placements blended three ways. This shows why the edge-only Poisson blend is used.
- `new_forcepsfish_donors.jpg`: the 10 extra Forcepsfish donors (frames 163–166, 939–941).

- `qa_synthetic_samples_scaled.jpg`: 24 random size-matched pastes, zoomed in, with label outlines.

The prepare job regenerates `data/qa_synthetic_samples*.jpg` on Aqua. It should look the same, since the split and pastes are seeded.
