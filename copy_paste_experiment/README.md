# Copy-paste experiment: baseline vs copy-paste (Forcepsfish + Moorish Idol)

Four training runs on Aqua, all on the same train/val/test split:

| Run | Model | Training images |
|---|---|---|
| `detect_baseline` | yolo26m (detection) | real frames only |
| `detect_copypaste` | yolo26m (detection) | real frames + 300 copy-paste images |
| `segment_baseline` | yolo26m-seg | real frames only |
| `segment_copypaste` | yolo26m-seg | real frames + 300 copy-paste images |

Everything else is identical between baseline and copy-paste: split, model, hyperparameters, seed.
Ultralytics' own built-in copy-paste augmentation is off in all four runs.

## Run it on Aqua

1. **Copy this folder to Aqua**, e.g. next to your `aqua_yolo` folder:
   `scp -r copy_paste_experiment your_qut_id@<your Aqua login host>:~/` (the same host you normally ssh into)
2. **Point it at your frames.** Edit `IMAGES_DIR` in `config.sh` to the folder holding `frame_<N>.jpg`.
   It must contain frames **1000–2461** and the Forcepsfish donor frames **163–167 and 939–941**.
   The prepare step stops with a list of missing frames if any are absent.
3. **Log in to W&B once** on the login node: `source ~/yolo-venv/bin/activate && wandb login`
4. **Submit everything:** `cd ~/copy_paste_experiment && bash submit_all.sh`
   This downloads the pretrained weights (on the login node), then queues:
   prepare (CPU, a few minutes) → the 4 training jobs in parallel (H100, up to 24 h each) → compare.
   Check progress with `qstat -u $USER`.

If a training job hits walltime, continue it with:
`qsub -v TASK=detect,ARM=baseline,RESUME=1 job_train.pbs` (same TASK/ARM as the job that stopped),
then rerun the comparison afterwards with `qsub job_compare.pbs`.

**If W&B shows nothing while jobs run**, compute nodes may not have internet. Uncomment
`export WANDB_MODE=offline` in `config.sh`, resubmit, and push the logs from the login node with
`wandb sync wandb/offline-run-*`. All results are also saved locally, so nothing is lost either way.

## Where to look in W&B

Project **`egh490-copypaste-experiment`**. Runs are grouped by model type (`detect` / `segment`),
and the run names are `detect_baseline`, `detect_copypaste`, etc.

- **Per-epoch charts for your species** (validation set): search the workspace for `focus/`.
  - `focus/Forcepsfish/box_mAP50`, `box_R`, `box_P`, `box_F1`, `box_mAP50-95`
  - `focus/Moorish_Idol/...`
  - Segmentation runs also have `mask_...` versions.
  - Put baseline and copy-paste on the same chart by selecting both runs.
- **The final answer is the test set**, in each run's summary under `test/Forcepsfish/...` and `test/Moorish_Idol/...`.
  - Add these as columns in the Runs table to compare all four runs side by side.
  - `test/others_mean/...` is the mean over every class you didn't augment, to check copy-paste didn't hurt them.
  - `test/all/...` is the overall score.
- **Every class**: the `per_class_test` and `per_class_val` tables in each run.
- **Test confusion matrix and PR curves**: under `test_plots/`.
- **The `comparison` run**: created after all four finish, with the baseline vs copy-paste table and chart.
- **Ultralytics' usual overall curves** (losses, `metrics/mAP50(B)` …) are in the same runs as normal.

## Results on disk

- **Each run**, in `runs/<run>/`:
  - `per_class_test.csv` and `per_class_val.csv`, with mask mAP columns added for segmentation.
  - `focus_metrics.json`
  - The usual Ultralytics plots, plus `eval_test/` with the test-set confusion matrix.
- **The comparison**, in `runs/comparison/`:
  - `comparison.md`: the test table, ready for the thesis.
  - `comparison.csv`
  - `comparison.png`

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
  - Each paste gets a random rotation within ±20° and a 50% chance of a horizontal flip. There's no vertical flip and no rescaling.
  - Pastes never overlap a real fish's box.
- **Blend**: Poisson (`cv2.seamlessClone`, normal clone) in a thin ring of about 3 px at the fish's edge, fading to the fish's original pixels inside.
  - Plain Poisson recoloured the whole fish from its background. Bright yellow Forcepsfish went grey-blue, and the Moorish Idol's thin dorsal filament vanished while still being labelled.
  - See `data/qa_synthetic_samples.jpg`. Its outlines are drawn from the written label files, so it also checks that labels line up with the pasted fish.
- **Not changed from your v3 runs**: yolo26m(-seg), imgsz 1280, 150 epochs, patience 50, AutoBatch.
  - `--cls-pw` is 0 (off) here, unlike your v3 runs' 0.5, so that copy-paste is the only difference between arms.

**Caveats to state:**
- There is one run per arm, so differences of a few points on about 20 test instances may be noise.
- The copy-paste arm sees 300 more images per epoch, so it also gets about 30% more gradient steps.

## `qa_preview/` (from the test run on your Mac)

- `synthetic_samples_with_label_outlines.jpg`: 24 random pastes. The pink outline is read back from the label file that was written for each one.
- `blend_comparison_poisson_vs_edge_vs_matched.jpg`: the same placements blended three ways. This shows why the edge-only Poisson blend is used.
- `new_forcepsfish_donors.jpg`: the 10 extra Forcepsfish donors (frames 163–166, 939–941).

The prepare job regenerates `data/qa_synthetic_samples.jpg` on Aqua. It should look the same, since the split and pastes are seeded.
