# YOLO26 training on QUT Aqua — workflow

## What's in this folder

- `train_yolo.py` — one script, used for both models via `--task detect` or `--task segment`. Defaults to `yolo26n.pt` / `yolo26n-seg.pt` (nano) and `imgsz=1280` (full resolution of your 1280x720 frames).
- `job_detect.pbs` / `job_seg.pbs` — PBS batch job scripts (GPU, H100) for Aqua, calling the script above with those defaults.
- `data_detect.yaml` / `data_seg.yaml` — already filled in with your real 58-species class list (from the CVAT `data.yaml`). Still need `train:`/`val:` pointed at real folders once the split below is done.

## Model / resolution choices

- **YOLO26 nano** (`yolo26n.pt`, `yolo26n-seg.pt`) as the starting point, per your request. Bumping up to `yolo26s(-seg).pt` etc. later is just a `--model` change.
- **imgsz 1280** — your frames are 1280x720, so 1280 trains at full resolution (long side), letterboxed rather than cropped/stretched. `--rect` is available if you want rectangular batching instead of square padding.
- **AutoBatch** (`--batch -1`, the default) — lets Ultralytics pick the largest batch size that fits in the GPU's memory, since 1280px images use a lot more VRAM per image than the usual 640px default.
- YOLO26 is a newer architecture (NMS-free head, no DFL loss) — make sure the venv on Aqua has `pip install -U ultralytics` run at least once, an older pinned version won't know the `yolo26*` checkpoints.

## Live progress with Weights & Biases

Ultralytics has a built-in W&B integration — no code changes needed in `train_yolo.py`. It activates automatically once `wandb` is installed and you're logged in, and reuses the `--project`/`--name` you already pass on the command line as the W&B project/run name.

1. `pip install -U wandb` in the same venv as `ultralytics` (added to the job scripts' setup comments already).
2. **On the login node**, once: `wandb login`, paste the API key from [wandb.ai/authorize](https://wandb.ai/authorize). This writes credentials to `~/.netrc`, so the batch job on the compute node picks them up automatically — you never put the key in a job script.
3. Submit as normal. Metrics (loss curves, mAP, per-class PR curves, sample predictions) stream to your wandb.ai dashboard live as each epoch finishes.

**One thing to verify before relying on this:** GPU compute nodes on some HPC clusters have no outbound internet (only the login node does), which would silently block the live dashboard even though training itself runs fine. Quick check — from a short interactive GPU job (`qsub -I -l select=1:ncpus=2:ngpus=1:mem=8gb -l walltime=00:10:00`):
```bash
curl -sI --max-time 10 https://api.wandb.ai | head -1
```
If that hangs or errors, compute nodes are firewalled. Fallback: add `export WANDB_MODE=offline` in the job script (already there, commented out) — this logs to disk instead of streaming, and from the login node (which does have internet) you periodically run `wandb sync <path-to-run-dir>` — including while the job is still running — to push what's been logged so far, giving you near-live updates instead of truly live ones.

## Confirmed from QUT Aqua docs (Getting Started / Submitting Jobs)

- Login: `ssh <your_qut_username>@aqua.qut.edu.au` (VPN required if off-campus).
- Scheduler is **PBS Pro**, not SLURM. Jobs go through `qsub`.
- Max walltime for a batch job is **48 hours**; interactive jobs max **12 hours**.
- GPU batch jobs use `gpu_id=H100` or `gpu_id=A100`, requested with `ngpus=N` on a whole GPU (not the MIG-sliced interactive queue).
- Monitor with `qstat -u $USER`, full detail with `qstat -fx <jobid>`, live-tail with `qpeek -f -o <jobid>` (stdout) / `qpeek -f -e <jobid>` (stderr).
- eResearch's own advice: run a short **interactive job first** to sanity-check the script/environment before committing to a long batch run.

## Still need before this is submit-ready

1. **"Specific software instructions for Aqua"** page — I don't know Aqua's actual Python module name, or whether they want `module load` + venv, conda, or a container for a CUDA-matched PyTorch + `ultralytics`. The job scripts have a placeholder venv-activation line flagged `# VERIFY`.
2. **"Data transfer to Aqua"** page — for the exact command/steps to copy the dataset + scripts across (Finder-mounted drive vs `rsync`/`scp` vs a data-transfer node).
3. **Dataset split** — `full_dataset/` (on your Mac, combined + renamed to `frame_<N>.jpg`) currently holds all 2817 frames; your labelled window is frame_1000–frame_1771 (772 frames, 698 of them with a CVAT label). Still to do: rename the CVAT labels to match (`frame_<N>.txt`), convert them to bbox format for the detection model (min/max of the polygon, as discussed earlier), and do the **temporal split** (not random) into `images/train`+`images/val` — e.g. an early contiguous block as train and a later block as val, so consecutive near-duplicate frames don't leak across the split.

## Once those are confirmed, the flow is

1. Finish the dataset split (#3 above) and point `data_detect.yaml`/`data_seg.yaml` `train:`/`val:` at the real folders.
2. Transfer everything to Aqua (exact method depends on #2).
3. SSH into Aqua, set up the venv/env once (exact commands depend on #1), `pip install -U ultralytics`.
4. Test with a short interactive GPU job (1 GPU, ~1hr walltime, low epoch count) to confirm the script runs end-to-end at imgsz=1280 before committing to a long run.
5. Submit the real runs: `qsub job_detect.pbs` and `qsub job_seg.pbs`.
6. Monitor with `qstat -u $USER` / `qpeek`.
