#!/usr/bin/env python3
"""
Train a YOLO26 model (Ultralytics) for either bounding-box detection or
instance segmentation, on QUT's Aqua HPC (or locally).

Usage examples
--------------
Detection model, nano, full frame resolution (1280x720 source frames):
    python train_yolo.py --task detect --data data_detect.yaml \
        --model yolo26n.pt --imgsz 1280 --epochs 150 \
        --project runs_detect --name aqua_run1

Segmentation model, nano, full frame resolution:
    python train_yolo.py --task segment --data data_seg.yaml \
        --model yolo26n-seg.pt --imgsz 1280 --epochs 150 \
        --project runs_seg --name aqua_run1

If you omit --model, it defaults to yolo26n.pt / yolo26n-seg.pt based on
--task, so you can just run:
    python train_yolo.py --task detect --data data_detect.yaml
    python train_yolo.py --task segment --data data_seg.yaml

Notes
-----
- `--data` must point to a YOLO-format data.yaml (train/val image paths + class names).
- `--model` is a pretrained checkpoint name (downloaded automatically by
  ultralytics on first use) or a path to your own .pt/.yaml. Use the '-seg'
  variant for --task segment. Requires an up-to-date `ultralytics` package
  (`pip install -U ultralytics`) since YOLO26 is a recent architecture.
- `--imgsz 1280` trains at your source frames' full resolution (1280x720) —
  ultralytics letterboxes the non-square frame to a multiple of stride
  rather than cropping/stretching it. Pass --rect to batch similarly-shaped
  images together instead of square-padding every image (usually faster,
  slightly less augmentation diversity) — worth trying once you've confirmed
  the default works.
- `--batch -1` (the default) uses Ultralytics' AutoBatch: it probes GPU
  memory and picks the largest batch size that fits, which matters more at
  imgsz=1280 than it did at 640. Set an explicit int to override.
- Device selection is automatic (ultralytics uses CUDA if available); on a
  PBS GPU job this will pick up the GPU(s) PBS allocated to the job.
- `--cls-pw` (default 0.0 = off, matches Ultralytics' own default) turns on
  inverse-class-frequency loss weighting to help with the long-tailed class
  distribution -- 0.0=disabled, 1.0=full inverse frequency, values in
  between dampen it. Computed automatically from the training set's actual
  per-class instance counts, no extra data prep needed. Try 0.5 as a
  starting point rather than jumping straight to 1.0 -- with classes that
  only have 1-3 training instances, full inverse-frequency weighting can
  make training noisy/unstable and hurt precision more than it helps
  recall. Does nothing for a class with zero training instances -- it can
  only reweight the loss among classes the model actually saw.
- Confusion matrix / PR curves / etc. are already saved automatically by
  Ultralytics (plots=True is the library default; passed explicitly below
  for clarity) in <project>/<name>/confusion_matrix.png and friends. In
  addition, this script now writes a clean per-class metrics table --
  Class, Images, Instances, P, R, F1, mAP50, mAP50-95 (plus Mask-P/R/F1 for
  segmentation) -- to <project>/<name>/per_class_metrics.csv after
  training, from the same final-validation results Ultralytics already
  computes. That's the file to check whether your well-represented species
  (e.g. Yellow Tang, Brown Surgeonfish) are scoring meaningfully better
  than the long tail, or whether something else is capping every class
  equally.
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

DEFAULT_MODELS = {
    "detect": "yolo26n.pt",
    "segment": "yolo26n-seg.pt",
}


def parse_args():
    p = argparse.ArgumentParser(description="Train a YOLO26 detection or segmentation model.")
    p.add_argument("--task", required=True, choices=["detect", "segment"],
                    help="Which kind of model to train.")
    p.add_argument("--data", required=True, type=str,
                    help="Path to the YOLO data.yaml file.")
    p.add_argument("--model", type=str, default=None,
                    help="Base model to start from. Defaults to yolo26n.pt (detect) or "
                         "yolo26n-seg.pt (segment) if omitted. Can also be a path to a "
                         "local checkpoint/yaml, e.g. to bump up to yolo26s(-seg).pt later.")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=1280,
                    help="Training image size. Default 1280 = full resolution of the "
                         "1280x720 source frames (long side, letterboxed).")
    p.add_argument("--batch", type=float, default=-1,
                    help="Batch size. -1 = Ultralytics AutoBatch (fit to GPU memory). "
                         "Set an explicit int to override.")
    p.add_argument("--rect", action="store_true",
                    help="Rectangular training — batch images by similar aspect ratio "
                         "instead of square-padding every image to imgsz x imgsz.")
    p.add_argument("--workers", type=int, default=8,
                    help="Dataloader workers — keep <= ncpus requested in the PBS job.")
    p.add_argument("--project", type=str, default="runs",
                    help="Top-level output directory for training runs.")
    p.add_argument("--name", type=str, default="exp",
                    help="Name of this specific run (subfolder under --project).")
    p.add_argument("--patience", type=int, default=50,
                    help="Early-stopping patience (epochs with no improvement).")
    p.add_argument("--resume", action="store_true",
                    help="Resume training from the last checkpoint in this run "
                         "(use after a job hits walltime — see checkpointing notes).")
    p.add_argument("--cls-pw", type=float, default=0.0, dest="cls_pw",
                    help="Inverse-class-frequency loss weighting power. "
                         "0.0 = off (default, matches Ultralytics' own default), "
                         "1.0 = full inverse frequency. Try 0.5 first -- see notes above.")
    return p.parse_args()


def write_per_class_csv(results, out_path):
    """Write Ultralytics' built-in per-class summary (results.summary()) to a plain
    CSV using only the stdlib -- deliberately avoids relying on the optional
    `polars` dependency that Ultralytics' own .to_csv() needs, so a missing
    package can't crash the run after 100+ epochs of real training."""
    if results is None:
        print("no results object returned from training -- skipping per_class_metrics.csv")
        return
    try:
        rows = results.summary()
    except Exception as e:
        print(f"could not build per-class summary ({e}) -- skipping per_class_metrics.csv")
        return
    if not rows:
        print("per-class summary was empty -- skipping per_class_metrics.csv")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote per-class metrics for {len(rows)} classes to {out_path}")


def main():
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_path.resolve()}")

    model_name = args.model or DEFAULT_MODELS[args.task]
    model = YOLO(model_name)

    results = model.train(
        data=str(data_path),
        task=args.task,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        rect=args.rect,
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        cls_pw=args.cls_pw,
        plots=True,  # explicit -- this is already Ultralytics' own default, but spelled
                     # out here so confusion_matrix.png etc. are guaranteed regardless of
                     # library defaults changing later. Saved to <project>/<name>/.
        # `device` is left unset so ultralytics auto-selects whatever GPU(s)
        # PBS/CUDA_VISIBLE_DEVICES exposes to the job.
    )

    run_dir = Path(args.project) / args.name
    write_per_class_csv(results, run_dir / "per_class_metrics.csv")
    print(f"confusion matrix and other plots (if generated): {run_dir}/confusion_matrix.png")


if __name__ == "__main__":
    main()
