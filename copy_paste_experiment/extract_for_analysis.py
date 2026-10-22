#!/usr/bin/env python3
"""
Collects what the supplementary analysis needs from the finished CV runs (no retraining or inference).

Reads runs_cv/<run>/ (test_predictions.json, results.csv, args.yaml, weights/best.pt),
data_cv/folds.csv, the fold test labels and the PBS logs in this folder.

Writes analysis_bundle/:
  folds.csv, gt.json     test folds and GT boxes in pixels (same as cv_evaluate.py uses)
  preds/<run>.npz        all saved predictions (cls, conf, xyxy, mask IoU vs GT)
  run_meta.csv           epochs, best epoch, training time, batch, optimiser, checkpoint size
  model_complexity.csv   params and GFLOPs at 640 and 1280 px per architecture (CPU)
  logs/, versions.txt
"""
import csv, json, glob, os, re, shutil, sys
from pathlib import Path
import numpy as np

W_IMG, H_IMG = 1280, 720
here = Path(__file__).resolve().parent
runs = here / "runs_cv"
out = here / "analysis_bundle"
(out / "preds").mkdir(parents=True, exist_ok=True)
(out / "logs").mkdir(exist_ok=True)

# ---- folds and ground truth (same files cv_evaluate.py reads)
data = here / "data_cv"
shutil.copy(data / "folds.csv", out / "folds.csv")
folds = {}
for r in csv.DictReader(open(data / "folds.csv")):
    folds[int(r["frame"])] = (int(r["fold"]), int(r["block"]))
gt = {}
for f, (k, b) in folds.items():
    lf = data / f"fold{k}" / "det_baseline" / "labels" / "test" / f"frame_{f}.txt"
    rows = []
    for line in (lf.read_text().splitlines() if lf.exists() else []):
        p = line.split()
        if len(p) >= 5:
            c, cx, cy, w, h = int(p[0]), *(float(v) for v in p[1:5])
            rows.append([c, round((cx - w / 2) * W_IMG, 2), round((cy - h / 2) * H_IMG, 2),
                         round((cx + w / 2) * W_IMG, 2), round((cy + h / 2) * H_IMG, 2)])
    gt[f] = rows
names = {}
for line in (data / "fold0" / "det_baseline.yaml").read_text().splitlines():
    s = line.strip()
    if line.startswith("  ") and ":" in s and s.split(":", 1)[0].isdigit():
        names[int(s.split(":", 1)[0])] = s.split(":", 1)[1].strip()
json.dump({"gt": {str(k): v for k, v in gt.items()}, "names": names,
           "focus": json.loads((data / "cv_summary.json").read_text())["focus_species"]},
          open(out / "gt.json", "w"))
print(f"ground truth: {len(gt)} test frames, {sum(len(v) for v in gt.values())} fish")

# ---- predictions + per-run metadata
pat = re.compile(r"^(detect|detect_s|detect_l|segment_s|segment_m|segment_l)_(baseline|copypaste)_f(\d)(_s1)?$")
meta, total = [], 0
for d in sorted(runs.iterdir()):
    m = pat.match(d.name)
    pf = d / "test_predictions.json"
    if not m or not pf.exists():
        continue
    cfg, arm, k, s1 = m.groups()
    cfg = "detect_m" if cfg == "detect" else cfg
    seed = 1 if s1 else 0
    P = json.loads(pf.read_text())
    cls, conf, box, frame, miou_flat, miou_len = [], [], [], [], [], []
    for stem, rows in P.items():
        f = int(stem.split("_")[1])
        for r in rows:
            cls.append(r[0]); conf.append(r[1]); box.append(r[2:6]); frame.append(f)
            if len(r) > 6:
                miou_flat.extend(r[6]); miou_len.append(len(r[6]))
    npz = out / "preds" / f"{cfg}_{arm}_f{k}_s{seed}.npz"
    np.savez_compressed(npz, frame=np.array(frame, np.int32), cls=np.array(cls, np.int16),
                        conf=np.array(conf, np.float32), box=np.array(box, np.float32).reshape(-1, 4),
                        miou=np.array(miou_flat, np.float32), miou_len=np.array(miou_len, np.int16))
    total += npz.stat().st_size
    # training log
    epochs = train_s = None
    rc = d / "results.csv"
    if rc.exists():
        rr = list(csv.DictReader(open(rc)))
        rr = [{k.strip(): v for k, v in x.items()} for x in rr]
        epochs = len(rr)
        if rr and "time" in rr[-1]:
            train_s = float(rr[-1]["time"])
    args = {}
    ay = d / "args.yaml"
    if ay.exists():
        for line in ay.read_text().splitlines():
            if ":" in line:
                a, v = line.split(":", 1); args[a.strip()] = v.strip()
    best = d / "weights" / "best.pt"
    meta.append(dict(run=d.name, setup=cfg, arm=arm, fold=k, seed=seed, n_test_images=len(P), n_preds=len(cls),
                     epochs_run=epochs, train_seconds=train_s,
                     batch=args.get("batch"), optimizer=args.get("optimizer"), imgsz=args.get("imgsz"),
                     model=args.get("model"), best_pt_bytes=best.stat().st_size if best.exists() else None))
    print(f"{d.name}: {len(cls)} predictions, {epochs} epochs")
with open(out / "run_meta.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(meta[0].keys())); w.writeheader(); w.writerows(meta)
print(f"{len(meta)} runs packed, {total / 1e6:.0f} MB of predictions")

# ---- PBS logs
for lf in glob.glob(str(here / "*.o[0-9]*")):
    shutil.copy(lf, out / "logs")

# ---- model complexity (one trained checkpoint per architecture so the 58-class head is included)
rows = []
try:
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    import ultralytics
    open(out / "versions.txt", "w").write(f"ultralytics {ultralytics.__version__}\ntorch {torch.__version__}\n")
    for cfg in ("detect_s", "detect_m", "detect_l", "segment_s", "segment_m", "segment_l"):
        run = ("detect" if cfg == "detect_m" else cfg) + "_baseline_f0"
        ck = runs / run / "weights" / "best.pt"
        if not ck.exists():
            continue
        model = YOLO(str(ck)).model.float().eval()
        params = sum(p.numel() for p in model.parameters())
        rows.append(dict(setup=cfg, checkpoint=str(ck.relative_to(here)), params=params,
                         gflops_640=round(get_flops(model, 640), 2), gflops_1280=round(get_flops(model, 1280), 2)))
        print(rows[-1])
except Exception as e:
    print(f"model complexity skipped: {e}")
if rows:
    with open(out / "model_complexity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"done -> {out}")
