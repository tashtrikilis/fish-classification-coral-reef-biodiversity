#!/usr/bin/env python3
"""
Trains one run (task, arm, seed) and logs it to W&B, with per-class metrics for the focus species.

    python3 train_experiment.py --task detect --arm baseline
    python3 train_experiment.py --task segment --arm copypaste --seed 1

The arms only differ in their training images. Ultralytics' built-in copy_paste is 0 for both.

Logged to W&B (one run per task/arm):
  - every epoch on val: focus/<species>/box_P, box_R, box_F1, box_mAP50, box_mAP50-95 (+ mask_* for seg)
  - best checkpoint on val and test: test/<species>/..., test/all/..., test/others_mean/...
  - per_class_val / per_class_test tables, confusion matrix, PR curves
Also saved in <project>/<name>/: per_class_val.csv, per_class_test.csv, focus_metrics.json.
--save-preds also writes test_predictions.json, which cv_evaluate.py reads.
"""
import argparse
import csv
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

DEFAULT_MODELS = {"detect": "yolo26m.pt", "segment": "yolo26m-seg.pt"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", required=True, choices=["detect", "segment"])
    p.add_argument("--arm", required=True, choices=["baseline", "copypaste", "copypaste_scaled"])
    p.add_argument("--data-dir", type=Path, default=HERE / "data")
    p.add_argument("--model", default=None, help="Default: yolo26m.pt / yolo26m-seg.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=float, default=-1, help="-1 = AutoBatch")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cls-pw", type=float, default=0.0, dest="cls_pw",
                   help="Inverse-class-frequency loss weighting (your v3 runs used 0.5). Default 0 = off, "
                        "so copy-paste is the only thing that differs between arms.")
    p.add_argument("--project", type=Path, default=HERE / "runs")
    p.add_argument("--name", default=None, help="Default: <task>_<arm>, plus _i<imgsz> when --imgsz isn't 1280 and _s<seed> when --seed isn't 0")
    p.add_argument("--focus", default=None,
                   help="Comma-separated focus species. Default: focus_species from <data-dir>/prepare_summary.json, "
                        "else Forcepsfish,Moorish Idol")
    p.add_argument("--wandb-project", default="egh490-copypaste-experiment")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume from <project>/<name>/weights/last.pt")
    p.add_argument("--fraction", type=float, default=1.0, help="Fraction of train set (smoke tests only).")
    p.add_argument("--save-preds", action="store_true",
                   help="Also save raw TEST predictions (conf>=0.001) to <run>/test_predictions.json, "
                        "for pooling across cross-validation folds (cv_evaluate.py).")
    p.add_argument("--tags", default="", help="Extra W&B tags separated by commas or colons, e.g. cv:fold3 (colons survive qsub -v)")
    p.add_argument("--wandb-group", default=None, help="W&B group (default: the task)")
    return p.parse_args()


# ---- metrics helpers

def _per_class(metric_obj, class_id):
    """(P, R, F1, mAP50, mAP50-95) for one class from an Ultralytics Metric (.box / .seg),
    or None if the class has no instances in this split."""
    idx = list(getattr(metric_obj, "ap_class_index", []))
    if class_id not in idx:
        return None
    p, r, ap50, ap = metric_obj.class_result(idx.index(class_id))
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"P": float(p), "R": float(r), "F1": float(f1), "mAP50": float(ap50), "mAP50-95": float(ap)}


def _parts(metrics):
    """[('box', Metric), ('mask', Metric)] -- mask only for segmentation."""
    out = [("box", metrics.box)]
    if hasattr(metrics, "seg"):
        out.append(("mask", metrics.seg))
    return out


def _instances_per_class(metrics, nc):
    for attr in ("nt_per_class",):
        v = getattr(metrics, attr, None)
        if v is not None and len(v) == nc:
            return [int(x) for x in v]
    return None


def focus_metrics(metrics, focus_ids, names):
    """Nested dict: {species: {"box": {...}, "mask": {...}, "instances": n}, "all": {...}, "others_mean": {...}}"""
    nc = len(names)
    counts = _instances_per_class(metrics, nc)
    out = {}
    for sp, cid in focus_ids.items():
        out[sp] = {"instances": counts[cid] if counts else None}
        for part, m in _parts(metrics):
            out[sp][part] = _per_class(m, cid)
    out["all"], out["others_mean"] = {}, {}
    for part, m in _parts(metrics):
        out["all"][part] = {"P": float(m.mp), "R": float(m.mr), "mAP50": float(m.map50), "mAP50-95": float(m.map)}
        others = [_per_class(m, c) for c in m.ap_class_index if c not in focus_ids.values()]
        others = [o for o in others if o]
        out["others_mean"][part] = ({k: sum(o[k] for o in others) / len(others) for k in others[0]}
                                    if others else None)
    return out


def flatten(prefix, fm):
    flat = {}
    for sp, d in fm.items():
        key_sp = sp.replace(" ", "_")
        for part, vals in d.items():
            if part == "instances":
                if vals is not None:
                    flat[f"{prefix}/{key_sp}/instances"] = vals
                continue
            if vals:
                for k, v in vals.items():
                    flat[f"{prefix}/{key_sp}/{part}_{k}"] = v
    return flat


def write_summary_csv(metrics, path):
    try:
        rows = metrics.summary()
        if hasattr(metrics, "seg"):  # Ultralytics' seg summary has mask P/R/F1 but not mask mAP
            for i, r in enumerate(rows):
                r["Mask-mAP50"] = round(float(metrics.seg.ap50[i]), 5)
                r["Mask-mAP50-95"] = round(float(metrics.seg.ap[i]), 5)
    except Exception as e:  # never let reporting crash a finished training run
        print(f"could not build per-class summary ({e})")
        return []
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} classes)")
    return rows or []


def _mask_ious(pred_polys, pred_boxes, gt_polys, w, h):
    """Mask IoU of each prediction vs each GT instance in the image (any class), polygons in pixels.
    Each pair is only rasterised inside the union of their boxes; pairs that don't overlap get 0.
    Returns one list per prediction (one value per GT)."""
    import cv2
    import numpy as np
    gts = []
    for g in gt_polys:
        x0, y0 = np.floor(g.min(0)).astype(int)
        x1, y1 = np.ceil(g.max(0)).astype(int)
        gts.append((g, max(0, x0), max(0, y0), min(w, x1 + 1), min(h, y1 + 1)))
    out = []
    for poly, (bx0, by0, bx1, by1) in zip(pred_polys, pred_boxes):
        row = []
        if poly is None or len(poly) < 3:
            out.append([0.0] * len(gts))
            continue
        px0, py0 = np.floor(np.minimum(poly.min(0), [bx0, by0])).astype(int)
        px1, py1 = np.ceil(np.maximum(poly.max(0), [bx1, by1])).astype(int)
        px0, py0, px1, py1 = max(0, px0), max(0, py0), min(w, px1 + 1), min(h, py1 + 1)
        for g, gx0, gy0, gx1, gy1 in gts:
            if px1 <= gx0 or gx1 <= px0 or py1 <= gy0 or gy1 <= py0:
                row.append(0.0)
                continue
            ux0, uy0, ux1, uy1 = min(px0, gx0), min(py0, gy0), max(px1, gx1), max(py1, gy1)
            mp = np.zeros((uy1 - uy0, ux1 - ux0), np.uint8)
            mg = np.zeros_like(mp)
            cv2.fillPoly(mp, [np.round(poly - [ux0, uy0]).astype(np.int32)], 1)
            cv2.fillPoly(mg, [np.round(g - [ux0, uy0]).astype(np.int32)], 1)
            inter = int(np.logical_and(mp, mg).sum())
            union = int(np.logical_or(mp, mg).sum())
            row.append(round(inter / union, 4) if union else 0.0)
        out.append(row)
    return out


def save_test_predictions(model, data_yaml, imgsz, out_path, task="detect"):
    """Raw test predictions per image: [[cls, conf, x1, y1, x2, y2], ...] in pixels, conf >= 0.001 and
    NMS IoU 0.7 (same as Ultralytics val). For seg each row also gets the mask IoU against every GT
    instance in the image, in label-file order, so cv_evaluate.py can compute mask AP."""
    import numpy as np
    root, test = None, None
    for line in data_yaml.read_text().splitlines():
        if line.startswith("path:"):
            root = Path(line.split(":", 1)[1].strip())
        elif line.startswith("test:"):
            test = line.split(":", 1)[1].strip()
    test_dir = root / test if not Path(test).is_absolute() else Path(test)
    label_dir = Path(str(test_dir).replace("/images/", "/labels/"))
    preds = {}
    for r in model.predict(source=str(test_dir), imgsz=imgsz, conf=0.001, iou=0.7, max_det=300,
                           stream=True, verbose=False):
        b = r.boxes
        stem = Path(r.path).stem
        rows = []
        if b is not None and len(b):
            xyxy = b.xyxy.cpu().numpy()
            for c, cf, (x1, y1, x2, y2) in zip(b.cls.cpu().numpy(), b.conf.cpu().numpy(), xyxy):
                rows.append([int(c), round(float(cf), 5), round(float(x1), 1), round(float(y1), 1),
                             round(float(x2), 1), round(float(y2), 1)])
        if task == "segment":
            h, w = r.orig_shape
            gt_polys = []
            lf = label_dir / f"{stem}.txt"
            for line in (lf.read_text().splitlines() if lf.exists() else []):
                p = line.split()
                if len(p) >= 7:
                    gt_polys.append(np.array(p[1:], float).reshape(-1, 2) * [w, h])
            if rows:
                polys = list(r.masks.xy) if r.masks is not None else [None] * len(rows)
                ious = _mask_ious(polys, [row[2:6] for row in rows], gt_polys, w, h)
                for row, iou_row in zip(rows, ious):
                    row.append(iou_row)
        preds[stem] = rows
    out_path.write_text(json.dumps(preds))
    print(f"wrote {out_path} ({len(preds)} test images)")


# ---- main

def main():
    args = parse_args()
    from ultralytics import YOLO, settings

    kind = "det" if args.task == "detect" else "seg"
    data_yaml = args.data_dir / f"{kind}_{args.arm}.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found -- run prepare_experiment.py first")
    name = args.name or (f"{args.task}_{args.arm}" + (f"_i{args.imgsz}" if args.imgsz != 1280 else "")
                         + (f"_s{args.seed}" if args.seed else ""))
    args.project = args.project.resolve()  # Ultralytics puts relative project paths under runs/<task>/
    run_dir = args.project / name
    summary_path = args.data_dir / "prepare_summary.json"
    prep = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    model_path = run_dir / "weights" / "last.pt" if args.resume else (args.model or DEFAULT_MODELS[args.task])
    model = YOLO(str(model_path))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    # class names come from the data yaml once training starts; read them now from the yaml
    yaml_names = {}
    for line in data_yaml.read_text().splitlines():
        if line.startswith("  ") and ":" in line:
            k, v = line.strip().split(":", 1)
            if k.isdigit():
                yaml_names[int(k)] = v.strip()
    names = yaml_names or names
    focus_str = args.focus or ",".join(prep.get("focus_species", [])) or "Forcepsfish,Moorish Idol"
    focus = [s.strip() for s in focus_str.split(",") if s.strip()]
    focus_ids = {s: next(k for k, v in names.items() if v == s) for s in focus}

    # ---- W&B: start the run here so it gets our name/group/config. Ultralytics' W&B callback
    # (if it's enabled) logs into this run instead of making a new one.
    wb = None
    if not args.no_wandb:
        try:
            import wandb
            wb = wandb
            builtin_wandb = bool(settings.get("wandb", False))
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=name,
                       group=args.wandb_group or args.task, job_type=args.arm,
                       tags=[args.task, args.arm, f"seed{args.seed}", f"imgsz{args.imgsz}"] + focus
                            + [t.strip() for t in args.tags.replace(":", ",").split(",") if t.strip()],
                       id=(open(run_dir / "wandb_id.txt").read().strip()
                           if args.resume and (run_dir / "wandb_id.txt").exists() else None),
                       resume="allow",
                       config={"task": args.task, "arm": args.arm, "model": str(args.model or DEFAULT_MODELS[args.task]),
                               "epochs": args.epochs, "imgsz": args.imgsz, "seed": args.seed, "cls_pw": args.cls_pw,
                               "focus_species": focus, "data_yaml": str(data_yaml), "prepare": prep})
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "wandb_id.txt").write_text(wandb.run.id)
            for sp in focus:
                k = sp.replace(" ", "_")
                for part in (["box", "mask"] if args.task == "segment" else ["box"]):
                    for met in ("P", "R", "F1", "mAP50", "mAP50-95"):
                        wandb.define_metric(f"focus/{k}/{part}_{met}", step_metric="epoch")
            wandb.define_metric("epoch")
        except Exception as e:
            print(f"W&B disabled ({e}) -- training continues, metrics still saved locally")
            wb, builtin_wandb = None, False
    else:
        builtin_wandb = False

    def on_fit_epoch_end(trainer):
        if wb is None:
            return
        try:
            log = {"epoch": trainer.epoch + 1}
            if not builtin_wandb:  # Ultralytics isn't logging the overall curves, so we do
                log.update({k: float(v) for k, v in trainer.metrics.items()})
                log.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
            fm = focus_metrics(trainer.validator.metrics, focus_ids, names)
            log.update(flatten("focus", {sp: fm[sp] for sp in focus}))
            log.update(flatten("val_summary", {"all": fm["all"], "others_mean": fm["others_mean"]}))
            wb.log(log, step=trainer.epoch + 1)  # same step convention as Ultralytics' own W&B callback
        except Exception as e:
            print(f"(per-class W&B logging skipped this epoch: {e})")

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    # -1 = AutoBatch, 0<x<1 = fraction of GPU memory, >=1 = a whole-number batch size
    batch = int(args.batch) if (args.batch >= 1 or args.batch == -1) else args.batch
    train_kwargs = dict(data=str(data_yaml), task=args.task, epochs=args.epochs, imgsz=args.imgsz,
                        batch=batch, workers=args.workers, project=str(args.project), name=name,
                        exist_ok=True, patience=args.patience, seed=args.seed, deterministic=True,
                        copy_paste=0.0, plots=True, fraction=args.fraction)
    if args.cls_pw:
        train_kwargs["cls_pw"] = args.cls_pw
    if args.resume:
        model.train(resume=True)
    else:
        model.train(**train_kwargs)

    # ---- final evaluation of the best checkpoint on val and test
    if getattr(model, "trainer", None) is not None and getattr(model.trainer, "save_dir", None):
        run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        best = run_dir / "weights" / "last.pt"
    final = YOLO(str(best))
    results = {}
    for split in ("val", "test"):
        m = final.val(data=str(data_yaml), split=split, imgsz=args.imgsz, batch=8, project=str(run_dir),
                      name=f"eval_{split}", exist_ok=True, plots=True)
        rows = write_summary_csv(m, run_dir / f"per_class_{split}.csv")
        fm = focus_metrics(m, focus_ids, names)
        results[split] = fm
        if wb is not None:
            try:
                wb.run.summary.update(flatten(split, fm))
                if rows:
                    wb.log({f"per_class_{split}": wb.Table(columns=list(rows[0].keys()),
                                                           data=[list(r.values()) for r in rows])})
                if split == "test":
                    imgs = sorted((run_dir / "eval_test").glob("*.png"))
                    wb.log({f"test_plots/{p.stem}": wb.Image(str(p)) for p in imgs})
            except Exception as e:
                print(f"(W&B final logging problem: {e})")

    if args.save_preds:
        save_test_predictions(final, data_yaml, args.imgsz, run_dir / "test_predictions.json", args.task)

    out = {"task": args.task, "arm": args.arm, "seed": args.seed, "imgsz": args.imgsz, "name": name, "best_weights": str(best),
           "focus_species": focus, "metrics": results, "prepare": prep}
    (run_dir / "focus_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {run_dir / 'focus_metrics.json'}")
    for sp in focus:
        t = results["test"][sp]
        print(f"TEST {sp}: instances={t['instances']} box={t['box']}" + (f" mask={t.get('mask')}" if args.task == "segment" else ""))
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
