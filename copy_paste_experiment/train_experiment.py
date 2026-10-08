#!/usr/bin/env python3
"""
Train one arm of the baseline vs copy-paste experiment and log it to Weights & Biases,
with per-class metrics for the focus species (Forcepsfish, Moorish Idol by default).

    python3 train_experiment.py --task detect  --arm baseline
    python3 train_experiment.py --task detect  --arm copypaste
    python3 train_experiment.py --task segment --arm baseline
    python3 train_experiment.py --task segment --arm copypaste

Everything except the training images is identical between the two arms: same split,
same model, same hyperparameters, same seed. Ultralytics' own built-in copy_paste
augmentation is forced OFF in both arms so the only copy-paste is ours.

What ends up in W&B (project --wandb-project, one run per task x arm, grouped by task)
---------------------------------------------------------------------------------
Every epoch (validation set):
    focus/<Species>/box_P, box_R, box_F1, box_mAP50, box_mAP50-95
    focus/<Species>/mask_P, ... (segment runs only)
    plus Ultralytics' usual overall curves (losses, metrics/mAP50(B), ...)
After training, the best checkpoint is evaluated on val AND the held-out TEST split:
    run summary:  test/<Species>/box_mAP50 ... (sortable/comparable in the W&B runs table)
                  test/all/box_mAP50, test/others_mean/box_mAP50 (all non-focus classes,
                  to check copy-paste didn't hurt them)
    tables:       per_class_test, per_class_val (every class, every metric)
    images:       test confusion matrix + PR curves
Locally, in <project>/<name>/:  per_class_val.csv, per_class_test.csv, focus_metrics.json
(compare_results.py reads these, so the comparison works even if W&B was offline).
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
    p.add_argument("--arm", required=True, choices=["baseline", "copypaste"])
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
    p.add_argument("--name", default=None, help="Default: <task>_<arm>")
    p.add_argument("--focus", default="Forcepsfish,Moorish Idol")
    p.add_argument("--wandb-project", default="egh490-copypaste-experiment")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--resume", action="store_true", help="Resume from <project>/<name>/weights/last.pt")
    p.add_argument("--fraction", type=float, default=1.0, help="Fraction of train set (smoke tests only).")
    return p.parse_args()


# --------------------------------------------------------------------------- metrics helpers

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


# --------------------------------------------------------------------------- main

def main():
    args = parse_args()
    from ultralytics import YOLO, settings

    kind = "det" if args.task == "detect" else "seg"
    data_yaml = args.data_dir / f"{kind}_{args.arm}.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found -- run prepare_experiment.py first")
    name = args.name or f"{args.task}_{args.arm}"
    args.project = args.project.resolve()  # Ultralytics nests RELATIVE project paths under runs/<task>/
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
    focus = [s.strip() for s in args.focus.split(",") if s.strip()]
    focus_ids = {s: next(k for k, v in names.items() if v == s) for s in focus}

    # ---- W&B: we start the run ourselves so it has a proper name/group/config; Ultralytics'
    # built-in W&B callback (if enabled in your settings) reuses this run instead of making its own.
    wb = None
    if not args.no_wandb:
        try:
            import wandb
            wb = wandb
            builtin_wandb = bool(settings.get("wandb", False))
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=name,
                       group=args.task, job_type=args.arm, tags=[args.task, args.arm] + focus,
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

    out = {"task": args.task, "arm": args.arm, "name": name, "best_weights": str(best),
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
