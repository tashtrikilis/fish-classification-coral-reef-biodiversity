#!/usr/bin/env python3
"""
Pool the 5 cross-validation test folds and compare baseline vs copy-paste, with confidence intervals.

Each frame in data_cv/ is in exactly one fold's TEST set, so pooling the per-fold test predictions
(runs_cv/detect_<arm>_f<k>/test_predictions.json, written by train_experiment.py --save-preds)
scores every labelled fish once, by the one model that never trained on it.

Metrics (computed here, COCO-style, from the raw predictions):
  AP50, AP50-95          101-point interpolated average precision per class
  P, R, F1 @ conf 0.25   at Ultralytics' default prediction threshold
  any fish AP50          class-agnostic: is there a fish here at all? (cross-class NMS 0.6 first)
  macro mAP50            mean AP50 over the "reliable" classes (>= --min-instances pooled test labels)

Uncertainty: a paired BLOCK bootstrap. The 30-frame blocks (the unit the folds were built from) are
resampled with replacement, the same resample for both arms, --boot times; each metric and the
copy-paste minus baseline difference is recomputed on every resample. Reported: the 95% percentile
interval and a two-sided bootstrap p-value for the difference. Blocks, not frames, are resampled
because neighbouring frames show the same fish. Training randomness (seed) is NOT in this interval;
run a second seed per fold if you need it.

Primary endpoint (fixed before the seed-1 confirmation run): "Focus species (mean)" = the mean over the
focus species of AP50, and its copy-paste minus baseline difference with 95% CI.

Writes to <runs-dir>/cv_results[_seeds...]/:
  cv_results.md          headline table (focus species, focus mean, any fish, macro) + per-fold table
  per_class_pooled.csv   every class, both arms, all metrics
  per_fold.csv           focus species AP50 / recall per fold and arm
  fig_ap50.png           AP50 per arm with 95% CI, and the difference with its 95% CI
  fig_recall.png         the same for recall at conf 0.25
  fig_all_metrics.png    focus species: AP50, AP50-95, P, R, F1, baseline vs copy-paste
  fig_per_fold.png       focus species AP50 per fold, baseline -> copy-paste (marker size = test labels)
  (cv_results.png        = fig_ap50.png, kept for the round-4 file name)

Usage (from copy_paste_experiment/):
    python3 cv_evaluate.py                       # seed 0 (round 4)
    python3 cv_evaluate.py --seeds 1             # the confirmation run on its own (primary)
    python3 cv_evaluate.py --seeds 0,1           # both seeds averaged
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W_IMG, H_IMG = 1280, 720
ARMS = [("baseline", "Baseline"), ("copypaste", "Copy-paste")]
COL = {"baseline": "#2a78d6", "copypaste": "#eb6834"}  # validated pair (light surface)
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
IOUS = np.linspace(0.5, 0.95, 10)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=HERE / "data_cv")
    p.add_argument("--runs-dir", type=Path, default=HERE / "runs_cv")
    p.add_argument("--run-pattern", default="detect_{arm}_f{k}{suffix}",
                   help="{suffix} is '' for seed 0 and '_s<seed>' otherwise (train_experiment.py's naming)")
    p.add_argument("--seeds", default="0", help="Comma-separated training seeds to include, e.g. 1 or 0,1. "
                   "With several seeds each metric is the mean over seeds, inside every bootstrap resample.")
    p.add_argument("--out-name", default=None, help="Output folder under --runs-dir (default cv_results[_seeds...])")
    p.add_argument("--wandb-project", default=None, help="Also log tables, figures and summary to this W&B project")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--iou-type", choices=["box", "mask"], default="box",
                   help="Match predictions to labels by box IoU, or by mask IoU (segmentation runs only)")
    p.add_argument("--label", default="", help="Model description shown in titles, e.g. 'yolo26m-seg, mask'")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--min-instances", type=int, default=20)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def iou_matrix(a, b):
    """a: (n,4) xyxy, b: (m,4) xyxy -> (n,m)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-9)


def match(pred_boxes, pred_conf, gt_boxes, thr, ious=None):
    """Greedy COCO-style matching, highest confidence first. Returns tp flags (n_pred,).
    ious: optional precomputed (n_pred, n_gt) IoU matrix (e.g. mask IoU); otherwise box IoU."""
    order = np.argsort(-pred_conf, kind="stable")
    tp = np.zeros(len(pred_boxes), bool)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return tp
    m = iou_matrix(pred_boxes, gt_boxes) if ious is None else ious
    used = np.zeros(len(gt_boxes), bool)
    for i in order:
        cand = np.where(~used & (m[i] >= thr))[0]
        if len(cand):
            j = cand[np.argmax(m[i, cand])]
            used[j] = True
            tp[i] = True
    return tp


def nms(boxes, scores, thr):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < thr]
    return np.array(keep, int)


def ap_from_sorted(tp_w, fp_w, n_gt):
    """tp_w/fp_w: weights of predictions already sorted by descending confidence."""
    if n_gt <= 0:
        return np.nan
    ctp, cfp = np.cumsum(tp_w), np.cumsum(fp_w)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
    rs = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, rs, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


class ClassData:
    """Per class and arm: predictions (sorted by confidence) with their block and TP flags at each
    IoU threshold, plus the ground-truth count per block. Lets the bootstrap re-weight blocks
    without re-matching."""

    def __init__(self, conf, block, tp, gt_per_block, n_blocks):
        o = np.argsort(-conf, kind="stable")
        self.conf, self.block, self.tp = conf[o], block[o], tp[:, o]  # tp: (n_iou, n_pred)
        self.gt = np.bincount(gt_per_block, minlength=n_blocks) if len(gt_per_block) else np.zeros(n_blocks, int)

    def metrics(self, w, conf_thr, full=True):
        """w: weight (multiplicity) per block. full=False skips AP50-95 (used inside the bootstrap)."""
        pw = w[self.block] if len(self.block) else np.zeros(0)
        n_gt = float((self.gt * w).sum())
        if n_gt == 0:
            return None
        ap = [ap_from_sorted(pw * self.tp[t], pw * ~self.tp[t], n_gt) for t in (range(len(IOUS)) if full else [0])]
        sel = self.conf >= conf_thr
        tp = float((pw[sel] * self.tp[0][sel]).sum())
        fp = float((pw[sel] * ~self.tp[0][sel]).sum())
        r = tp / n_gt
        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        return {"AP50": ap[0], "AP50-95": float(np.mean(ap)), "P": p, "R": r,
                "F1": 2 * p * r / (p + r) if p + r > 0 else 0.0, "instances": n_gt}


def build_classdata(folds, gt, preds_arm, classes, bidx, n_blocks, iou_type="box"):
    ANY = -1
    acc = defaultdict(lambda: {"conf": [], "block": [], "tp": [[] for _ in IOUS], "gtb": []})
    for f, (k, b) in folds.items():
        g = gt[f]
        pr = preds_arm.get(f, [])
        P = np.array([r[2:6] for r in pr], float).reshape(-1, 4)
        S = np.array([r[1] for r in pr], float)
        C = np.array([r[0] for r in pr], int)
        G = np.array([r[1:5] for r in g], float).reshape(-1, 4)
        GC = np.array([r[0] for r in g], int)
        MI = None
        if iou_type == "mask":
            if pr and len(pr[0]) < 7:
                raise SystemExit("--iou-type mask needs segmentation predictions with mask IoUs (train_experiment.py "
                                 "--task segment --save-preds)")
            MI = np.array([r[6] for r in pr], float).reshape(len(pr), len(g))
        for c in set(classes) | set(C.tolist()):
            pm, gm = C == c, GC == c
            a = acc[c]
            a["conf"].extend(S[pm]); a["block"].extend([bidx[b]] * int(pm.sum())); a["gtb"].extend([bidx[b]] * int(gm.sum()))
            sub = MI[np.ix_(pm, gm)] if MI is not None else None
            for t, thr in enumerate(IOUS):
                a["tp"][t].extend(match(P[pm], S[pm], G[gm], thr, sub))
        keep = nms(P, S, 0.6) if len(P) else np.array([], int)
        a = acc[ANY]
        a["conf"].extend(S[keep]); a["block"].extend([bidx[b]] * len(keep)); a["gtb"].extend([bidx[b]] * len(G))
        subk = MI[keep] if MI is not None else None
        for t, thr in enumerate(IOUS):
            a["tp"][t].extend(match(P[keep], S[keep], G, thr, subk))
    return {c: ClassData(np.array(a["conf"], float), np.array(a["block"], int),
                         np.array([np.array(x, bool) for x in a["tp"]]).reshape(len(IOUS), -1),
                         np.array(a["gtb"], int), n_blocks) for c, a in acc.items()}


def main():
    args = parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    folds = {}
    with open(args.data_dir / "folds.csv") as fh:
        for r in csv.DictReader(fh):
            folds[int(r["frame"])] = (int(r["fold"]), int(r["block"]))
    blocks = sorted({b for _, b in folds.values()})
    bidx = {b: i for i, b in enumerate(blocks)}
    n_blocks = len(blocks)
    n_folds = max(k for k, _ in folds.values()) + 1
    yaml = (args.data_dir / "fold0" / "det_baseline.yaml").read_text().splitlines()
    names = {int(l.strip().split(":", 1)[0]): l.split(":", 1)[1].strip() for l in yaml
             if l.startswith("  ") and l.strip().split(":", 1)[0].isdigit()}
    focus = json.loads((args.data_dir / "cv_summary.json").read_text())["focus_species"]
    focus_ids = [next(k for k, v in names.items() if v == s) for s in focus]
    ANY = -1
    FOCUS_MEAN = "Focus species (mean)"
    ANY_ROW = "Any fish (class-agnostic)"

    gt = {}
    for f, (k, b) in folds.items():
        lf = args.data_dir / f"fold{k}" / "det_baseline" / "labels" / "test" / f"frame_{f}.txt"
        rows = []
        for line in (lf.read_text().splitlines() if lf.exists() else []):
            p = line.split()
            if len(p) >= 5:
                c, cx, cy, w, h = int(p[0]), *(float(v) for v in p[1:5])
                rows.append((c, (cx - w / 2) * W_IMG, (cy - h / 2) * H_IMG, (cx + w / 2) * W_IMG, (cy + h / 2) * H_IMG))
        gt[f] = rows
    classes = sorted({c for rows in gt.values() for c, *_ in rows})

    missing, data = [], {}
    for sd in seeds:
        data[sd] = {}
        for arm, _ in ARMS:
            preds = {}
            for k in range(n_folds):
                pf = args.runs_dir / args.run_pattern.format(arm=arm, k=k, suffix="" if sd == 0 else f"_s{sd}") / "test_predictions.json"
                if not pf.exists():
                    missing.append(str(pf))
                    continue
                for stem, rows in json.loads(pf.read_text()).items():
                    if stem.startswith("frame_"):
                        preds[int(stem.split("_")[1])] = rows
            data[sd][arm] = build_classdata(folds, gt, preds, classes, bidx, n_blocks, args.iou_type)
    if missing:
        raise SystemExit("missing prediction files (did every fold finish, with --save-preds?):\n  " + "\n  ".join(missing))

    def M(arm, c, w, full=True):
        """metric dict averaged over the selected seeds"""
        ms = [data[sd][arm][c].metrics(w, args.conf, full) for sd in seeds if c in data[sd][arm]]
        ms = [m for m in ms if m]
        if not ms:
            return None
        return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}

    ones = np.ones(n_blocks)
    full = {arm: {c: M(arm, c, ones) for c in classes + [ANY]} for arm, _ in ARMS}
    reliable = [c for c in classes if full["baseline"].get(c) and full["baseline"][c]["instances"] >= args.min_instances]
    MACRO = f"Macro mAP50, {len(reliable)} classes with >= {args.min_instances} labels"

    def headline(w, fullm=True):
        out = {}
        for arm, _ in ARMS:
            m = {s: M(arm, c, w, fullm) for s, c in zip(focus, focus_ids)}
            fm = [m[s] for s in focus if m[s]]
            m[FOCUS_MEAN] = {"AP50": float(np.mean([x["AP50"] for x in fm])), "R": float(np.mean([x["R"] for x in fm])),
                             "instances": float(sum(x["instances"] for x in fm))}
            m[ANY_ROW] = M(arm, ANY, w, fullm)
            aps = [M(arm, c, w, fullm) for c in reliable]
            m[MACRO] = {"AP50": float(np.nanmean([a["AP50"] for a in aps if a])), "R": float(np.nanmean([a["R"] for a in aps if a]))}
            out[arm] = m
        return out

    point = headline(ones)
    rows_keys = list(point["baseline"].keys())
    rng = np.random.default_rng(args.seed)
    boot = {(k, arm, met): [] for k in rows_keys for arm, _ in ARMS for met in ("AP50", "R")}
    for _ in range(args.boot):
        w = np.bincount(rng.integers(0, n_blocks, n_blocks), minlength=n_blocks).astype(float)
        h = headline(w, fullm=False)
        for k in rows_keys:
            for arm, _ in ARMS:
                for met in ("AP50", "R"):
                    boot[(k, arm, met)].append(h[arm][k][met] if h[arm][k] else np.nan)

    def ci(v):
        v = np.array(v, float)
        v = v[~np.isnan(v)]
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)

    res = []
    for k in rows_keys:
        for met in ("AP50", "R"):
            b = np.array(boot[(k, "baseline", met)], float)
            c = np.array(boot[(k, "copypaste", met)], float)
            d = c - b
            d = d[~np.isnan(d)]
            p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if len(d) else np.nan
            pb, pc = point["baseline"][k], point["copypaste"][k]
            res.append({"row": k, "metric": met,
                        "instances": int(pb["instances"]) if pb and "instances" in pb else "",
                        "baseline": pb[met] if pb else np.nan, "baseline_ci": ci(b),
                        "copypaste": pc[met] if pc else np.nan, "copypaste_ci": ci(c),
                        "delta": (pc[met] - pb[met]) if (pb and pc) else np.nan, "delta_ci": ci(d), "p": p})

    per_fold = []
    for k in range(n_folds):
        w = np.array([1.0 if any(folds[f] == (k, b) for f in folds) else 0.0 for b in blocks])
        for s, c in zip(focus, focus_ids):
            row = {"fold": k, "species": s, "instances": 0}
            for arm, _ in ARMS:
                m = M(arm, c, w)
                row[f"{arm}_AP50"] = round(m["AP50"], 4) if m else ""
                row[f"{arm}_R"] = round(m["R"], 4) if m else ""
                row["instances"] = int(m["instances"]) if m else row["instances"]
            per_fold.append(row)

    out = args.runs_dir / (args.out_name or ("cv_results" if seeds == [0] else "cv_results_seeds" + "-".join(map(str, seeds))))
    out.mkdir(parents=True, exist_ok=True)
    pc_rows = []
    for c in classes + [ANY]:
        for arm, _ in ARMS:
            m = full[arm].get(c)
            if m:
                pc_rows.append([names.get(c, ANY_ROW), arm, int(m["instances"]),
                                *(round(m[x], 4) for x in ("AP50", "AP50-95", "P", "R", "F1")), int(c in reliable)])
    pc_head = ["class", "arm", "instances", "AP50", "AP50-95", "P", "R", "F1", "reliable"]
    with open(out / "per_class_pooled.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(pc_head)
        wr.writerows(pc_rows)
    with open(out / "per_fold.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(per_fold[0].keys()))
        wr.writeheader()
        wr.writerows(per_fold)

    f3 = lambda v: "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    seed_txt = f"seed {seeds[0]}" if len(seeds) == 1 else f"seeds {' + '.join(map(str, seeds))}, averaged"
    if args.label:
        seed_txt = f"{args.label}; {seed_txt}"
    lines = [f"# Cross-validation: baseline vs copy-paste ({'mask' if args.iou_type == 'mask' else 'box'} AP, pooled over 5 test folds, {seed_txt})", "",
             f"Every labelled frame in {len(folds)} frames is scored once, by the fold model that never trained on it. "
             f"Brackets: 95% interval from a paired block bootstrap ({args.boot} resamples of {n_blocks} 30-frame blocks). "
             f"p: two-sided bootstrap p-value for the difference. Recall is at confidence {args.conf}. "
             f"Primary endpoint: '{FOCUS_MEAN}' AP50.", "",
             "| | Labels | Metric | Baseline | Copy-paste | Difference [95% CI] | p |", "|---|---|---|---|---|---|---|"]
    for r in res:
        lab = f"**{r['row']}**" if r["row"] == FOCUS_MEAN else r["row"]
        lines.append(f"| {lab} | {r['instances']} | {r['metric']} | {f3(r['baseline'])} [{f3(r['baseline_ci'][0])}–{f3(r['baseline_ci'][1])}] | "
                     f"{f3(r['copypaste'])} [{f3(r['copypaste_ci'][0])}–{f3(r['copypaste_ci'][1])}] | "
                     f"{r['delta']:+.3f} [{r['delta_ci'][0]:+.3f} to {r['delta_ci'][1]:+.3f}] | {f3(r['p'])} |")
    lines += ["", "A difference counts as a finding only if its 95% interval excludes 0. The focus-species mean is the "
              "single pre-specified test; treat the per-species rows as supporting detail.", "",
              "## All metrics, focus species", "", "| Species | Metric | Baseline | Copy-paste | Change |", "|---|---|---|---|---|"]
    for s, c in zip(focus, focus_ids):
        for met in ("AP50", "AP50-95", "P", "R", "F1"):
            b, cpm = full["baseline"].get(c), full["copypaste"].get(c)
            if b and cpm:
                lines.append(f"| {s} | {met} | {b[met]:.3f} | {cpm[met]:.3f} | {cpm[met] - b[met]:+.3f} |")
    lines += ["", "## Per fold (AP50 / recall)", "",
              "| Fold | Species | Test labels | Baseline AP50 | Copy-paste AP50 | Baseline R | Copy-paste R |", "|---|---|---|---|---|---|---|"]
    for r in per_fold:
        lines.append(f"| {r['fold']} | {r['species']} | {r['instances']} | {r['baseline_AP50']} | {r['copypaste_AP50']} | {r['baseline_R']} | {r['copypaste_R']} |")
    lines += ["", f"Reliable classes (>= {args.min_instances} pooled test labels): " + ", ".join(names[c] for c in reliable)]
    (out / "cv_results.md").write_text("\n".join(lines) + "\n")

    figs = make_figures(out, res, full, focus, focus_ids, per_fold, seed_txt)
    print((out / "cv_results.md").read_text())
    print(f"wrote {out}")

    if args.wandb_project:
        try:
            import wandb
            run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                             name="cv_eval_" + out.name.replace("cv_results_", "").replace("cv_results", "seeds" + "-".join(map(str, seeds))),
                             job_type="evaluation", group="evaluation",
                             tags=["evaluation", "cv"] + [f"seed{sd}" for sd in seeds],
                             config={"seeds": seeds, "boot": args.boot, "iou_type": args.iou_type, "label": args.label,
                                     "run_pattern": args.run_pattern, "data_dir": str(args.data_dir), "conf": args.conf, "folds": n_folds,
                                     "blocks": n_blocks, "focus_species": focus, "primary_endpoint": f"{FOCUS_MEAN} AP50"})
            cols = ["row", "metric", "labels", "baseline", "baseline_lo", "baseline_hi", "copypaste", "copypaste_lo",
                    "copypaste_hi", "delta", "delta_lo", "delta_hi", "p"]
            tab = [[r["row"], r["metric"], r["instances"], r["baseline"], *r["baseline_ci"], r["copypaste"], *r["copypaste_ci"],
                    r["delta"], *r["delta_ci"], r["p"]] for r in res]
            wandb.log({"results/headline": wandb.Table(columns=cols, data=tab, allow_mixed_types=True),
                       "results/per_class_pooled": wandb.Table(columns=pc_head, data=pc_rows, allow_mixed_types=True),
                       "results/per_fold": wandb.Table(columns=list(per_fold[0].keys()), data=[list(r.values()) for r in per_fold], allow_mixed_types=True),
                       **{f"figures/{p.stem}": wandb.Image(str(p)) for p in figs}})
            summ = {}
            for r in res:
                key = r["row"].split(",")[0].replace(" ", "_").replace("(", "").replace(")", "")
                for part in ("baseline", "copypaste", "delta"):
                    summ[f"{key}/{r['metric']}/{part}"] = r[part]
                summ[f"{key}/{r['metric']}/delta_lo"], summ[f"{key}/{r['metric']}/delta_hi"] = r["delta_ci"]
                summ[f"{key}/{r['metric']}/p"] = r["p"]
            run.summary.update(summ)
            wandb.finish()
        except Exception as e:
            print(f"W&B logging skipped ({e}); everything is in {out}")


def _style(plt):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2})


def _clean(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for sd in ("top", "right", "left"):
        ax.spines[sd].set_visible(False)
    ax.tick_params(length=0)
    (ax.xaxis if grid_axis == "x" else ax.yaxis).grid(True, color=GRID, lw=0.8, zorder=0)


def make_figures(out, res, full, focus, focus_ids, per_fold, seed_txt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    paths = []

    def ci_figure(met, title, fname):
        rows = [r for r in res if r["metric"] == met]
        labels = [r["row"].replace(" (class-agnostic)", "\n(class-agnostic)").replace(" (mean)", "\n(mean)").replace(", ", "\n", 1)
                  for r in rows]
        y = np.arange(len(rows))[::-1]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 0.75 * len(rows) + 1.8), facecolor=SURFACE,
                                     gridspec_kw={"width_ratios": [1.4, 1]}, sharey=True)
        for ax in (a1, a2):
            _clean(ax)
        h = 0.34
        for j, (arm, lab) in enumerate(ARMS):
            yy = y + (h / 2 + 0.02) * (1 if j == 0 else -1)
            vals = [r[arm] for r in rows]
            lo = [r[arm] - r[f"{arm}_ci"][0] for r in rows]
            hi = [r[f"{arm}_ci"][1] - r[arm] for r in rows]
            a1.barh(yy, vals, h, color=COL[arm], label=lab, zorder=3, linewidth=0)
            a1.errorbar(vals, yy, xerr=[lo, hi], fmt="none", ecolor=INK, lw=1, capsize=2, zorder=4)
            for v, yv, e in zip(vals, yy, hi):
                a1.text(v + e + 0.015, yv, f"{v:.2f}", va="center", fontsize=7.5, color=INK2)
        a1.set_yticks(y)
        a1.set_yticklabels(labels)
        for t in a1.get_yticklabels():
            if "Focus species" in t.get_text():
                t.set_fontweight("bold")
        a1.set_xlim(0, 1.05)
        a1.set_title(f"{title}, pooled test folds (95% CI)", loc="left", fontsize=10, color=INK, pad=8)
        a1.legend(frameon=False, loc="lower right", fontsize=8)
        a2.axvline(0, color=AXIS, lw=1, zorder=2)
        a2.errorbar([r["delta"] for r in rows], y,
                    xerr=[[r["delta"] - r["delta_ci"][0] for r in rows], [r["delta_ci"][1] - r["delta"] for r in rows]],
                    fmt="o", color=COL["copypaste"], ecolor=INK, ms=8, lw=1, capsize=2, zorder=4)
        for yv, r in zip(y, rows):
            a2.text(r["delta_ci"][1] + 0.01, yv, f"{r['delta']:+.3f}  p={r['p']:.3f}", va="center", fontsize=7.5, color=INK2)
        m = max(0.1, max(abs(r["delta_ci"][0]) for r in rows), max(abs(r["delta_ci"][1]) for r in rows)) * 1.7
        a2.set_xlim(-m, m)
        a2.set_title("Copy-paste minus baseline (95% CI)", loc="left", fontsize=10, color=INK, pad=8)
        fig.suptitle(seed_txt, x=0.01, ha="left", fontsize=8, color=MUTED)
        fig.tight_layout()
        p = out / fname
        fig.savefig(p, dpi=200, facecolor=SURFACE)
        plt.close(fig)
        paths.append(p)
        return p

    ap = ci_figure("AP50", "AP50", "fig_ap50.png")
    import shutil
    shutil.copy(ap, out / "cv_results.png")
    ci_figure("R", "Recall at conf 0.25", "fig_recall.png")

    # all metrics, focus species: small multiples, one panel per species
    mets = ["AP50", "AP50-95", "P", "R", "F1"]
    fig, axes = plt.subplots(1, len(focus), figsize=(3.6 * len(focus), 3.2), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(mets))
    for ax, s, c in zip(axes, focus, focus_ids):
        _clean(ax, "y")
        for j, (arm, lab) in enumerate(ARMS):
            m = full[arm].get(c)
            vals = [m[k] if m else 0 for k in mets]
            xx = x + (j - 0.5) * 0.38
            ax.bar(xx, vals, 0.36, color=COL[arm], label=lab, zorder=3, linewidth=0)
            for xv, v in zip(xx, vals):
                ax.text(xv, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
        n = int(full["baseline"][c]["instances"]) if full["baseline"].get(c) else 0
        ax.set_title(f"{s} ({n} test labels)", loc="left", fontsize=9.5, color=INK, pad=6)
        ax.set_xticks(x)
        ax.set_xticklabels(["AP50", "AP50-95", "Precision", "Recall", "F1"], fontsize=8, color=INK2)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].legend(frameon=False, loc="upper left", fontsize=8, ncol=2)
    fig.suptitle(f"All metrics, pooled test folds ({seed_txt}); P, R, F1 at conf 0.25", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    p = out / "fig_all_metrics.png"
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)

    # per fold: paired baseline -> copy-paste AP50, marker size = test labels
    fig, axes = plt.subplots(1, len(focus), figsize=(3.2 * len(focus), 3.4), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, focus):
        _clean(ax, "y")
        rows = [r for r in per_fold if r["species"] == s and r["baseline_AP50"] != "" and r["copypaste_AP50"] != ""]
        # label positions: spread apart so neighbouring fold labels don't overlap
        order = sorted(rows, key=lambda r: r["copypaste_AP50"])
        ypos, last = {}, -1.0
        for r in order:
            yv = max(r["copypaste_AP50"], last + 0.055)
            ypos[r["fold"]] = yv
            last = yv
        for r in rows:
            b, cpv = r["baseline_AP50"], r["copypaste_AP50"]
            ax.plot([0, 1], [b, cpv], color=AXIS, lw=1.5, zorder=2)
            sz = 18 + 6 * r["instances"]
            ax.scatter([0], [b], s=sz, color=COL["baseline"], zorder=3, edgecolors=SURFACE, linewidths=1.5)
            ax.scatter([1], [cpv], s=sz, color=COL["copypaste"], zorder=3, edgecolors=SURFACE, linewidths=1.5)
            ax.text(1.1, ypos[r["fold"]], f"fold {r['fold']} (n={r['instances']})", va="center", fontsize=6.5, color=INK2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "Copy-paste"], color=INK2)
        ax.set_xlim(-0.25, 1.75)
        ax.set_ylim(0, max(1.05, last + 0.05))
        ax.set_title(s, loc="left", fontsize=9.5, color=INK, pad=6)
    axes[0].set_ylabel("AP50 on that fold's test set", color=INK2)
    fig.suptitle(f"Per-fold AP50 ({seed_txt}); marker size = test labels in the fold", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    p = out / "fig_per_fold.png"
    fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    paths.append(p)
    return paths


if __name__ == "__main__":
    main()
