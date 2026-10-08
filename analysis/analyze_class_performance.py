#!/usr/bin/env python3
"""
Quick per-class diagnostic analysis from a trained run's per_class_metrics.csv
(written by train_yolo.py's write_per_class_csv()). Answers, without any new
training:

  - "isolate whether poor aggregate performance is because of a few
    poor-performing classes" -- prints each class's instance count next to
    its recall/precision/mAP50, sorted worst-to-best.
  - "weighted accuracy" -- prints both the macro average (plain mean across
    classes, what mAP normally reports) and the instance-weighted average
    (weighted by each class's real instance count) for recall/precision/
    mAP50, so you can see how much a handful of tiny classes are dragging
    the macro number down vs. how the model does on the data it actually
    sees most.
  - "check what the majority [class] is doing" -- explicitly calls out the
    largest and smallest classes by instance count.
  - instances-vs-performance scatter plot (saved as PNG) -- the "size vs
    performance, hypothesis for why it's bad-performing" analysis, to look
    at BEFORE deciding whether to label more data.
  - optional bbox-size-vs-performance, if you pass --labels-dir (computes
    each class's average bounding-box area from the YOLO label files and
    correlates it against recall).

Column names in per_class_metrics.csv can vary a bit by Ultralytics version
(e.g. "Box-R" vs "R" vs "recall", or "Mask-R" for segmentation models) --
this script prints every column it found and which ones it auto-matched to
class/instances/precision/recall/mAP50, so you can sanity-check or override
with --class-col/--instances-col/--precision-col/--recall-col/--map50-col if
the auto-detection picks the wrong one (e.g. it picks Box- metrics but you
want Mask- metrics from a segmentation run).

IMPORTANT for --labels-dir (size-vs-performance): the label .txt files use
numeric YOLO class IDs, but per_class_metrics.csv's Class column already
contains species NAMES (Ultralytics reads those from your data.yaml). To
bridge the two, pass --data-yaml pointing at the SAME data.yaml the run used
(e.g. data_detect_v3.yaml) -- this script reads its `names:` field to build
the id<->name mapping automatically. Without --data-yaml (or --names typed
out by hand), the size section cannot match IDs to names and every recall
will print as nan.

Usage
-----
    python3 analyze_class_performance.py \\
        --metrics-csv runs_seg_top10/aqua_run1_medium/per_class_metrics.csv \\
        --data-yaml data_seg_top10.yaml \\
        --out-dir analysis_top10_seg_medium

    # add bbox-size-vs-performance:
    python3 analyze_class_performance.py \\
        --metrics-csv runs_detect_v3_equalize/aqua_run1_medium/per_class_metrics.csv \\
        --labels-dir dataset_detect_v3/labels/train \\
        --data-yaml data_detect_v3.yaml \\
        --out-dir analysis_full_detect_equalize_with_size
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-csv", required=True, type=Path)
    p.add_argument("--labels-dir", type=Path, default=None,
                   help="Optional: a labels/train dir of YOLO detect-format .txt files, to compute "
                        "avg bbox area per class and correlate against recall.")
    p.add_argument("--data-yaml", type=Path, default=None,
                   help="Optional: the data.yaml the run used (e.g. data_detect_v3.yaml). Its `names:` "
                        "field is read to build the class-id <-> class-name mapping needed to match "
                        "--labels-dir (numeric IDs) against per_class_metrics.csv (names). Preferred "
                        "over --names since it can't get out of sync with what the run actually used.")
    p.add_argument("--names", type=str, default=None,
                   help="Optional comma-separated class names in class-id order, for readability. "
                        "Overridden by --data-yaml if both are given.")
    p.add_argument("--out-dir", type=Path, default=Path("class_perf_analysis"))
    p.add_argument("--class-col", type=str, default=None)
    p.add_argument("--instances-col", type=str, default=None)
    p.add_argument("--precision-col", type=str, default=None)
    p.add_argument("--recall-col", type=str, default=None)
    p.add_argument("--map50-col", type=str, default=None)
    return p.parse_args()


def load_names_from_yaml(yaml_path):
    """Returns {class_id: name} read from a YOLO data.yaml's `names:` field.
    Tries PyYAML first (virtually always present alongside ultralytics); falls
    back to a small regex-based parser for the two common formats (a flat
    list, or an id: name mapping) if PyYAML isn't importable."""
    text = yaml_path.read_text()
    try:
        import yaml
        doc = yaml.safe_load(text)
        names = doc.get("names")
        if isinstance(names, dict):
            return {int(k): v for k, v in names.items()}
        if isinstance(names, list):
            return {i: v for i, v in enumerate(names)}
    except ImportError:
        pass

    # ---- fallback: manual parse ----
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r"^names\s*:", l)), None)
    if start is None:
        return {}
    block = []
    base_indent = None
    for l in lines[start + 1:]:
        if not l.strip():
            continue
        indent = len(l) - len(l.lstrip())
        if base_indent is None:
            base_indent = indent
        if indent < base_indent and l.strip() and not l.strip().startswith("-"):
            break
        block.append(l.strip())
    id_to_name = {}
    # dict style: "0: Species Name"
    dict_style = [l for l in block if re.match(r"^\d+\s*:", l)]
    if dict_style:
        for l in dict_style:
            k, _, v = l.partition(":")
            id_to_name[int(k.strip())] = v.strip().strip("'\"")
        return id_to_name
    # list style: "- Species Name"
    list_style = [l[1:].strip().strip("'\"") for l in block if l.startswith("-")]
    if list_style:
        return {i: v for i, v in enumerate(list_style)}
    return {}


def autodetect(columns, override, patterns, exclude_patterns=()):
    if override:
        return override
    low = {c: c.lower() for c in columns}
    for c, cl in low.items():
        if any(ex in cl for ex in exclude_patterns):
            continue
        if any(pat in cl for pat in patterns):
            return c
    return None


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.metrics_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{args.metrics_csv} has no data rows")
    columns = list(rows[0].keys())
    print(f"columns found in {args.metrics_csv}:\n  {columns}\n")

    class_col = autodetect(columns, args.class_col, ["class"])
    instances_col = autodetect(columns, args.instances_col, ["instance"])
    map50_col = autodetect(columns, args.map50_col, ["map50"], exclude_patterns=["map50-95", "map5095"])
    recall_col = autodetect(columns, args.recall_col, ["recall", "-r", "_r"], exclude_patterns=["precision"])
    precision_col = autodetect(columns, args.precision_col, ["precision", "-p", "_p"])

    print("auto-matched columns (override with --class-col etc. if any of these look wrong):")
    print(f"  class      -> {class_col}")
    print(f"  instances  -> {instances_col}")
    print(f"  precision  -> {precision_col}")
    print(f"  recall     -> {recall_col}")
    print(f"  mAP50      -> {map50_col}\n")

    if not all([class_col, instances_col, recall_col]):
        raise SystemExit("Could not auto-detect required columns (class/instances/recall). "
                          "Re-run with --class-col/--instances-col/--recall-col set explicitly "
                          "using the exact column names printed above.")

    id_to_name = {}
    if args.data_yaml:
        if not args.data_yaml.exists():
            print(f"WARNING: --data-yaml {args.data_yaml} does not exist -- ignoring it")
        else:
            id_to_name = load_names_from_yaml(args.data_yaml)
            print(f"loaded {len(id_to_name)} class names from {args.data_yaml}\n")
    elif args.names:
        id_to_name = {i: n.strip() for i, n in enumerate(args.names.split(","))}
    name_to_id = {v: k for k, v in id_to_name.items()}

    if args.labels_dir and not id_to_name:
        print("WARNING: --labels-dir was given but neither --data-yaml nor --names resolved any class "
              "names -- the size-vs-performance section below will not be able to match IDs to names "
              "and every recall will show as nan. Pass --data-yaml pointing at the data.yaml this run "
              "used (e.g. --data-yaml data_detect_v3.yaml).\n")

    data = []
    for r in rows:
        cid_raw = r[class_col]
        cid = None
        try:
            cid = int(float(cid_raw))
            label = id_to_name.get(cid, str(cid_raw))
        except (TypeError, ValueError):
            # Class column already holds a name string (the common case), not a numeric id.
            label = str(cid_raw)
            cid = name_to_id.get(label)
        data.append({
            "class": label,
            "cid": cid,
            "instances": to_float(r[instances_col]),
            "precision": to_float(r[precision_col]) if precision_col else np.nan,
            "recall": to_float(r[recall_col]),
            "map50": to_float(r[map50_col]) if map50_col else np.nan,
        })

    data.sort(key=lambda d: d["recall"])

    print("=== per-class performance, worst recall first ===")
    print(f"{'class':<28} {'instances':>10} {'precision':>10} {'recall':>8} {'mAP50':>8}")
    for d in data:
        print(f"{d['class']:<28} {d['instances']:>10.0f} {d['precision']:>10.3f} "
              f"{d['recall']:>8.3f} {d['map50']:>8.3f}")

    insts = np.array([d["instances"] for d in data])
    total = insts.sum()

    def macro(key):
        vals = np.array([d[key] for d in data])
        return np.nanmean(vals)

    def weighted(key):
        vals = np.array([d[key] for d in data])
        mask = ~np.isnan(vals)
        return float(np.sum(vals[mask] * insts[mask]) / insts[mask].sum()) if mask.any() else float("nan")

    print("\n=== macro (unweighted mean across classes) vs instance-weighted average ===")
    for key in ("precision", "recall", "map50"):
        m, w = macro(key), weighted(key)
        gap_note = "  <-- macro notably lower: a few low-support classes are dragging the mean down" \
            if (not np.isnan(m) and not np.isnan(w) and m < w - 0.05) else ""
        print(f"  {key:<10} macro={m:.3f}   weighted={w:.3f}{gap_note}")

    biggest = max(data, key=lambda d: d["instances"])
    smallest = min(data, key=lambda d: d["instances"])
    print(f"\nmajority class:  {biggest['class']} ({biggest['instances']:.0f} instances) "
          f"-> recall={biggest['recall']:.3f}, mAP50={biggest['map50']:.3f}")
    print(f"minority class:  {smallest['class']} ({smallest['instances']:.0f} instances) "
          f"-> recall={smallest['recall']:.3f}, mAP50={smallest['map50']:.3f}")

    # ---- instances vs recall scatter ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(insts, [d["recall"] for d in data], s=60)
        for d in data:
            ax.annotate(d["class"], (d["instances"], d["recall"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("real train instance count")
        ax.set_ylabel("recall")
        ax.set_title("Per-class instance count vs. recall")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "instances_vs_recall.png", dpi=150)
        print(f"\nwrote {args.out_dir / 'instances_vs_recall.png'}")

        valid = [(d["instances"], d["recall"]) for d in data if not np.isnan(d["recall"])]
        if len(valid) >= 3:
            xs, ys = zip(*valid)
            corr = float(np.corrcoef(xs, ys)[0, 1])
            print(f"Pearson correlation (instances vs recall): {corr:.3f} "
                  f"({'supports' if corr > 0.3 else 'does NOT clearly support' if corr < 0.15 else 'weakly supports'}"
                  f" 'more instances -> better recall' as the main driver)")
    except ImportError:
        print("matplotlib not installed -- skipping scatter plot (pip install matplotlib --break-system-packages)")

    # ---- optional: bbox size vs performance ----
    if args.labels_dir:
        if not args.labels_dir.exists():
            print(f"\n--labels-dir {args.labels_dir} does not exist -- skipping size-vs-performance")
        else:
            areas_by_class = {}
            counts_by_class = {}
            for txt in args.labels_dir.glob("*.txt"):
                for line in txt.read_text().splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    cid = int(float(parts[0]))
                    w, h = float(parts[3]), float(parts[4])  # normalized YOLO detect format
                    areas_by_class.setdefault(cid, []).append(w * h)
            print("\n=== avg normalized bbox area per class (proxy for object size) vs recall ===")
            size_rows = []
            for cid, areas in sorted(areas_by_class.items()):
                label = id_to_name.get(cid, str(cid))
                match = next((d for d in data if d["cid"] == cid), None)
                avg_area = float(np.mean(areas))
                rec = match["recall"] if match else float("nan")
                size_rows.append((label, avg_area, rec, len(areas)))
                print(f"  {label:<28} avg_bbox_area={avg_area:.5f}  n={len(areas):<5}  recall={rec}")
            valid = [(a, r) for _, a, r, _ in size_rows if not np.isnan(r)]
            if len(valid) >= 3:
                areas_v, recalls_v = zip(*valid)
                corr = float(np.corrcoef(areas_v, recalls_v)[0, 1])
                print(f"\nPearson correlation (avg bbox area vs recall): {corr:.3f}")


if __name__ == "__main__":
    main()
