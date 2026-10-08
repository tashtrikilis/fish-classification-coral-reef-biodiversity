#!/usr/bin/env python3
"""
Tests the "dense schooling causes missed detections" hypothesis, raised at
the 2026-09-18 supervisor meeting: Orangeband Surgeonfish has the worst
recall residual in the full 58-class dataset despite having the 2nd-most
validation instances, and was observed during annotation to almost always
travel in dense schools -- meaning heavy same-class occlusion/overlap in a
frame, not necessarily inter-species confusion, could be why it's missed so
often. This is a different, independently testable hypothesis from the
family-confusability one in analyze_family_confusion.py.

For each class, computes from the YOLO label files alone (no training
needed):
  - avg "school size": among frames that contain at least one instance of
    this class, the average number of instances of that SAME class in that
    frame (1.0 = always alone, higher = frequently multiple in-frame).
  - % of that class's frames with 2+ instances present.
  - avg / max pairwise IoU between same-class boxes within the same frame,
    among frames with 2+ instances (a direct occlusion/overlap proxy).
  - fraction of instances that have at least one same-class neighbor
    overlapping above --iou-threshold in the same frame ("crowded fraction").

If a class's crowding metrics are unusually high AND it also has a notably
low/negative recall residual, that's direct support for the schooling
hypothesis for that class specifically (as opposed to it being a dataset-
wide pattern -- most classes should NOT show heavy same-class crowding).

Pass --metrics-csv to also print the correlation between crowding and
recall directly.

Usage
-----
    python3 analyze_class_crowding.py \\
        --labels-dir dataset_detect_v3/labels/train \\
        --data-yaml data_detect_v3.yaml \\
        --metrics-csv runs_detect_v3_equalize/aqua_run1_medium/per_class_metrics.csv \\
        --out-dir crowding_analysis_detect_equalize
"""
import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from analyze_class_performance import autodetect, to_float, load_names_from_yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels-dir", required=True, type=Path,
                   help="YOLO detect-format labels dir, e.g. dataset_detect_v3/labels/train.")
    p.add_argument("--data-yaml", type=Path, default=None, help="For class-id -> name mapping.")
    p.add_argument("--metrics-csv", type=Path, default=None,
                   help="Optional per_class_metrics.csv to correlate crowding against recall.")
    p.add_argument("--iou-threshold", type=float, default=0.05,
                   help="Minimum IoU between two same-class boxes in the same frame to count as "
                        "'overlapping neighbors' (default 0.05 -- fairly loose, since even partial "
                        "occlusion from a nearby schoolmate can confuse a detector).")
    p.add_argument("--out-dir", type=Path, default=Path("crowding_analysis"))
    p.add_argument("--class-col", type=str, default=None)
    p.add_argument("--instances-col", type=str, default=None)
    p.add_argument("--recall-col", type=str, default=None)
    p.add_argument("--min-instances-for-correlation", type=int, default=40,
                   help="IMPORTANT: classes with very few instances almost never have 2+ in the same "
                        "frame simply because they're rare, not because they're genuinely solitary -- "
                        "this shows up as frac_crowded=0.0 for the wrong reason and confounds the "
                        "dataset-wide correlation (rare classes are also disproportionately the "
                        "recall-floor classes, creating a spurious POSITIVE crowding-vs-recall "
                        "correlation that has nothing to do with crowding). The correlation is computed "
                        "only among classes with at least this many instances, where crowding has a "
                        "fair chance to be measured either way. Default 40 -- lower it if too few "
                        "classes qualify, but check by eye whether the confound is creeping back in.")
    return p.parse_args()


def yolo_to_xyxy(xc, yc, w, h):
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def iou(box_a, box_b):
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    id_to_name = load_names_from_yaml(args.data_yaml) if args.data_yaml and args.data_yaml.exists() else {}

    # ---- index instances per frame per class ----
    per_frame_class_boxes = defaultdict(lambda: defaultdict(list))  # frame -> cls -> [boxes]
    total_instances = defaultdict(int)
    for txt in sorted(args.labels_dir.glob("*.txt")):
        for line in txt.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(x) for x in parts[1:5])
            per_frame_class_boxes[txt.stem][cls].append(yolo_to_xyxy(xc, yc, w, h))
            total_instances[cls] += 1

    # ---- per-class school-size / overlap stats ----
    school_sizes = defaultdict(list)   # cls -> [count in each frame it appears in]
    pairwise_ious = defaultdict(list)  # cls -> [iou for every same-class pair in a frame]
    crowded_instance_count = defaultdict(int)  # cls -> # instances with >=1 overlapping same-class neighbor

    for frame, cls_boxes in per_frame_class_boxes.items():
        for cls, boxes in cls_boxes.items():
            school_sizes[cls].append(len(boxes))
            if len(boxes) < 2:
                continue
            has_neighbor = [False] * len(boxes)
            for i, j in combinations(range(len(boxes)), 2):
                v = iou(boxes[i], boxes[j])
                pairwise_ious[cls].append(v)
                if v >= args.iou_threshold:
                    has_neighbor[i] = True
                    has_neighbor[j] = True
            crowded_instance_count[cls] += sum(has_neighbor)

    rows = []
    for cls in sorted(total_instances):
        name = id_to_name.get(cls, str(cls))
        sizes = school_sizes[cls]
        ious = pairwise_ious[cls]
        n = total_instances[cls]
        rows.append({
            "cls": cls,
            "name": name,
            "n_instances": n,
            "avg_school_size": float(np.mean(sizes)) if sizes else 0.0,
            "pct_frames_multi": 100.0 * sum(1 for s in sizes if s >= 2) / len(sizes) if sizes else 0.0,
            "avg_iou_neighbors": float(np.mean(ious)) if ious else 0.0,
            "max_iou_neighbors": float(np.max(ious)) if ious else 0.0,
            "frac_crowded": crowded_instance_count[cls] / n if n else 0.0,
        })

    rows.sort(key=lambda r: r["frac_crowded"], reverse=True)

    print(f"{'class':<32} {'n':>6} {'avg_school':>11} {'%frames_2+':>11} "
          f"{'avg_iou_nb':>11} {'max_iou_nb':>11} {'frac_crowded':>13}")
    for r in rows:
        print(f"{r['name']:<32} {r['n_instances']:>6} {r['avg_school_size']:>11.2f} "
              f"{r['pct_frames_multi']:>10.1f}% {r['avg_iou_neighbors']:>11.3f} "
              f"{r['max_iou_neighbors']:>11.3f} {r['frac_crowded']:>13.3f}")

    # ---- optional: correlate against recall ----
    if args.metrics_csv:
        with open(args.metrics_csv, newline="") as f:
            metric_rows = list(csv.DictReader(f))
        columns = list(metric_rows[0].keys())
        class_col = autodetect(columns, args.class_col, ["class"])
        recall_col = autodetect(columns, args.recall_col, ["recall", "-r", "_r"], exclude_patterns=["precision"])
        instances_col = autodetect(columns, args.instances_col, ["instance"])
        name_to_id = {v: k for k, v in id_to_name.items()}

        recall_by_cls = {}
        for r in metric_rows:
            raw = r[class_col]
            try:
                cid = int(float(raw))
            except (TypeError, ValueError):
                cid = name_to_id.get(str(raw))
            if cid is not None:
                recall_by_cls[cid] = to_float(r[recall_col])

        print("\n=== crowding vs. recall (ALL classes, for reference only) ===")
        print(f"{'class':<32} {'n':>6} {'frac_crowded':>13} {'avg_iou_nb':>11} {'recall':>8}")
        matched_all = []
        for r in rows:
            rec = recall_by_cls.get(r["cls"])
            if rec is None or np.isnan(rec):
                continue
            print(f"{r['name']:<32} {r['n_instances']:>6} {r['frac_crowded']:>13.3f} "
                  f"{r['avg_iou_neighbors']:>11.3f} {rec:>8.3f}")
            matched_all.append((r["n_instances"], r["frac_crowded"], r["avg_iou_neighbors"], rec))

        print(f"\nWARNING: a correlation across ALL classes here is confounded -- classes with very few "
              f"instances show frac_crowded=0.0 simply because they're too rare to ever co-occur, not "
              f"because they're solitary, and those same rare classes are disproportionately the "
              f"recall-floor classes from the earlier analysis. That can produce a spurious POSITIVE "
              f"crowding-vs-recall correlation that has nothing to do with crowding. The number below "
              f"restricts to classes with >= --min-instances-for-correlation "
              f"({args.min_instances_for_correlation}) instances, where crowding has a fair chance to "
              f"show up either way -- this is the number that actually tests the hypothesis.\n")

        matched = [(fc, ai, rc) for n, fc, ai, rc in matched_all if n >= args.min_instances_for_correlation]
        qualifying = [r["name"] for r in rows if r["n_instances"] >= args.min_instances_for_correlation
                      and recall_by_cls.get(r["cls"]) is not None and not np.isnan(recall_by_cls.get(r["cls"]))]
        print(f"=== crowding vs. recall, restricted to classes with >= {args.min_instances_for_correlation} "
              f"instances ({len(matched)} classes: {qualifying}) ===")
        if len(matched) >= 3:
            fc, ai, rc = zip(*matched)
            corr_fc = float(np.corrcoef(fc, rc)[0, 1])
            corr_ai = float(np.corrcoef(ai, rc)[0, 1])
            print(f"Pearson correlation (frac_crowded vs recall): {corr_fc:.3f} "
                  f"({'supports' if corr_fc < -0.3 else 'does not clearly support'} "
                  f"'more same-class crowding -> lower recall')")
            print(f"Pearson correlation (avg_iou_neighbors vs recall): {corr_ai:.3f} "
                  f"({'supports' if corr_ai < -0.3 else 'does not clearly support'} "
                  f"'more same-class overlap -> lower recall')")
        else:
            print(f"Only {len(matched)} classes qualify at this threshold -- too few for a correlation. "
                  f"Lower --min-instances-for-correlation, or just compare the qualifying classes by eye "
                  f"(printed above) -- with this few classes a formal correlation isn't much more "
                  f"trustworthy than reading the table directly.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter([r["frac_crowded"] for r in rows], [r["avg_school_size"] for r in rows], s=60)
        for r in rows:
            ax.annotate(r["name"], (r["frac_crowded"], r["avg_school_size"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("fraction of instances with an overlapping same-class neighbor")
        ax.set_ylabel("avg school size (mean same-class count per occupied frame)")
        ax.set_title("Same-class crowding by species")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "crowding_by_class.png", dpi=150)
        print(f"\nwrote {args.out_dir / 'crowding_by_class.png'}")
    except ImportError:
        print("matplotlib not installed -- skipping plot")


if __name__ == "__main__":
    main()
