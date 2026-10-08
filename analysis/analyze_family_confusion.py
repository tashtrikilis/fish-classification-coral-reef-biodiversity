#!/usr/bin/env python3
"""
Groups per-class performance (from per_class_metrics.csv) by taxonomic
FAMILY, to test the "inter-species visual confusability" hypothesis
directly: do classes that share a family with several other classes in the
dataset (e.g. the many Acanthuridae surgeonfish/tang/unicornfish species)
systematically underperform classes that are visually/taxonomically alone
in the dataset (e.g. Moorish Idol, Chinese Trumpetfish) -- beyond what
instance count alone would predict?

CAVEAT: the FAMILY_GROUPS mapping below is built from common-name
conventions (e.g. "...Surgeonfish/Tang/Unicornfish" = Acanthuridae,
"...Butterflyfish" = Chaetodontidae), not a verified taxonomic reference --
the same annotation-accuracy caveat that applies to the original labelling
(not done by a marine biologist) applies here too. Sanity-check groupings
that matter to your argument before citing them as fact; the overall
family-size-vs-performance pattern should hold even if a species or two
is regrouped.

What this does, beyond the flat per-class table in analyze_class_performance.py:
  1. Groups classes into families, printing each family's member classes
     with their individual recall/instances, sorted worst-family-avg-recall
     first.
  2. Fits a simple linear model recall ~ instances across ALL classes, then
     looks at each class's RESIDUAL (actual recall minus what instance count
     alone would predict). If bigger families have systematically more
     negative residuals than singleton-family classes, that's direct
     evidence family membership hurts recall beyond what data volume alone
     explains -- the confusability hypothesis, tested quantitatively rather
     than just pointed at anecdotally.
  3. Correlates family size (how many sibling species share that family in
     the full 58-species taxonomy) against each class's residual.

Usage
-----
    python3 analyze_family_confusion.py \\
        --metrics-csv runs_detect_v3_equalize/aqua_run1_medium/per_class_metrics.csv \\
        --data-yaml data_detect_v3.yaml \\
        --out-dir family_analysis_detect_equalize
"""
import argparse
import csv
from pathlib import Path

import numpy as np

from analyze_class_performance import autodetect, to_float, load_names_from_yaml


# ---------------------------------------------------------------------------
# Common-name-based family groupings for the 58-species taxonomy used
# throughout this project (see cvat_masks_to_yolo.py's NAMES list).
# NOT a verified taxonomic reference -- see module docstring.
FAMILY_GROUPS = {
    "Acanthuridae (surgeonfish/tang/unicornfish)": [
        "Acanthuridae", "Brown Surgeonfish", "Orangeband Surgeonfish", "Goldring Surgeonfish",
        "Bluelined Surgeonfish", "Whitebar Surgeonfish", "Whitecheek Surgeonfish",
        "Whitespotted Surgeonfish", "Ringtail Surgeonfish", "Convict Tang", "Chevron Tang",
        "Yellow Tang", "Bluespine Unicornfish", "Bullethead Unicornfish", "Orangespine Unicornfish",
    ],
    "Chaetodontidae (butterflyfish)": [
        "Forcepsfish", "Fourspot Butterflyfish", "Lined Butterflyfish", "Lognnose Butterflyfish",
        "Multiband Butterflyfish", "Ornate Butterfly Fish", "Pyramid Butterflyfish",
        "Raccoon Butterflyfish", "Threadfin Butterflyfish",
    ],
    "Scaridae (parrotfish)": [
        "Bullethead Parrotfish - Female", "Palenose Parrotfish", "Palenose Parrotfish - Male",
        "Redlip Parrotfish - Male", "Redlip Parrotfish - Female", "Speckled Parrotfish - Male",
        "Stareye Parrotfish - Male",
    ],
    "Balistidae (triggerfish)": [
        "Black Triggerfish", "Bridled Triggerfish", "Lei Triggerfish", "Pinktail Triggerfish",
    ],
    "Mullidae (goatfish)": [
        "Blue Goatfish", "Doublebar Goatfish", "Manybar Goatfish", "Yellowfin Goatfish",
    ],
    "Pomacentridae (damselfish/chromis/sergeant)": [
        "Blackfin Chromis", "Blackspot Sergeant", "Hawaiian Gregory", "Hawaiian Sergeant",
        "Indo-Pacific Sergeant", "Oval Chromis",
    ],
    "Labridae (wrasse)": [
        "Hawaiian Cleaner Wrasse", "Yellowtail Coris", "Saddle Wrasse - Male",
    ],
    "Kyphosidae (chub)": [
        "Lowfin Chub", "Highfin Chub",
    ],
    "Tetraodontidae (puffer/toby)": [
        "Hawaiian Whitespotted Toby", "Stripebelly Puffer",
    ],
}
# anything not listed above is treated as its own singleton "family" (no
# dataset sibling), e.g. Moorish Idol, Chinese Trumpetfish, Bluefin Trevally,
# Bluestripe Snapper, Spotted Boxfish, Peacock Grouper.

SPECIES_TO_FAMILY = {}
for fam, members in FAMILY_GROUPS.items():
    for m in members:
        SPECIES_TO_FAMILY[m] = fam


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics-csv", required=True, type=Path)
    p.add_argument("--data-yaml", type=Path, default=None,
                   help="Optional: data.yaml to resolve numeric class IDs to names if the metrics CSV "
                        "reports IDs instead of names (usually not needed -- Ultralytics reports names).")
    p.add_argument("--out-dir", type=Path, default=Path("family_confusion_analysis"))
    p.add_argument("--class-col", type=str, default=None)
    p.add_argument("--instances-col", type=str, default=None)
    p.add_argument("--precision-col", type=str, default=None)
    p.add_argument("--recall-col", type=str, default=None)
    p.add_argument("--map50-col", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    id_to_name = load_names_from_yaml(args.data_yaml) if args.data_yaml and args.data_yaml.exists() else {}

    with open(args.metrics_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{args.metrics_csv} has no data rows")
    columns = list(rows[0].keys())

    class_col = autodetect(columns, args.class_col, ["class"])
    instances_col = autodetect(columns, args.instances_col, ["instance"])
    map50_col = autodetect(columns, args.map50_col, ["map50"], exclude_patterns=["map50-95", "map5095"])
    recall_col = autodetect(columns, args.recall_col, ["recall", "-r", "_r"], exclude_patterns=["precision"])
    precision_col = autodetect(columns, args.precision_col, ["precision", "-p", "_p"])
    if not all([class_col, instances_col, recall_col]):
        raise SystemExit(f"Could not auto-detect required columns from {columns} -- "
                          f"override with --class-col/--instances-col/--recall-col.")

    data = []
    unrecognized = []
    for r in rows:
        raw = r[class_col]
        try:
            name = id_to_name.get(int(float(raw)), str(raw))
        except (TypeError, ValueError):
            name = str(raw)
        family = SPECIES_TO_FAMILY.get(name, f"(singleton) {name}")
        if name not in SPECIES_TO_FAMILY and not name.startswith("("):
            unrecognized.append(name)
        data.append({
            "class": name,
            "family": family,
            "instances": to_float(r[instances_col]),
            "precision": to_float(r[precision_col]) if precision_col else np.nan,
            "recall": to_float(r[recall_col]),
            "map50": to_float(r[map50_col]) if map50_col else np.nan,
        })

    if unrecognized:
        print(f"NOTE: {len(unrecognized)} classes in the CSV weren't matched to any hardcoded family group "
              f"(treated as singletons) -- check for name mismatches: {unrecognized}\n")

    # ---- fit recall ~ instances (simple linear model), compute residuals ----
    valid = [d for d in data if not np.isnan(d["recall"]) and not np.isnan(d["instances"])]
    if len(valid) >= 3:
        xs = np.array([d["instances"] for d in valid])
        ys = np.array([d["recall"] for d in valid])
        slope, intercept = np.polyfit(xs, ys, 1)
        for d in data:
            if not np.isnan(d["recall"]) and not np.isnan(d["instances"]):
                predicted = slope * d["instances"] + intercept
                d["residual"] = d["recall"] - predicted
            else:
                d["residual"] = np.nan
        print(f"fitted recall ~ {slope:.6f} * instances + {intercept:.3f} across {len(valid)} classes\n")
    else:
        for d in data:
            d["residual"] = np.nan

    # ---- group by family ----
    families = {}
    for d in data:
        families.setdefault(d["family"], []).append(d)

    fam_summary = []
    for fam, members in families.items():
        recs = [m["recall"] for m in members if not np.isnan(m["recall"])]
        insts = [m["instances"] for m in members if not np.isnan(m["instances"])]
        resids = [m["residual"] for m in members if not np.isnan(m["residual"])]
        fam_summary.append({
            "family": fam,
            "n_members_in_run": len(members),
            "n_members_measured": len(recs),
            "avg_recall": float(np.mean(recs)) if recs else np.nan,
            "avg_instances": float(np.mean(insts)) if insts else np.nan,
            "avg_residual": float(np.mean(resids)) if resids else np.nan,
        })

    fam_summary.sort(key=lambda f: (np.nan_to_num(f["avg_recall"], nan=999)))

    print("=== per-family summary, worst avg recall first ===")
    print(f"{'family':<45} {'#classes':>9} {'avg_inst':>9} {'avg_recall':>11} {'avg_residual':>13}")
    for f in fam_summary:
        print(f"{f['family']:<45} {f['n_members_in_run']:>9} {f['avg_instances']:>9.1f} "
              f"{f['avg_recall']:>11.3f} {f['avg_residual']:>13.3f}")

    print("\n=== per-class detail within each family (worst family first) ===")
    for f in fam_summary:
        print(f"\n-- {f['family']} --")
        for m in sorted(families[f["family"]], key=lambda d: d["recall"] if not np.isnan(d["recall"]) else -1):
            res_str = f"{m['residual']:+.3f}" if not np.isnan(m["residual"]) else "n/a"
            print(f"    {m['class']:<32} instances={m['instances']:>6.0f}  recall={m['recall']:.3f}  "
                  f"residual={res_str}")

    # ---- does family size predict residual, controlling for instance count? ----
    multi = [f for f in fam_summary if f["n_members_in_run"] > 1 and not np.isnan(f["avg_residual"])]
    single = [f for f in fam_summary if f["n_members_in_run"] == 1 and not np.isnan(f["avg_residual"])]
    if multi and single:
        multi_avg = float(np.mean([f["avg_residual"] for f in multi]))
        single_avg = float(np.mean([f["avg_residual"] for f in single]))
        print(f"\n=== multi-member families vs. singleton-family classes ===")
        print(f"  avg residual, classes in a multi-species family (n={len(multi)} families): {multi_avg:+.3f}")
        print(f"  avg residual, singleton-family classes (n={len(single)}):                   {single_avg:+.3f}")
        if multi_avg < single_avg - 0.03:
            print(f"  -- supports the confusability hypothesis: classes sharing a family with other "
                  f"dataset species underperform what their instance count alone predicts, relative to "
                  f"visually/taxonomically 'alone' classes.")
        else:
            print(f"  -- does NOT clearly show a family-size penalty beyond instance count in this run.")

    if fam_summary and any(not np.isnan(f["avg_residual"]) for f in fam_summary):
        sizes = [f["n_members_in_run"] for f in fam_summary if not np.isnan(f["avg_residual"])]
        resids = [f["avg_residual"] for f in fam_summary if not np.isnan(f["avg_residual"])]
        if len(sizes) >= 3:
            corr = float(np.corrcoef(sizes, resids)[0, 1])
            print(f"\nPearson correlation (family size in this run vs. avg residual): {corr:.3f} "
                  f"({'supports' if corr < -0.3 else 'does not clearly support'} "
                  f"'bigger families -> worse than instance-count-predicted recall')")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        sizes_all = [f["n_members_in_run"] for f in fam_summary if not np.isnan(f["avg_residual"])]
        resids_all = [f["avg_residual"] for f in fam_summary if not np.isnan(f["avg_residual"])]
        labels_all = [f["family"] for f in fam_summary if not np.isnan(f["avg_residual"])]
        ax.scatter(sizes_all, resids_all, s=70)
        for x, y, lbl in zip(sizes_all, resids_all, labels_all):
            ax.annotate(lbl.split(" (")[0], (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("number of sibling species present in this run (family size)")
        ax.set_ylabel("avg residual (actual recall - instance-count-predicted recall)")
        ax.set_title("Family size vs. recall residual")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_dir / "family_size_vs_residual.png", dpi=150)
        print(f"\nwrote {args.out_dir / 'family_size_vs_residual.png'}")
    except ImportError:
        print("matplotlib not installed -- skipping plot")


if __name__ == "__main__":
    main()
