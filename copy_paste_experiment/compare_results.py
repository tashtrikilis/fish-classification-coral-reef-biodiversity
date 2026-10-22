#!/usr/bin/env python3
"""
Compare baseline vs copy-paste vs size-matched copy-paste after the training jobs finish.

Reads every runs/*/focus_metrics.json written by train_experiment.py. Each run records its
task (detect / segment), arm (baseline / copypaste / copypaste_scaled) and seed; round-1 runs
written before seeds were recorded count as seed 0. Runs of the same task + arm with different
seeds are pooled, so each arm is reported as mean ± standard deviation across seeds, with n.
Works with whatever runs exist so far.

Writes to runs/comparison/:
    comparison.csv   every split x task x part (box/mask) x species x metric x arm:
                     n runs, mean, sd, the individual values, and test instance counts
    comparison.md    TEST split, mean ± sd per arm and the change from baseline (for the thesis)
    comparison.png   test mAP50 and recall per arm: bars = mean, whiskers = sd, dots = single runs
and logs the table + chart to W&B as a run named "comparison".

"Other classes (mean)" is the check that copy-paste didn't hurt the species you didn't augment.
With 3 runs per arm the sd is itself rough, so treat a change smaller than the spread as no change.
"""
import argparse
import csv
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ["P", "R", "F1", "mAP50", "mAP50-95"]
ARMS = [("baseline", "Baseline"), ("copypaste", "Copy-paste"), ("copypaste_scaled", "Copy-paste, size-matched")]
REF_ARM = "baseline_ref"  # baseline runs from --ref-runs-dir (e.g. round 2 at imgsz 1280), shown as a reference column

# Validated categorical colours (blue / orange / aqua) + chart chrome
ARM_COLOURS = {REF_ARM: "#898781", "baseline": "#2a78d6", "copypaste": "#eb6834", "copypaste_scaled": "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    p.add_argument("--ref-runs-dir", type=Path, default=None,
                   help="Also load the BASELINE runs from this folder as a reference column, "
                        "e.g. --runs-dir runs_i2560 --ref-runs-dir runs (does resolution alone help?)")
    p.add_argument("--wandb-project", default="egh490-copypaste-experiment")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def load(runs_dir, ref_runs_dir=None):
    runs = defaultdict(list)  # (task, arm) -> [run json, ...]
    for f in sorted(runs_dir.glob("*/focus_metrics.json")):
        r = json.loads(f.read_text())
        r.setdefault("seed", 0)
        r.setdefault("imgsz", 1280)  # runs written before imgsz was recorded were all 1280
        runs[(r["task"], r["arm"])].append(r)
    if ref_runs_dir is not None:
        for f in sorted(ref_runs_dir.glob("*/focus_metrics.json")):
            r = json.loads(f.read_text())
            r.setdefault("seed", 0)
            r.setdefault("imgsz", 1280)
            if r["arm"] == "baseline":
                runs[(r["task"], REF_ARM)].append(r)
    for (task, arm), rs in sorted(runs.items()):
        print(f"{task:8s} {arm:17s} {len(rs)} run(s), seeds {sorted(r['seed'] for r in rs)}")
    return runs


def summarise(values):
    v = [x for x in values if x is not None]
    if not v:
        return None
    return {"n": len(v), "mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0, "values": v}


def build_rows(runs):
    rows = []
    for task in ("detect", "segment"):
        if not runs.get((task, "baseline")):
            continue
        ref = runs[(task, "baseline")][0]
        focus = ref["focus_species"]
        parts = ["box", "mask"] if task == "segment" else ["box"]
        for split in ("test", "val"):
            for part in parts:
                for sp in focus + ["others_mean", "all"]:
                    label = {"others_mean": "Other classes (mean)", "all": "All classes"}.get(sp, sp)
                    inst = ref["metrics"][split][sp].get("instances") if sp in focus else None
                    for met in METRICS:
                        for arm, _ in ARMS:
                            if not runs.get((task, arm)):
                                continue
                            vals = []
                            for r in runs.get((task, arm), []):
                                d = r["metrics"][split][sp].get(part)
                                vals.append(d.get(met) if d else None)
                            s = summarise(vals)
                            if s:
                                rows.append({"split": split, "task": task, "part": part, "species": label,
                                             "metric": met, "arm": arm, "n": s["n"], "mean": s["mean"], "sd": s["sd"],
                                             "values": ";".join(f"{x:.4f}" for x in s["values"]), "instances": inst})
    return rows


def lookup(rows, **k):
    for r in rows:
        if all(r[a] == b for a, b in k.items()):
            return r
    return None


def delta_label(a, labels):
    if a == "baseline" and REF_ARM in labels:
        return f"Δ vs {labels[REF_ARM].split(', ', 1)[-1]}"
    return {"copypaste": "Δ copy-paste", "copypaste_scaled": "Δ size-matched"}.get(a)


def write_md(rows, path, split="test", source=""):
    labels = dict(ARMS)
    present = [a for a, _ in ARMS if any(r["arm"] == a for r in rows)]
    head = ["Model", "Type", "Species", f"{split.capitalize()} inst.", "Metric"]
    for a in present:
        head.append(labels[a])
        if delta_label(a, labels) and (a != "baseline" or REF_ARM in present):
            head.append(delta_label(a, labels))
    note = ("Use this split to CHOOSE between variants; open the test table once, at the end."
            if split == "val" else "Final evaluation. Choose variants on the val table, not this one.")
    lines = [f"# Baseline vs copy-paste -- {split.upper()} split", "",
             (f"Runs from `{source}`. " if source else "")
             + "Mean ± standard deviation across seeds (n runs in brackets). Δ = change in the mean "
             "(copy-paste arms vs baseline; baseline vs the reference column).", "", note, "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    order = ["Forcepsfish", "Moorish Idol", "Other classes (mean)", "All classes"]
    keys = sorted({(r["task"], r["part"], r["species"], r["metric"]) for r in rows if r["split"] == split},
                  key=lambda k: (k[0], k[1], order.index(k[2]) if k[2] in order else 9, METRICS.index(k[3])))
    for task, part, sp, met in keys:
        if met == "F1":
            continue
        base = lookup(rows, split=split, task=task, part=part, species=sp, metric=met, arm="baseline")
        ref = lookup(rows, split=split, task=task, part=part, species=sp, metric=met, arm=REF_ARM)
        cells = [task, part, sp, str(base["instances"]) if base and base["instances"] is not None else "", met]
        for a in present:
            r = lookup(rows, split=split, task=task, part=part, species=sp, metric=met, arm=a)
            cells.append(f"{r['mean']:.3f} ± {r['sd']:.3f} ({r['n']})" if r else "n/a")
            if a == "baseline" and REF_ARM in present:
                cells.append(f"{r['mean'] - ref['mean']:+.3f}" if (r and ref) else "n/a")
            elif a in ("copypaste", "copypaste_scaled"):
                cells.append(f"{r['mean'] - base['mean']:+.3f}" if (r and base) else "n/a")
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def chart(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    panels = [(t, p) for t, p in (("detect", "box"), ("segment", "box"), ("segment", "mask"))
              if any(r["task"] == t and r["part"] == p for r in rows)]
    if not panels:
        return False
    arms = [(a, lab) for a, lab in ARMS if any(r["arm"] == a for r in rows)]
    species = []
    for r in rows:
        if r["split"] == "test" and r["species"] not in species and r["species"] != "All classes":
            species.append(r["species"])
    mets = [("mAP50", "mAP50"), ("R", "Recall")]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": MUTED})
    fig, axes = plt.subplots(len(mets), len(panels), figsize=(max(7.5, 4.6 * len(panels)), 3.6 * len(mets)),
                             squeeze=False, facecolor=SURFACE)
    x = np.arange(len(species))
    width = 0.8 / len(arms)
    for ri, (met, met_label) in enumerate(mets):
        for ci, (task, part) in enumerate(panels):
            ax = axes[ri][ci]
            ax.set_facecolor(SURFACE)
            for ai, (arm, lab) in enumerate(arms):
                off = (ai - (len(arms) - 1) / 2) * width
                for si, sp in enumerate(species):
                    r = lookup(rows, split="test", task=task, part=part, species=sp, metric=met, arm=arm)
                    if not r:
                        continue
                    xpos = x[si] + off
                    ax.bar(xpos, r["mean"], width * 0.92, color=ARM_COLOURS[arm], zorder=3, linewidth=0,
                           label=lab if si == 0 else None)
                    if r["n"] > 1:
                        ax.errorbar(xpos, r["mean"], yerr=r["sd"], color=INK, lw=1, capsize=2, zorder=4)
                    vals = [float(v) for v in r["values"].split(";")]
                    ax.scatter([xpos] * len(vals), vals, s=9, color=INK, zorder=5, linewidths=0)
                    ax.text(xpos, min(1.0, r["mean"] + (r["sd"] if r["n"] > 1 else 0)) + 0.02, f"{r['mean']:.2f}",
                            ha="center", va="bottom", fontsize=6.5, color=INK2)
            ax.set_ylim(0, 1.1)
            ax.set_xticks(x)

            def xlab(sp):
                b = lookup(rows, split="test", task=task, part=part, species=sp, metric=met, arm="baseline")
                n = b["instances"] if b else None
                return sp.replace(" (mean)", "\n(mean)") + (f"\nn={n} test" if n is not None else "")
            ax.set_xticklabels([xlab(sp) for sp in species])
            ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis="both", length=0)
            ax.set_title(f"{'Detection' if task == 'detect' else 'Segmentation'} model, {part} {met_label}",
                         loc="left", fontsize=10, color=INK, pad=8)
    axes[0][0].legend(frameon=False, loc="upper left", ncol=1, fontsize=8)
    fig.suptitle("Test split. Bars: mean across seeds, whiskers: sd, dots: single runs",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    runs = load(args.runs_dir, args.ref_runs_dir)
    global ARMS
    if args.ref_runs_dir is not None:
        def sizes(arm):
            return sorted({r["imgsz"] for (t, a), rs in runs.items() if a == arm for r in rs})
        main_sz, ref_sz = sizes("baseline"), sizes(REF_ARM)
        fmt = lambda v: "/".join(str(x) for x in v) or "?"
        ARMS = [(REF_ARM, f"Baseline, imgsz {fmt(ref_sz)}"), ("baseline", f"Baseline, imgsz {fmt(main_sz)}")] + ARMS[1:]
    rows = build_rows(runs)
    if not rows:
        print("no baseline runs found yet -- nothing to compare")
        return
    out = args.runs_dir / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_md(rows, out / "comparison.md", "test", args.runs_dir.name)
    write_md(rows, out / "comparison_val.md", "val", args.runs_dir.name)
    has_chart = chart(rows, out / "comparison.png")
    print((out / "comparison.md").read_text())
    print(f"wrote {out}/comparison.csv, comparison.md, comparison_val.md" + (", comparison.png" if has_chart else ""))

    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name="comparison" if args.runs_dir.name == "runs" else f"comparison_{args.runs_dir.name}",
                       group="comparison", job_type="analysis", tags=["comparison"])
            cols = list(rows[0].keys())
            wandb.log({"comparison_table": wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])})
            if has_chart:
                wandb.log({"comparison_chart": wandb.Image(str(out / "comparison.png"))})
            summ = {}
            for r in rows:
                if r["split"] != "test" or r["arm"] in ("baseline", REF_ARM):
                    continue
                b = lookup(rows, split="test", task=r["task"], part=r["part"], species=r["species"],
                           metric=r["metric"], arm="baseline")
                if b:
                    summ[f"delta/{r['arm']}/{r['task']}/{r['part']}/{r['species'].replace(' ', '_')}/{r['metric']}"] = \
                        r["mean"] - b["mean"]
            wandb.run.summary.update(summ)
            wandb.finish()
        except Exception as e:
            print(f"W&B logging skipped ({e}); results are in {out}")


if __name__ == "__main__":
    main()
