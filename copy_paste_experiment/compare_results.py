#!/usr/bin/env python3
"""
Compare baseline vs copy-paste after the training jobs finish.

Reads runs/<task>_<arm>/focus_metrics.json (written by train_experiment.py) for
task in {detect, segment} and arm in {baseline, copypaste}. Works with whatever
subset of the four runs exists. Writes to runs/comparison/:

    comparison.csv   every split x task x part (box/mask) x species x metric,
                     baseline, copy-paste, delta, and instance counts
    comparison.md    the TEST-set rows as a readable table (paste into the thesis)
    comparison.png   test-set mAP50 and recall, baseline vs copy-paste, for the
                     focus species and the mean over all other classes

and logs the same table + chart to W&B as a run named "comparison".

"Other classes (mean)" is the check that copy-paste didn't hurt the classes you
didn't augment. With one training run per arm, treat small differences (a few
mAP points), especially on species with ~20 test instances, as possibly noise.
"""
import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ["P", "R", "F1", "mAP50", "mAP50-95"]

# Validated categorical pair (blue / orange) + chart chrome
C_BASE, C_CP = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", type=Path, default=HERE / "runs")
    p.add_argument("--wandb-project", default="egh490-copypaste-experiment")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--no-wandb", action="store_true")
    return p.parse_args()


def load(runs_dir):
    runs = {}
    for task in ("detect", "segment"):
        for arm in ("baseline", "copypaste"):
            f = runs_dir / f"{task}_{arm}" / "focus_metrics.json"
            if f.exists():
                runs[(task, arm)] = json.loads(f.read_text())
            else:
                print(f"missing {f} -- skipping that run")
    return runs


def build_rows(runs):
    rows = []
    for task in ("detect", "segment"):
        b, c = runs.get((task, "baseline")), runs.get((task, "copypaste"))
        if not (b and c):
            continue
        focus = b["focus_species"]
        parts = ["box", "mask"] if task == "segment" else ["box"]
        for split in ("test", "val"):
            for part in parts:
                for sp in focus + ["others_mean", "all"]:
                    bd = b["metrics"][split][sp].get(part)
                    cd = c["metrics"][split][sp].get(part)
                    inst = b["metrics"][split][sp].get("instances") if sp in focus else None
                    label = {"others_mean": "Other classes (mean)", "all": "All classes"}.get(sp, sp)
                    for met in METRICS:
                        bv = bd.get(met) if bd else None
                        cv = cd.get(met) if cd else None
                        if bv is None and cv is None:
                            continue
                        rows.append({"split": split, "task": task, "part": part, "species": label,
                                     "metric": met, "baseline": bv, "copypaste": cv,
                                     "delta": (cv - bv) if (bv is not None and cv is not None) else None,
                                     "instances": inst})
    return rows


def fmt(v, signed=False):
    if v is None:
        return "n/a"
    return f"{v:+.3f}" if signed else f"{v:.3f}"


def write_md(rows, path):
    lines = ["# Baseline vs copy-paste -- TEST split", "",
             "| Model | Metric type | Species | Test inst. | Metric | Baseline | Copy-paste | Change |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["split"] != "test" or r["metric"] not in ("P", "R", "mAP50", "mAP50-95"):
            continue
        lines.append(f"| {r['task']} | {r['part']} | {r['species']} | {r['instances'] if r['instances'] is not None else ''} | "
                     f"{r['metric']} | {fmt(r['baseline'])} | {fmt(r['copypaste'])} | {fmt(r['delta'], True)} |")
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
    mets = [("mAP50", "mAP50"), ("R", "Recall")]
    species = []
    for r in rows:
        if r["split"] == "test" and r["species"] not in species and r["species"] != "All classes":
            species.append(r["species"])

    def val(task, part, sp, met, arm):
        for r in rows:
            if (r["split"], r["task"], r["part"], r["species"], r["metric"]) == ("test", task, part, sp, met):
                return r[arm]
        return None

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": MUTED})
    fig, axes = plt.subplots(len(mets), len(panels), figsize=(4.2 * len(panels), 3.4 * len(mets)),
                             squeeze=False, facecolor=SURFACE)
    x = np.arange(len(species))
    bw = 0.36
    for ri, (met, met_label) in enumerate(mets):
        for ci, (task, part) in enumerate(panels):
            ax = axes[ri][ci]
            ax.set_facecolor(SURFACE)
            for off, arm, color, lab in ((-bw / 2 - 0.01, "baseline", C_BASE, "Baseline"),
                                         (bw / 2 + 0.01, "copypaste", C_CP, "Copy-paste")):
                vals = [val(task, part, sp, met, arm) for sp in species]
                hts = [v if v is not None else 0 for v in vals]
                bars = ax.bar(x + off, hts, bw, color=color, label=lab, zorder=3, linewidth=0)
                for rect, v in zip(bars, vals):
                    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.015,
                            "n/a" if v is None else f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
            ax.set_ylim(0, 1.08)
            ax.set_xticks(x)
            def xlab(sp):
                n = next((r["instances"] for r in rows if r["split"] == "test" and r["task"] == task
                          and r["species"] == sp and r["instances"] is not None), None)
                return sp.replace(" (mean)", "\n(mean)") + (f"\nn={n} test" if n is not None else "")
            ax.set_xticklabels([xlab(sp) for sp in species])
            ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", length=0)
            title = f"{'Detection' if task == 'detect' else 'Segmentation'} model, {part} {met_label}"
            ax.set_title(title, loc="left", fontsize=10, color=INK, pad=8)
    axes[0][0].legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5)
    fig.suptitle("Test split: baseline vs copy-paste", x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    runs = load(args.runs_dir)
    rows = build_rows(runs)
    if not rows:
        print("need at least one complete baseline + copy-paste pair -- nothing to compare yet")
        return
    out = args.runs_dir / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_md(rows, out / "comparison.md")
    has_chart = chart(rows, out / "comparison.png")
    print((out / "comparison.md").read_text())
    print(f"wrote {out}/comparison.csv, comparison.md" + (", comparison.png" if has_chart else ""))

    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name="comparison",
                       group="comparison", job_type="analysis", tags=["comparison"])
            cols = list(rows[0].keys())
            wandb.log({"comparison_table": wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])})
            if has_chart:
                wandb.log({"comparison_chart": wandb.Image(str(out / "comparison.png"))})
            wandb.run.summary.update({
                f"delta/{r['task']}/{r['part']}/{r['species'].replace(' ', '_')}/{r['metric']}": r["delta"]
                for r in rows if r["split"] == "test" and r["delta"] is not None})
            wandb.finish()
        except Exception as e:
            print(f"W&B logging skipped ({e}); results are in {out}")


if __name__ == "__main__":
    main()
