#!/usr/bin/env python3
"""
5-fold temporal block cross-validation of baseline vs copy-paste (detection and segmentation data).

Why: consecutive frames show the same fish, and a single train/val/test split holds only a few
individual fish of each focus species. Cross-validation tests every sighting once.

What this builds
----------------
Frames --min-frame..--max-frame (default 1000-2461, the CVAT-labelled range; unlabelled frames in
range are labelled background) are cut into --block-size (30-frame, ~1 min) blocks. Blocks are dealt
into --folds groups; the assignment is chosen from --fold-search random candidates to balance the
focus species and the overall instance counts across folds. It's chosen on LABEL COUNTS ONLY, never
on results.

For fold k:  test = group k,  val = every other block of group k+1 (about half a group; used
             only to pick the best epoch), train = everything else, minus a --buffer-frames band
             (default 15) around every TEST frame, so the same fish seconds apart can't sit in
             train and test. (Val has no buffer: a val/train neighbour can only nudge which epoch
             is kept, never the test score.)
Every frame is in exactly one test set, so pooling the 5 test sets covers the whole range once.

Two arms per fold, identical except for the extra synthetic training images:
  baseline   real train frames only
  copypaste  real train frames + --synth-per-species synthetic images per focus species, each with
             1..--max-pastes pasted fish of that species:
               donors     Forcepsfish: the visually checked donors in experiment_donors.csv;
                          other species: cv_donor_candidates.csv minus the candidates excluded at
                          the visual quality check (cv_donor_veto.txt). See README.md,
                          "Donor and paste-frame selection".
                          A donor is used in fold k only if its frame is in fold k's train set
                          or outside the frame range entirely (never a val/test fish).
               size       target size drawn from that species' REAL train instances in fold k
                          (shorter box side); donors are only shrunk (INTER_AREA), never enlarged.
               context    a donor is pasted into a train frame of the SAME camera view as its own
                          frame (reef view vs pole/open-water/turning, from the camera-view labels in
                          approved_acceptor_frames.csv;
                          frames <1000 count as pole view). Hazy donors go into hazy scenes, clear
                          into clear, so pastes look like the real fish in that view.
               transform  +/- --max-rotation degrees, horizontal flip 0.5, no overlap with real fish.
               blend      edge-only Poisson (seamlessClone in a 1-3 px ring, donor pixels inside).

Output (--out, default data_cv/):
  folds.csv                       frame, block, fold (the test fold of every frame)
  cv_summary.json                 fold assignment, counts per fold, donors, settings
  fold<k>/det_{baseline,copypaste}/{images,labels}/{train,val,test}
  fold<k>/det_{baseline,copypaste}.yaml, fold<k>/prepare_summary.json, fold<k>/synth_report.csv
  qa_cv_synthetic_fold0.jpg       random pasted fish (fold 0), zoomed, with the written label box

Usage (from copy_paste_experiment/):
    python3 cv_prepare.py --images-dir ../full_dataset --folds-only     # check the folds, no images written
    python3 cv_prepare.py --images-dir ../full_dataset --force
"""
import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from prepare_experiment import (read_names, read_export_labels, poly_bbox_px, iou, det_line, seg_line,
                                paste_poisson_edge, random_transform, find_free_spot,
                                mask_to_polygon_norm, link_or_copy)

HERE = Path(__file__).resolve().parent
W_IMG, H_IMG = 1280, 720


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--labels-dir", type=Path, default=HERE / "labels_export" / "labels" / "train")
    p.add_argument("--names-yaml", type=Path, default=HERE / "labels_export" / "data.yaml")
    p.add_argument("--approved-donors", type=Path, default=HERE / "experiment_donors.csv")
    p.add_argument("--candidates", type=Path, default=HERE / "cv_donor_candidates.csv")
    p.add_argument("--veto", type=Path, default=HERE / "cv_donor_veto.txt")
    p.add_argument("--acceptor-csv", type=Path, default=HERE / "approved_acceptor_frames.csv")
    p.add_argument("--out", type=Path, default=HERE / "data_cv")
    p.add_argument("--species", default="Forcepsfish,Orangespine Unicornfish,Convict Tang")
    p.add_argument("--hand-approved-species", default="Forcepsfish",
                   help="Species whose donors come from the --approved-donors list; the rest come from "
                        "--candidates minus the exclusions in --veto.")
    p.add_argument("--min-frame", type=int, default=1000)
    p.add_argument("--max-frame", type=int, default=2461)
    p.add_argument("--block-size", type=int, default=30)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--buffer-frames", type=int, default=15)
    p.add_argument("--fold-search", type=int, default=3000)
    p.add_argument("--fold-seed", type=int, default=0)
    p.add_argument("--synth-per-species", type=int, default=150)
    p.add_argument("--max-pastes", type=int, default=3)
    p.add_argument("--max-rotation", type=float, default=20.0)
    p.add_argument("--hflip-prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--qa-samples", type=int, default=24)
    p.add_argument("--folds-only", action="store_true", help="Only choose + report the folds; write nothing.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-seg", action="store_true", help="Build detection roots only (the round-4/5 layout)")
    return p.parse_args()


# --------------------------------------------------------------------------- folds

def choose_folds(blocks, block_counts, focus_ids, other_ids, n_folds, n_search, seed):
    """Deal blocks into n_folds groups of (near) equal size; keep the candidate whose per-fold
    instance counts are most even for the focus species, the bigger other classes and overall."""
    rng = random.Random(seed)
    tot = Counter()
    for b in blocks:
        tot.update(block_counts[b])
    n_all = sum(tot.values())

    def score(assign):
        per = [Counter() for _ in range(n_folds)]
        for b, k in assign.items():
            per[k].update(block_counts[b])
        s = 0.0
        for c, w in [(c, 1.0) for c in focus_ids] + [(c, 0.25) for c in other_ids]:
            t = tot[c] / n_folds
            if t == 0:
                continue
            s += w * sum((per[k][c] - t) ** 2 for k in range(n_folds)) / t ** 2
            s += w * 2.0 * sum(per[k][c] == 0 for k in range(n_folds))  # every fold should test it
        t = n_all / n_folds
        s += sum((sum(per[k].values()) - t) ** 2 for k in range(n_folds)) / t ** 2
        return s

    best, best_s, best_i = None, None, None
    for i in range(n_search):
        order = blocks[:]
        rng.shuffle(order)
        assign = {b: j % n_folds for j, b in enumerate(order)}
        sc = score(assign)
        if best_s is None or sc < best_s:
            best, best_s, best_i = assign, sc, i
    return best, best_s, best_i


def main():
    args = parse_args()
    global KINDS
    KINDS = ("det",) if args.no_seg else ("det", "seg")
    names = read_names(args.names_yaml)
    name_to_id = {v: k for k, v in names.items()}
    focus = [s.strip() for s in args.species.split(",") if s.strip()]
    if any(s not in name_to_id for s in focus):
        sys.exit(f"species not in class list: {[s for s in focus if s not in name_to_id]}")
    focus_ids = [name_to_id[s] for s in focus]
    hand = {s.strip() for s in args.hand_approved_species.split(",") if s.strip()}

    labels = read_export_labels(args.labels_dir)
    frames = list(range(args.min_frame, args.max_frame + 1))
    block_of = {f: (f - args.min_frame) // args.block_size for f in frames}
    blocks = sorted(set(block_of.values()))
    block_counts = defaultdict(Counter)
    for f in frames:
        block_counts[block_of[f]].update(c for c, _ in labels.get(f, []))
    totals = Counter()
    for b in blocks:
        totals.update(block_counts[b])
    other_ids = [c for c, n in totals.items() if n >= 50 and c not in focus_ids]

    assign, sc, cand = choose_folds(blocks, block_counts, focus_ids, other_ids, args.folds,
                                    args.fold_search, args.fold_seed)
    fold_of = {f: assign[block_of[f]] for f in frames}

    # per fold: test = group k; val = every other block of group k+1 (about half a group; it only
    # picks the best epoch); train = everything else minus a buffer band around the TEST frames.
    sets = {}
    for k in range(args.folds):
        test = [f for f in frames if fold_of[f] == k]
        nxt = sorted(b for b in blocks if assign[b] == (k + 1) % args.folds)
        val_blocks = set(nxt[::2])
        val = [f for f in frames if block_of[f] in val_blocks]
        held = np.array(sorted(test))
        train, dropped = [], []
        for f in frames:
            if fold_of[f] == k or block_of[f] in val_blocks:
                continue
            d = np.min(np.abs(held - f))
            (dropped if d <= args.buffer_frames else train).append(f)
        sets[k] = {"train": train, "val": val, "test": test, "buffer_dropped": dropped}

    def count(fs, cid):
        return sum(1 for f in fs for c, _ in labels.get(f, []) if c == cid)

    print(f"frames {args.min_frame}-{args.max_frame}: {len(frames)} in {len(blocks)} blocks of {args.block_size}; "
          f"{args.folds} folds (candidate {cand} of {args.fold_search}, balance score {sc:.2f})")
    print(f"{'':28s}" + "".join(f"  fold{k} tr/va/te " for k in range(args.folds)))
    for s, cid in zip(focus, focus_ids):
        print(f"  {s:26s}" + "".join(f"  {count(sets[k]['train'], cid):4d}/{count(sets[k]['val'], cid):3d}/{count(sets[k]['test'], cid):3d}   "
                                      for k in range(args.folds)))
    print(f"  {'frames':26s}" + "".join(f"  {len(sets[k]['train']):4d}/{len(sets[k]['val']):3d}/{len(sets[k]['test']):3d}   "
                                         for k in range(args.folds)))
    print(f"  {'(buffer frames dropped)':26s}" + "".join(f"  {len(sets[k]['buffer_dropped']):14d}   " for k in range(args.folds)))

    # ---- donors
    donors = []
    with open(args.approved_donors) as fh:
        for r in csv.DictReader(fh):
            if r["species"] not in focus or r["species"] not in hand:
                continue
            cid, fr = name_to_id[r["species"]], int(r["frame"])
            box = tuple(float(r[k]) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
            cands = [(iou(box, poly_bbox_px(p, W_IMG, H_IMG)), p) for c, p in labels.get(fr, []) if c == cid]
            best = max(cands, key=lambda t: t[0]) if cands else (0, None)
            if best[0] >= 0.8:
                donors.append({"species": r["species"], "cid": cid, "frame": fr, "poly": best[1], "source": "donor_list"})
    veto = defaultdict(set)
    keep_only = {}
    if args.veto.exists():
        for line in args.veto.read_text().splitlines():
            parts = [x.strip() for x in line.split(",") if x.strip()]
            if len(parts) < 2 or line.strip().startswith("#"):
                continue
            if parts[1] == "all-except":
                keep_only[parts[0]] = {int(x) for x in parts[2:]}
            else:
                veto[parts[0]].update(int(x) for x in parts[1:])
    n_veto = Counter()
    if args.candidates.exists():
        with open(args.candidates) as fh:
            for r in csv.DictReader(fh):
                s = r["species"]
                if s not in focus or s in hand:
                    continue
                idx = int(r["idx"])
                if idx in veto[s] or (s in keep_only and idx not in keep_only[s]):
                    n_veto[s] += 1
                    continue
                fr, inst = int(r["frame"]), int(r["instance"])
                c, poly = labels[fr][inst]
                donors.append({"species": s, "cid": c, "frame": fr, "poly": poly, "source": f"auto#{idx}"})
    for s in focus:
        n = sum(1 for d in donors if d["species"] == s)
        print(f"  donors {s:26s} {n:3d}" + (f"  ({n_veto[s]} excluded at quality check)" if n_veto[s] else ""))

    allowed_view = {}
    with open(args.acceptor_csv) as fh:
        for r in csv.DictReader(fh):
            allowed_view[int(r["frame"])] = r["allowed_as_acceptor"] == "1"
    view = lambda f: "reef" if allowed_view.get(f, False) else "pole"  # <1000 and unreviewed = pole view

    if args.folds_only:
        return

    needed = set(frames) | {d["frame"] for d in donors}
    missing = sorted(f for f in needed if not (args.images_dir / f"frame_{f}.jpg").exists())
    if missing:
        sys.exit(f"{len(missing)} frames missing from {args.images_dir}, e.g. {missing[:10]}")
    if args.out.exists():
        if not args.force:
            sys.exit(f"{args.out} exists -- pass --force to rebuild")
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    with open(args.out / "folds.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "block", "fold", "view"])
        for f in frames:
            wr.writerow([f, block_of[f], fold_of[f], view(f)])

    img_cache = {}

    def load_rgb(f):
        if f not in img_cache:
            img_cache[f] = cv2.cvtColor(cv2.imread(str(args.images_dir / f"frame_{f}.jpg")), cv2.COLOR_BGR2RGB)
        return img_cache[f]

    names_block = "names:\n" + "".join(f"  {k}: {names[k]}\n" for k in sorted(names))
    summary = {"settings": {k: str(v) for k, v in vars(args).items()}, "fold_candidate": cand,
               "fold_balance_score": sc, "focus_species": focus, "folds": {}}
    qa_rows = []
    for k in range(args.folds):
        fd = args.out / f"fold{k}"
        # one dataset root per task x arm (Ultralytics keeps a label cache per folder; never share them)
        for kind in KINDS:
            for arm in ("baseline", "copypaste"):
                for sub in ("train", "val", "test"):
                    (fd / f"{kind}_{arm}" / "images" / sub).mkdir(parents=True, exist_ok=True)
                    (fd / f"{kind}_{arm}" / "labels" / sub).mkdir(parents=True, exist_ok=True)
        for sub in ("train", "val", "test"):
            for f in sets[k][sub]:
                for kind in KINDS:
                    fmt = det_line if kind == "det" else seg_line
                    txt = "".join(fmt(c, p) + "\n" for c, p in labels.get(f, []))
                    for arm in ("baseline", "copypaste"):
                        link_or_copy(args.images_dir / f"frame_{f}.jpg", fd / f"{kind}_{arm}" / "images" / sub / f"frame_{f}.jpg", False)
                        (fd / f"{kind}_{arm}" / "labels" / sub / f"frame_{f}.txt").write_text(txt)

        # ---- synthetic images for this fold
        train_set = set(sets[k]["train"])
        in_range = lambda f: args.min_frame <= f <= args.max_frame
        usable = {s: [d for d in donors if d["species"] == s and ((not in_range(d["frame"])) or d["frame"] in train_set)]
                  for s in focus}
        acceptors = {v: [f for f in sets[k]["train"] if view(f) == v] for v in ("reef", "pole")}
        train_sizes = {s: [min(poly_bbox_px(p, W_IMG, H_IMG)[2:]) for f in sets[k]["train"]
                           for c, p in labels.get(f, []) if c == cid] for s, cid in zip(focus, focus_ids)}
        rng = random.Random(args.seed + 1000 * k)
        sdir = fd / "synth_images"
        sdir.mkdir()
        report, n_img, n_fish = [], Counter(), Counter()
        for s, cid in zip(focus, focus_ids):
            pool = usable[s]
            if not pool or not train_sizes[s]:
                print(f"  fold{k} {s}: no usable donors or no train instances -- skipped")
                continue
            order = []
            while len(order) < args.synth_per_species * args.max_pastes:
                b = pool[:]
                rng.shuffle(b)
                order.extend(b)
            oi = 0
            for j in range(args.synth_per_species):
                n_paste = rng.randint(1, args.max_pastes)
                first = order[oi]
                tv = view(first["frame"])
                tf = rng.choice(acceptors[tv] or acceptors["reef"] or acceptors["pole"])
                out = load_rgb(tf).copy()
                boxes = [poly_bbox_px(p, W_IMG, H_IMG) for _, p in labels.get(tf, [])]
                new_polys, used = [], []
                for _ in range(n_paste):
                    d = order[oi % len(order)]
                    oi += 1
                    if view(d["frame"]) != tv:
                        continue  # keep every paste in this image from the same camera view
                    src = load_rgb(d["frame"])
                    x0, y0, bw, bh = (int(round(v)) for v in poly_bbox_px(d["poly"], W_IMG, H_IMG))
                    x1, y1 = min(x0 + bw + 1, W_IMG), min(y0 + bh + 1, H_IMG)
                    full = np.zeros((H_IMG, W_IMG), np.uint8)
                    cv2.fillPoly(full, [np.round(d["poly"] * [W_IMG, H_IMG]).astype(np.int32)], 1)
                    patch, mask = src[y0:y1, x0:x1], full[y0:y1, x0:x1]
                    p2, m2, angle, flipped = random_transform(patch, mask, rng, args.max_rotation, args.hflip_prob)
                    target_px = rng.choice(train_sizes[s])
                    scale = min(1.0, target_px / max(1, min(m2.shape)))
                    if scale < 1.0:
                        nw, nh = max(1, round(m2.shape[1] * scale)), max(1, round(m2.shape[0] * scale))
                        p2 = cv2.resize(p2, (nw, nh), interpolation=cv2.INTER_AREA)
                        m2 = (cv2.resize(m2.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
                        ys, xs = np.where(m2 > 0)
                        if len(ys) < 6:
                            continue
                        p2 = np.ascontiguousarray(p2[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
                        m2 = np.ascontiguousarray(m2[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
                    spot = find_free_spot(rng, out.shape, m2.shape, boxes)
                    if spot is None:
                        continue
                    x, y = spot
                    out, fb = paste_poisson_edge(out, p2, m2, x, y, ring_px=float(np.clip(0.1 * min(m2.shape), 1.0, 3.0)))
                    poly = mask_to_polygon_norm(m2, x, y, W_IMG, H_IMG)
                    if poly is None:
                        continue
                    boxes.append((x, y, m2.shape[1], m2.shape[0]))
                    new_polys.append(poly)
                    used.append((d, angle, flipped, scale, target_px, x, y, m2.shape[1], m2.shape[0], fb))
                if not new_polys:
                    continue
                stem = f"synthCV_{s.replace(' ', '_')}_{j:04d}_t{tf}"
                ip = sdir / f"{stem}.jpg"
                cv2.imwrite(str(ip), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                for kind in KINDS:
                    fmt = det_line if kind == "det" else seg_line
                    link_or_copy(ip, fd / f"{kind}_copypaste" / "images" / "train" / f"{stem}.jpg", False)
                    lines = [fmt(c, p) for c, p in labels.get(tf, [])] + [fmt(cid, p) for p in new_polys]
                    (fd / f"{kind}_copypaste" / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                n_img[s] += 1
                n_fish[s] += len(new_polys)
                for (d, angle, flipped, scale, tpx, x, y, w, h, fb), poly in zip(used, new_polys):
                    report.append({"synth": stem, "species": s, "donor_frame": d["frame"], "donor_source": d["source"],
                                   "target_frame": tf, "view": tv, "angle": round(angle, 1), "hflip": int(flipped),
                                   "scale": round(scale, 3), "target_size_px": round(tpx, 1),
                                   "x": x, "y": y, "w": w, "h": h, "poisson_fallback": int(fb)})
                if k == 0:
                    qa_rows.append((ip, [(x, y, w, h) for (_, _, _, _, _, x, y, w, h, _) in used], s))
        with open(fd / "synth_report.csv", "w", newline="") as fh:
            if report:
                wr = csv.DictWriter(fh, fieldnames=list(report[0].keys()))
                wr.writeheader()
                wr.writerows(report)

        for kind, arm, desc in [(kd, a, d) for kd in KINDS for a, d in (
                ("baseline", "real frames only"),
                ("copypaste", "real frames + size-matched, view-matched copy-paste images"))]:
            root = (fd / f"{kind}_{arm}").resolve()
            (fd / f"{kind}_{arm}.yaml").write_text(f"# fold {k} / {kind} / {arm} -- built by cv_prepare.py (train = {desc})\n"
                                                f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                                                + names_block)
        fsum = {"fold": k, "focus_species": focus, "frames": {sub: len(sets[k][sub]) for sub in ("train", "val", "test", "buffer_dropped")},
                "focus": {s: {"class_id": cid,
                              "real_instances": {sub: count(sets[k][sub], cid) for sub in ("train", "val", "test")},
                              "donors_usable": len(usable[s]), "synthetic_images": n_img[s], "pasted_fish": n_fish[s]}
                          for s, cid in zip(focus, focus_ids)}}
        (fd / "prepare_summary.json").write_text(json.dumps(fsum, indent=2))
        summary["folds"][k] = fsum
        print(f"  fold{k}: " + ", ".join(f"{s} {n_img[s]} images / {n_fish[s]} fish ({len(usable[s])} donors)" for s in focus))

    (args.out / "cv_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- QA sheet: zoomed crops around pasted fish with the box read back from the label file
    rng = random.Random(0)
    picks = rng.sample(qa_rows, min(args.qa_samples, len(qa_rows))) if qa_rows else []
    tiles = []
    for ip, bxs, s in picks:
        img = cv2.imread(str(ip))
        lab = (args.out / "fold0" / "det_copypaste" / "labels" / "train" / (ip.stem + ".txt")).read_text().split("\n")
        lab = [l.split() for l in lab if l.strip()][-len(bxs):]
        for l in lab:
            cx, cy, w, h = (float(v) for v in l[1:5])
            cv2.rectangle(img, (int((cx - w / 2) * W_IMG) - 2, int((cy - h / 2) * H_IMG) - 2),
                          (int((cx + w / 2) * W_IMG) + 2, int((cy + h / 2) * H_IMG) + 2), (255, 0, 255), 1)
        x, y, w, h = bxs[0]
        r = int(max(40, 2.2 * max(w, h)))
        cx, cy = x + w // 2, y + h // 2
        x0, y0 = min(max(0, cx - r), W_IMG - 2 * r), min(max(0, cy - r), H_IMG - 2 * r)
        crop = cv2.resize(img[y0:y0 + 2 * r, x0:x0 + 2 * r], (240, 240), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(crop, (0, 0), (240, 18), (0, 0, 0), -1)
        cv2.putText(crop, f"{s[:16]} {min(w, h)}px x{len(bxs)}", (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(crop)
    if tiles:
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
        cv2.imwrite(str(args.out / "qa_cv_synthetic_fold0.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
