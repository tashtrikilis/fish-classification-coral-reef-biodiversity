#!/usr/bin/env python3
"""
Build the baseline vs copy-paste experiment datasets (detection AND segmentation)
for the targeted copy-paste study on Forcepsfish + Moorish Idol.

What this does, in order
------------------------
1. Reads the CVAT "Ultralytics YOLO segmentation" export in labels_export/
   (polygons, one label file per labelled frame) and keeps frames
   --min-frame..--max-frame (default 1000-2461). A frame in that range with no
   label file is a labelled BACKGROUND frame (no fish) and is kept as such.

2. Splits frames into train / val / test using contiguous frame blocks
   (--block-size, default 30 frames = ~1 min of footage at 30 frames/min), so
   near-identical neighbouring frames don't land on both sides of the split.
   Leakage guard for copy-paste specifically: any block containing a donor frame,
   or within --guard-frames of one, is forced into TRAIN. So every held-out
   (val/test) frame is more than --guard-frames away from every donor, and the
   copy-paste arm can't get credit for recognising a test fish it was pasted
   from. The focus species are then deliberately given val + test blocks, and
   remaining val/test quota is filled from random blocks (seeded). Baseline and
   copy-paste use the IDENTICAL split.

3. Writes four YOLO dataset roots, one per training run: data/det_baseline,
   data/det_copypaste (bbox labels), data/seg_baseline, data/seg_copypaste (polygon
   labels), each with images/{train,val,test}. They're separate so parallel jobs never
   share Ultralytics' label cache files. Frames are symlinked to --images-dir by default
   (--copy-images to copy instead), so the duplication costs no disk.

4. Generates synthetic copy-paste training images (TRAIN ONLY), one pasted fish
   per image, --synth-per-species per focus species:
     - donors: only the manually approved instances in experiment_donors.csv,
       and only if their frame is a TRAIN frame or lies outside the dataset's
       frame range entirely (so it is never evaluated on).
     - acceptors (targets): any train frame that is an allowed camera view per
       approved_acceptor_frames.csv (no pole / open-water view). Pass
       --acceptor-same-species to also require the frame to already contain
       that species.
     - augmentation: random rotation in +/- --max-rotation degrees (continuous,
       NOT 90-degree steps) and a horizontal flip with --hflip-prob. No
       vertical flip and no rescaling.
     - placement: never overlapping any real instance's bbox in the target.
     - blend: Poisson (cv2.seamlessClone NORMAL_CLONE), falling back to the
       LAB-matched feathered blend if seamlessClone errors.
   The synthetic image keeps all of the target frame's real labels, plus one
   new label for the pasted fish (polygon for seg, bbox for det).
   Image files are written once to data/synth_images/ and linked into the copy-paste
   roots' train folders.

5. Writes the four data yamls (det_baseline, det_copypaste, seg_baseline,
   seg_copypaste) with absolute paths, plus:
     data/split.csv               frame -> split (+ block id)
     data/synth_report.csv        every paste: donor, target, angle, flip, x, y
     data/prepare_summary.json    counts used by train_experiment.py for W&B config
     data/qa_synthetic_samples.jpg  random synthetic samples, zoomed, with the
                                  pasted fish's LABEL polygon drawn from the
                                  written label file (checks alignment)

Usage (from copy_paste_experiment/):
    python3 prepare_experiment.py --images-dir /path/to/full_dataset
"""
import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True, type=Path, help="Folder of frame_<N>.jpg source frames.")
    p.add_argument("--labels-dir", type=Path, default=HERE / "labels_export" / "labels" / "train")
    p.add_argument("--names-yaml", type=Path, default=HERE / "labels_export" / "data.yaml")
    p.add_argument("--donors-csv", type=Path, default=HERE / "experiment_donors.csv")
    p.add_argument("--acceptor-csv", type=Path, default=HERE / "approved_acceptor_frames.csv")
    p.add_argument("--out", type=Path, default=HERE / "data")
    p.add_argument("--min-frame", type=int, default=1000)
    p.add_argument("--max-frame", type=int, default=2461)
    p.add_argument("--species", type=str, default="Forcepsfish,Moorish Idol")
    p.add_argument("--block-size", type=int, default=30)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--guard-frames", type=int, default=15)
    p.add_argument("--split-seed", type=int, default=2)
    p.add_argument("--synth-per-species", type=int, default=150)
    p.add_argument("--max-rotation", type=float, default=20.0)
    p.add_argument("--hflip-prob", type=float, default=0.5)
    p.add_argument("--acceptor-same-species", action="store_true",
                   help="Only paste into training frames that already contain the same species. "
                        "Default off: any training frame in an allowed camera view can be a target.")
    p.add_argument("--blend", choices=["poisson_edge", "poisson", "matched"], default="poisson_edge")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy-images", action="store_true", help="Copy frames instead of symlinking them.")
    p.add_argument("--qa-samples", type=int, default=24)
    p.add_argument("--force", action="store_true", help="Delete an existing --out folder first.")
    p.add_argument("--split-only", action="store_true",
                   help="Only compute + report the split (no images needed). For checking.")
    return p.parse_args()


# --------------------------------------------------------------------------- labels

def read_names(yaml_path):
    names = {}
    for line in yaml_path.read_text().splitlines():
        m = re.match(r"\s+(\d+):\s*(.+?)\s*$", line)
        if m:
            names[int(m.group(1))] = m.group(2).strip().strip("'\"")
    if not names:
        sys.exit(f"no class names found in {yaml_path}")
    return names


FRAME_RE = re.compile(r"mp4-(\d+)_jpg")


def read_export_labels(labels_dir):
    """frame -> list of (class_id, Nx2 normalised polygon array). Export file names look like
    nelha-trimmed_mp4-<frame>_jpg.rf.<hash>.txt"""
    out = {}
    for f in labels_dir.glob("*.txt"):
        m = FRAME_RE.search(f.name)
        if not m:
            continue
        rows = []
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append((int(parts[0]), np.array(parts[1:], dtype=np.float64).reshape(-1, 2)))
        out[int(m.group(1))] = rows
    if not out:
        sys.exit(f"no label files matching *mp4-<frame>_jpg* found in {labels_dir}")
    return out


def poly_bbox_px(poly, w, h):
    xy = poly * [w, h]
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def seg_line(cls, poly_norm):
    return f"{cls} " + " ".join(f"{v:.6f}" for v in np.clip(poly_norm, 0, 1).ravel())


def det_line(cls, poly_norm):
    x0, y0 = np.clip(poly_norm.min(0), 0, 1)
    x1, y1 = np.clip(poly_norm.max(0), 0, 1)
    return f"{cls} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"


# --------------------------------------------------------------------------- split

def make_split(frames, frame_classes, focus_ids, donor_frames, args):
    """Block split with a donor guard band. Returns frame -> 'train'|'val'|'test', block_of."""
    frames = sorted(frames)
    blocks = [frames[i:i + args.block_size] for i in range(0, len(frames), args.block_size)]
    block_of = {f: bi for bi, blk in enumerate(blocks) for f in blk}

    protected = set()
    for bi, blk in enumerate(blocks):
        if any(abs(f - d) <= args.guard_frames for f in blk for d in donor_frames):
            protected.add(bi)

    rng = random.Random(args.split_seed)
    assign = {bi: "train" for bi in protected}
    n_total = len(frames)
    quota = {"val": round(n_total * args.val_frac), "test": round(n_total * args.test_frac)}
    used = {"val": 0, "test": 0}

    # Give each focus species roughly val_frac / test_frac of its instances in val / test.
    for cid in focus_ids:
        inst_in_block = Counter()
        for bi, blk in enumerate(blocks):
            inst_in_block[bi] = sum(frame_classes[f].count(cid) for f in blk)
        total = sum(inst_in_block.values())
        want = {"val": args.val_frac * total, "test": args.test_frac * total}
        have = {s: sum(inst_in_block[bi] for bi, a in assign.items() if a == s) for s in ("val", "test")}
        cands = [bi for bi in range(len(blocks)) if bi not in assign and inst_in_block[bi] > 0]
        rng.shuffle(cands)
        for bi in cands:
            need = [s for s in ("test", "val") if have[s] < want[s]]
            if not need:
                break
            s = need[0]
            assign[bi] = s
            have[s] += inst_in_block[bi]
            used[s] += len(blocks[bi])

    rest = [bi for bi in range(len(blocks)) if bi not in assign]
    rng.shuffle(rest)
    for bi in rest:
        for s in ("test", "val"):
            if used[s] < quota[s]:
                assign[bi] = s
                used[s] += len(blocks[bi])
                break
        else:
            assign[bi] = "train"

    split = {f: assign[block_of[f]] for f in frames}
    # hard check of the guard band
    held = [f for f, s in split.items() if s != "train"]
    for d in donor_frames:
        bad = [f for f in held if abs(f - d) <= args.guard_frames]
        assert not bad, f"guard band violated: donor frame {d} near held-out {bad[:5]}"
    return split, block_of, protected


# --------------------------------------------------------------------------- pasting
# Blending + augmentation are the same functions used for the visual review
# (blend_review_generate.py / blend_review_large_donors.py), copied here so this
# folder is self-contained on the HPC. Images are RGB numpy arrays.

def color_transfer_lab(patch, mask, target_region):
    patch_lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_region, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = mask.astype(bool)
    if m.sum() < 5:
        return patch
    out = patch_lab.copy()
    for c, s in enumerate((0.45, 0.15, 0.15)):
        src_vals = patch_lab[:, :, c][m]
        tgt_vals = target_lab[:, :, c]
        src_mean, src_std = src_vals.mean(), src_vals.std() + 1e-6
        tgt_mean, tgt_std = tgt_vals.mean(), tgt_vals.std() + 1e-6
        std_ratio = 1 + s * (np.clip(tgt_std / src_std, 0.8, 1.25) - 1)
        shift = np.clip(s * (tgt_mean - src_mean), -0.5 * src_std, 0.5 * src_std)
        out[:, :, c][m] = (src_vals - src_mean) * std_ratio + src_mean + shift
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def paste_matched_feathered(target, patch, mask, x, y):
    h, w = mask.shape
    feather_px = max(1, min(5, round(0.06 * min(h, w))))
    matched = color_transfer_lab(patch, mask, target[y:y + h, x:x + w])
    alpha = cv2.GaussianBlur((mask * 255).astype(np.uint8), (0, 0), sigmaX=feather_px).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    out = target.copy()
    region = out[y:y + h, x:x + w].astype(np.float32)
    out[y:y + h, x:x + w] = np.clip(region * (1 - alpha) + matched.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return out


def paste_poisson(target, patch, mask, x, y):
    """Returns (image, used_fallback)."""
    h, w = mask.shape
    mask255 = (mask * 255).astype(np.uint8)
    try:
        return cv2.seamlessClone(patch, target, mask255, (x + w // 2, y + h // 2), cv2.NORMAL_CLONE), False
    except cv2.error:
        return paste_matched_feathered(target, patch, mask, x, y), True


def paste_poisson_edge(target, patch, mask, x, y, ring_px=3.0):
    """Poisson only at the boundary: full cv2.seamlessClone result in a thin ring at the mask
    edge (seamless join), fading to the donor's ORIGINAL pixels inside. Plain Poisson
    re-derives the whole fish's colour from the surrounding background, which turned bright
    yellow Forcepsfish grey-blue on open-water targets and erased thin fins/filaments that the
    label still covers. Returns (image, used_fallback)."""
    pois, fb = paste_poisson(target, patch, mask, x, y)
    if fb:
        return pois, fb
    h, w = mask.shape
    m = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    alpha = (np.clip(dist / ring_px, 0, 1) * m)[..., None].astype(np.float32)
    region = pois[y:y + h, x:x + w].astype(np.float32)
    out = pois.copy()
    out[y:y + h, x:x + w] = np.clip(alpha * patch.astype(np.float32) + (1 - alpha) * region, 0, 255).astype(np.uint8)
    return out, False


def random_transform(patch, mask, rng, max_angle, hflip_prob):
    angle = rng.uniform(-max_angle, max_angle)
    h, w = mask.shape
    diag = int(np.ceil(np.hypot(h, w))) + 4
    pad_h, pad_w = (diag - h) // 2 + 2, (diag - w) // 2 + 2
    patch_p = cv2.copyMakeBorder(patch, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    mask_p = cv2.copyMakeBorder(mask, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    ph, pw = mask_p.shape
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    patch_r = cv2.warpAffine(patch_p, M, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_r = cv2.warpAffine(mask_p, M, (pw, ph), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.where(mask_r > 0)
    if len(ys) == 0:
        return patch, mask, 0.0, False
    patch_r = patch_r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    mask_r = mask_r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    flipped = rng.random() < hflip_prob
    if flipped:
        patch_r, mask_r = np.fliplr(patch_r), np.fliplr(mask_r)
    return np.ascontiguousarray(patch_r), np.ascontiguousarray(mask_r), angle, flipped


def find_free_spot(rng, img_shape, mask_shape, existing_boxes, margin=3, tries=60):
    th, tw = img_shape[:2]
    mh, mw = mask_shape
    if mh + 2 * margin >= th or mw + 2 * margin >= tw:
        return None
    for _ in range(tries):
        x = rng.randint(margin, tw - mw - margin)
        y = rng.randint(margin, th - mh - margin)
        if all(x + mw < bx or x > bx + bw or y + mh < by or y > by + bh for bx, by, bw, bh in existing_boxes):
            return x, y
    return None


def mask_to_polygon_norm(mask, x, y, w_img, h_img):
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, max(0.5, 0.004 * cv2.arcLength(c, True)), True).reshape(-1, 2).astype(np.float64)
    if len(c) < 3:
        return None
    c[:, 0] = (c[:, 0] + x) / w_img
    c[:, 1] = (c[:, 1] + y) / h_img
    return c


# --------------------------------------------------------------------------- main

def link_or_copy(src, dst, copy):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)


def main():
    args = parse_args()
    names = read_names(args.names_yaml)
    name_to_id = {v: k for k, v in names.items()}
    focus = [s.strip() for s in args.species.split(",") if s.strip()]
    missing_sp = [s for s in focus if s not in name_to_id]
    if missing_sp:
        sys.exit(f"species not in class names: {missing_sp}")
    focus_ids = [name_to_id[s] for s in focus]

    labels = read_export_labels(args.labels_dir)
    frames = list(range(args.min_frame, args.max_frame + 1))
    frame_classes = {f: [c for c, _ in labels.get(f, [])] for f in frames}
    W_IMG, H_IMG = 1280, 720  # source frame size; verified against real images below

    # ---- donors: match approved boxes to the export's polygons
    donors = []  # dicts: species, cid, frame, poly, source
    unmatched = []
    with open(args.donors_csv) as fh:
        for r in csv.DictReader(fh):
            if r["species"] not in focus:
                continue
            cid, fr = name_to_id[r["species"]], int(r["frame"])
            box = tuple(float(r[k]) for k in ("bbox_x", "bbox_y", "bbox_w", "bbox_h"))
            cands = [(iou(box, poly_bbox_px(poly, W_IMG, H_IMG)), poly)
                     for c, poly in labels.get(fr, []) if c == cid]
            best = max(cands, key=lambda t: t[0]) if cands else (0.0, None)
            if best[0] < 0.8:
                unmatched.append((r["species"], fr, round(best[0], 2)))
                continue
            donors.append({"species": r["species"], "cid": cid, "frame": fr, "poly": best[1], "source": r["source"]})
    if unmatched:
        print(f"WARNING: {len(unmatched)} approved donors had no matching polygon (IoU<0.8) and were skipped: {unmatched}")
    donor_frames_in_range = sorted({d["frame"] for d in donors if args.min_frame <= d["frame"] <= args.max_frame})

    split, block_of, protected = make_split(frames, frame_classes, focus_ids, donor_frames_in_range, args)

    # ---- report split
    n_split = Counter(split.values())
    inst = {s: Counter() for s in ("train", "val", "test")}
    for f in frames:
        for c in frame_classes[f]:
            inst[split[f]][c] += 1
    print(f"frames {args.min_frame}-{args.max_frame}: {len(frames)} "
          f"({sum(1 for f in frames if f in labels)} with labels, {sum(1 for f in frames if f not in labels)} background)")
    print(f"split (block={args.block_size}, guard={args.guard_frames}, seed={args.split_seed}): "
          f"train {n_split['train']}, val {n_split['val']}, test {n_split['test']} frames; "
          f"{len(protected)} donor-protected blocks forced to train")
    for s, cid in zip(focus, focus_ids):
        print(f"  {s:16s} instances  train {inst['train'][cid]:4d}  val {inst['val'][cid]:4d}  test {inst['test'][cid]:4d}")

    usable = defaultdict(list)
    for d in donors:
        in_range = args.min_frame <= d["frame"] <= args.max_frame
        if (not in_range) or split[d["frame"]] == "train":
            usable[d["species"]].append(d)
    for s in focus:
        n_all = sum(1 for d in donors if d["species"] == s)
        print(f"  {s:16s} donors usable {len(usable[s])}/{n_all} "
              f"(frames {sorted({d['frame'] for d in usable[s]})})")

    acceptor_ok = {}
    with open(args.acceptor_csv) as fh:
        for r in csv.DictReader(fh):
            acceptor_ok[int(r["frame"])] = r["allowed_as_acceptor"] == "1"
    targets = {s: [f for f in frames if split[f] == "train" and acceptor_ok.get(f, False)
                   and (cid in frame_classes[f] or not args.acceptor_same_species)]
               for s, cid in zip(focus, focus_ids)}
    rule = "train, allowed view" + (", contains species" if args.acceptor_same_species else "")
    for s in focus:
        print(f"  {s:16s} acceptor frames ({rule}): {len(targets[s])}")

    if args.split_only:
        return

    # ---- images
    needed = set(frames) | {d["frame"] for s in focus for d in usable[s]}
    missing = sorted(f for f in needed if not (args.images_dir / f"frame_{f}.jpg").exists())
    if missing:
        sys.exit(f"{len(missing)} needed frames are missing from {args.images_dir} "
                 f"(first few: {missing[:10]}). Transfer them before running this.")
    probe = cv2.imread(str(args.images_dir / f"frame_{frames[0]}.jpg"))
    if probe.shape[1] != W_IMG or probe.shape[0] != H_IMG:
        sys.exit(f"expected {W_IMG}x{H_IMG} frames, got {probe.shape[1]}x{probe.shape[0]}")

    if args.out.exists():
        if not args.force:
            sys.exit(f"{args.out} already exists -- pass --force to rebuild it")
        shutil.rmtree(args.out)
    # One dataset root per run (det/seg x baseline/copypaste). Ultralytics keeps a labels/*.cache
    # file per label folder and deletes/rewrites it when the image list changes, so two jobs running
    # in parallel must never share a label folder. Image files are links, so this costs no disk.
    ROOTS = [(kind, arm) for kind in ("det", "seg") for arm in ("baseline", "copypaste")]
    for kind, arm in ROOTS:
        for sub in ("train", "val", "test"):
            (args.out / f"{kind}_{arm}" / "images" / sub).mkdir(parents=True, exist_ok=True)
            (args.out / f"{kind}_{arm}" / "labels" / sub).mkdir(parents=True, exist_ok=True)
    (args.out / "synth_images").mkdir(parents=True, exist_ok=True)

    for f in frames:
        sub = split[f]
        rows = labels.get(f, [])
        src = args.images_dir / f"frame_{f}.jpg"
        for kind, arm in ROOTS:
            fmt = det_line if kind == "det" else seg_line
            link_or_copy(src, args.out / f"{kind}_{arm}" / "images" / sub / f"frame_{f}.jpg", args.copy_images)
            (args.out / f"{kind}_{arm}" / "labels" / sub / f"frame_{f}.txt").write_text(
                "".join(fmt(c, p) + "\n" for c, p in rows))

    with open(args.out / "split.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "split", "block", "donor_protected_block"])
        for f in frames:
            wr.writerow([f, split[f], block_of[f], int(block_of[f] in protected)])

    # ---- synthetic images
    rng = random.Random(args.seed)
    img_cache = {}

    def load_rgb(f):
        if f not in img_cache:
            img_cache[f] = cv2.cvtColor(cv2.imread(str(args.images_dir / f"frame_{f}.jpg")), cv2.COLOR_BGR2RGB)
        return img_cache[f]

    report = []
    synth_count = Counter()
    fallback_count = Counter()
    for s, cid in zip(focus, focus_ids):
        pool = usable[s]
        if not pool or not targets[s]:
            print(f"  {s}: no usable donors or acceptor frames -- no synthetic images")
            continue
        order = []
        while len(order) < args.synth_per_species:
            batch = pool[:]
            rng.shuffle(batch)
            order.extend(batch)
        order = order[:args.synth_per_species]  # balanced donor reuse
        for k, d in enumerate(order):
            src = load_rgb(d["frame"])
            x0, y0, bw, bh = (int(round(v)) for v in poly_bbox_px(d["poly"], W_IMG, H_IMG))
            x1, y1 = min(x0 + bw + 1, W_IMG), min(y0 + bh + 1, H_IMG)
            full_mask = np.zeros((H_IMG, W_IMG), np.uint8)
            cv2.fillPoly(full_mask, [np.round(d["poly"] * [W_IMG, H_IMG]).astype(np.int32)], 1)
            patch, mask = src[y0:y1, x0:x1], full_mask[y0:y1, x0:x1]
            placed = False
            for _ in range(20):
                tf = rng.choice(targets[s])
                timg = load_rgb(tf)
                p2, m2, angle, flipped = random_transform(patch, mask, rng, args.max_rotation, args.hflip_prob)
                boxes = [poly_bbox_px(poly, W_IMG, H_IMG) for _, poly in labels.get(tf, [])]
                spot = find_free_spot(rng, timg.shape, m2.shape, boxes)
                if spot is None:
                    continue
                x, y = spot
                if args.blend == "poisson_edge":
                    out, fb = paste_poisson_edge(timg, p2, m2, x, y)
                elif args.blend == "poisson":
                    out, fb = paste_poisson(timg, p2, m2, x, y)
                else:
                    out, fb = paste_matched_feathered(timg, p2, m2, x, y), False
                new_poly = mask_to_polygon_norm(m2, x, y, W_IMG, H_IMG)
                if new_poly is None:
                    continue
                stem = f"synth_{s.replace(' ', '_')}_{k:04d}_t{tf}_d{d['frame']}"
                img_path = args.out / "synth_images" / f"{stem}.jpg"
                cv2.imwrite(str(img_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                for kind in ("det", "seg"):  # synthetic images go ONLY into the copy-paste roots' train
                    link_or_copy(img_path, args.out / f"{kind}_copypaste" / "images" / "train" / f"{stem}.jpg",
                                 args.copy_images)
                real_rows = labels.get(tf, [])
                for kind, fmt in (("det", det_line), ("seg", seg_line)):
                    lines = [fmt(c, p) for c, p in real_rows] + [fmt(cid, new_poly)]
                    (args.out / f"{kind}_copypaste" / "labels" / "train" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                report.append({"synth": stem, "species": s, "donor_frame": d["frame"], "donor_source": d["source"],
                               "target_frame": tf, "angle": round(angle, 1), "hflip": int(flipped),
                               "x": x, "y": y, "w": m2.shape[1], "h": m2.shape[0], "poisson_fallback": int(fb)})
                synth_count[s] += 1
                fallback_count[s] += int(fb)
                placed = True
                break
            if not placed:
                print(f"  could not place donor f{d['frame']} after 20 targets -- skipped")
        print(f"  {s}: wrote {synth_count[s]} synthetic images ({fallback_count[s]} used the matched-blend fallback)")

    with open(args.out / "synth_report.csv", "w", newline="") as fh:
        if report:
            wr = csv.DictWriter(fh, fieldnames=list(report[0].keys()))
            wr.writeheader()
            wr.writerows(report)

    # ---- yamls
    names_block = "names:\n" + "".join(f"  {k}: {names[k]}\n" for k in sorted(names))
    for kind, arm in ROOTS:
        root = (args.out / f"{kind}_{arm}").resolve()
        (args.out / f"{kind}_{arm}.yaml").write_text(
            f"# {kind} / {arm} -- built by prepare_experiment.py"
            + (" (train = real frames + synthetic copy-paste images)" if arm == "copypaste" else " (train = real frames only)")
            + f"\npath: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n" + names_block)

    # ---- summary
    summary = {
        "frame_range": [args.min_frame, args.max_frame], "block_size": args.block_size,
        "guard_frames": args.guard_frames, "split_seed": args.split_seed, "synth_seed": args.seed,
        "blend": args.blend, "acceptor_same_species": args.acceptor_same_species, "max_rotation": args.max_rotation, "hflip_prob": args.hflip_prob,
        "frames_per_split": dict(n_split), "focus_species": focus,
        "focus": {s: {"class_id": cid,
                      "real_instances": {sp: inst[sp][cid] for sp in ("train", "val", "test")},
                      "donors_usable": len(usable[s]),
                      "donor_frames": sorted({d["frame"] for d in usable[s]}),
                      "acceptor_frames": len(targets[s]),
                      "synthetic_images": synth_count[s],
                      "poisson_fallbacks": fallback_count[s]}
                  for s, cid in zip(focus, focus_ids)},
    }
    (args.out / "prepare_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- QA sheet: draw the pasted fish's polygon FROM THE WRITTEN LABEL FILE
    if report and args.qa_samples > 0:
        tiles = []
        for r in random.Random(1).sample(report, min(args.qa_samples, len(report))):
            img = cv2.imread(str(args.out / "synth_images" / f"{r['synth']}.jpg"))
            last = (args.out / "seg_copypaste" / "labels" / "train" / f"{r['synth']}.txt").read_text().strip().splitlines()[-1]
            poly = (np.array(last.split()[1:], float).reshape(-1, 2) * [W_IMG, H_IMG]).astype(np.int32)
            cv2.polylines(img, [poly], True, (255, 0, 255), 1)
            x0, y0 = poly.min(0)
            x1, y1 = poly.max(0)
            pad = max(40, int(0.8 * max(x1 - x0, y1 - y0)))
            crop = img[max(0, y0 - pad):min(H_IMG, y1 + pad), max(0, x0 - pad):min(W_IMG, x1 + pad)]
            crop = cv2.resize(crop, (300, 300))
            cv2.rectangle(crop, (0, 0), (300, 22), (0, 0, 0), -1)
            cv2.putText(crop, f"{r['species'][:10]} d{r['donor_frame']} t{r['target_frame']}", (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(crop)
        while len(tiles) % 6:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)])
        cv2.imwrite(str(args.out / "qa_synthetic_samples.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    print(f"\ndone -> {args.out}")
    for kind in ("det", "seg"):
        for arm in ("baseline", "copypaste"):
            print(f"  {args.out / f'{kind}_{arm}.yaml'}")


if __name__ == "__main__":
    main()
