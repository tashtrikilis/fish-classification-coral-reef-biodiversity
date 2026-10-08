#!/usr/bin/env python3
"""
Generate copy-paste synthetic training images, targeting minority species,
from a v2/v3-style dataset pair (dataset_seg_* for masks/donors, paired
dataset_detect_* for the matching bbox labels). TRAIN SPLIT ONLY -- val is
never touched, read from, or written to, so evaluation stays on real data.

Donor eligibility comes from dataset_seg_*/donor_eligibility/train (written
by cvat_masks_to_yolo.py, carried through class-subset filtering by
filter_topk_classes.py's --donor-src/--donor-dst): an instance is only used
as a copy-paste SOURCE if it is not occluded and not cut off by the frame
edge, so nothing pasted is half a fish. This never touches the real label
files -- every real instance (occluded or not) stays in labels/train exactly
as it was; eligibility only gates what gets COPIED, not what counts as
ground truth.

Blending: two paste strategies, chosen by --blend:
  - "matched" (default): per-channel LAB histogram-match the donor crop to a
    local patch of its new background, then feather the edge with a
    Gaussian-blurred alpha mask. Validated in copy_paste_experiment.ipynb --
    corrects the color cast through the whole pasted instance, not just a
    thin ring at the boundary the way feathering alone does.
  - "poisson": OpenCV seamlessClone (Perez et al. 2003 Poisson blending) --
    solves for a smooth gradient across the whole boundary. Can occasionally
    produce its own artifacts on high-contrast/textured backgrounds (coral
    rubble), so it's offered as an alternative to compare, not a strict
    upgrade.
Both are followed by the same boundary-color-mismatch accept/reject gate
used in the exploratory notebook (tight-ring L2 color distance at the seam),
so a placement that still looks stuck on gets retried against a different
random target/location rather than kept.

Diversity: each donor instance gets randomized scale + rotation jitter
before pasting (see jitter_crop) -- plain "Copy-Paste" augmentation
(Ghiasi et al. 2021) relies on this kind of jittering for real diversity,
which matters more here since a handful of donor instances get reused many
times for the rarest classes.

Safety: a synthetic paste is rejected if it would overlap (in bbox) any
REAL instance already labelled in the target frame -- pasting on top of a
real fish would silently corrupt that instance's ground truth. --max-donor-
reuse caps how many times any single donor instance can be reused, so
"equalizing" a near-zero-instance class doesn't quietly become 200 copies
of the one donor that happens to exist.

Output: new images + labels go into images/train_synth and labels/train_synth
under BOTH --seg-src and --detect-src (same synthetic image bytes in both,
seg gets a polygon line, detect gets the matching bbox line, both also carry
over that target frame's existing REAL train instances unchanged) -- real
train images/labels are never modified. Pass --seg-yaml/--detect-yaml to
also rewrite `train:` in those data.yaml files into a 2-item list
([images/train, images/train_synth]) so Ultralytics picks up both.

Two quota modes (--mode):
  - equalize: every class's synthetic quota tops it up toward
    --target-count (default: the current largest class's real train count).
  - minority-cap: only classes with fewer than --minority-threshold real
    train instances get topped up, by a flat +--minority-cap synthetic
    instances each (not equalized to any target -- a fixed number added).
Either way, a class's quota is capped at --max-donor-reuse times its
eligible-donor count, and the run report says which classes couldn't reach
quota because they simply don't have enough eligible donors.

Usage
-----
    python3 copy_paste_augment.py \\
        --seg-src dataset_seg_top10_v3 --detect-src dataset_detect_top10_v3 \\
        --seg-yaml data_seg_top10_v3.yaml --detect-yaml data_detect_top10_v3.yaml \\
        --mode equalize \\
        --blend matched \\
        --report-csv copy_paste_report_top10_equalize.csv \\
        --seed 42
"""
import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg-src", required=True, type=Path)
    p.add_argument("--detect-src", required=True, type=Path)
    p.add_argument("--seg-yaml", type=Path, default=None)
    p.add_argument("--detect-yaml", type=Path, default=None)
    p.add_argument("--mode", choices=["equalize", "minority-cap"], required=True)
    p.add_argument("--target-count", type=int, default=None,
                   help="equalize mode only. Default: current largest class's real train instance count.")
    p.add_argument("--minority-threshold", type=int, default=50,
                   help="minority-cap mode only: classes with fewer real train instances than this get topped up.")
    p.add_argument("--minority-cap", type=int, default=30,
                   help="minority-cap mode only: flat number of synthetic instances added per qualifying class.")
    p.add_argument("--max-donor-reuse", type=int, default=15,
                   help="Max times any single donor instance can be reused across all synthetic pastes.")
    p.add_argument("--blend", choices=["matched", "poisson"], default="matched")
    p.add_argument("--scale-jitter", type=str, default="0.85,1.15", help="min,max scale multiplier.")
    p.add_argument("--rotate-jitter-deg", type=float, default=12.0, help="max abs rotation, degrees.")
    p.add_argument("--reject-ring", type=int, default=2)
    p.add_argument("--reject-threshold", type=float, default=80.0)
    p.add_argument("--max-target-attempts", type=int, default=10)
    p.add_argument("--report-csv", type=Path, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--min-instance-area", type=int, default=100)
    p.add_argument("--synth-suffix", type=str, default=None,
                   help="Output folders become images/train_synth_<suffix> and labels/train_synth_<suffix> "
                        "instead of plain train_synth -- required if you're running more than one "
                        "experiment (e.g. equalize AND minority-cap) against the same base dataset, so "
                        "the second run doesn't overwrite the first's synthetic images. Default: --mode "
                        "(e.g. 'equalize' or 'minority-cap').")
    return p.parse_args()


# ---------------------------------------------------------------- label I/O

def load_yolo_seg_labels(label_path, img_w, img_h):
    instances = []
    if not label_path.exists():
        return instances
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        coords = np.array(parts[1:], dtype=np.float64).reshape(-1, 2)
        coords[:, 0] *= img_w
        coords[:, 1] *= img_h
        instances.append((cls, coords))
    return instances


def load_yolo_det_labels(label_path):
    """Returns list of (cls, xc, yc, w, h) in normalized coords, plus the raw lines."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        out.append((int(parts[0]), *[float(x) for x in parts[1:5]]))
    return out


def polygon_to_mask(polygon, img_shape):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


# ---------------------------------------------------------------- crop / jitter / paste

def crop_instance(image, mask, polygon, pad=4):
    x, y, w, h = cv2.boundingRect(polygon.astype(np.int32))
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, image.shape[1]), min(y + h + pad, image.shape[0])
    return image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def jitter_crop(crop, crop_mask, scale, angle_deg):
    """Randomized rotation + scale of a crop+mask pair, re-cropped tight to the
    transformed mask's new bounding box. Returns (None, None) if the transform
    degenerates to nothing (shouldn't happen in practice for real fish crops)."""
    h, w = crop_mask.shape
    diag = int(np.ceil((h ** 2 + w ** 2) ** 0.5)) + 4
    pad_img = np.zeros((diag, diag, 3), dtype=crop.dtype)
    pad_mask = np.zeros((diag, diag), dtype=crop_mask.dtype)
    oy, ox = (diag - h) // 2, (diag - w) // 2
    pad_img[oy:oy + h, ox:ox + w] = crop
    pad_mask[oy:oy + h, ox:ox + w] = crop_mask
    center = (diag / 2, diag / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    rot_img = cv2.warpAffine(pad_img, M, (diag, diag), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rot_mask = cv2.warpAffine(pad_mask, M, (diag, diag), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    ys, xs = np.where(rot_mask > 0)
    if len(ys) == 0:
        return None, None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return rot_img[y0:y1, x0:x1], rot_mask[y0:y1, x0:x1]


def local_target_patch(target, crop_shape, top_left, margin=10):
    tx, ty = top_left
    ch, cw = crop_shape[:2]
    x0, y0 = max(tx - margin, 0), max(ty - margin, 0)
    x1, y1 = min(tx + cw + margin, target.shape[1]), min(ty + ch + margin, target.shape[0])
    return target[y0:y1, x0:x1]


def match_channel_histogram(source_values, reference_values):
    src_uniques, src_inverse, src_counts = np.unique(source_values, return_inverse=True, return_counts=True)
    ref_uniques, ref_counts = np.unique(reference_values, return_counts=True)
    src_cdf = np.cumsum(src_counts).astype(np.float64) / source_values.size
    ref_cdf = np.cumsum(ref_counts).astype(np.float64) / reference_values.size
    matched_uniques = np.interp(src_cdf, ref_cdf, ref_uniques)
    return matched_uniques[src_inverse]


def match_color_histogram(crop, crop_mask, target_patch):
    crop_lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    patch_lab = cv2.cvtColor(target_patch, cv2.COLOR_BGR2LAB)
    m = crop_mask > 0
    if m.sum() == 0 or target_patch.size == 0:
        return crop.copy()
    matched_lab = crop_lab.copy()
    for ch in range(3):
        matched_lab[..., ch][m] = np.clip(
            match_channel_histogram(crop_lab[..., ch][m], patch_lab[..., ch].ravel()), 0, 255
        ).astype(np.uint8)
    return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2BGR)


def paste_feathered(target, crop, crop_mask, top_left, blur_ksize=15):
    tx, ty = top_left
    ch, cw = crop.shape[:2]
    ty1, tx1 = min(ty + ch, target.shape[0]), min(tx + cw, target.shape[1])
    ch2, cw2 = ty1 - ty, tx1 - tx
    soft_mask = cv2.GaussianBlur(crop_mask[:ch2, :cw2].astype(np.float32), (blur_ksize, blur_ksize), 0) / 255.0
    soft_mask = soft_mask[..., None]
    out = target.copy()
    roi = out[ty:ty1, tx:tx1].astype(np.float32)
    blended = roi * (1 - soft_mask) + crop[:ch2, :cw2].astype(np.float32) * soft_mask
    out[ty:ty1, tx:tx1] = blended.astype(np.uint8)
    return out


def paste_matched(target, crop, crop_mask, top_left, blur_ksize=15, margin=10):
    bg_patch = local_target_patch(target, crop.shape, top_left, margin=margin)
    matched_crop = match_color_histogram(crop, crop_mask, bg_patch)
    return paste_feathered(target, matched_crop, crop_mask, top_left, blur_ksize=blur_ksize)


def paste_poisson(target, crop, crop_mask, top_left):
    tx, ty = top_left
    ch, cw = crop.shape[:2]
    center = (tx + cw // 2, ty + ch // 2)
    mask255 = (crop_mask > 0).astype(np.uint8) * 255
    try:
        return cv2.seamlessClone(crop, target, mask255, center, cv2.NORMAL_CLONE)
    except cv2.error:
        return paste_feathered(target, crop, crop_mask, top_left)  # fall back if seamlessClone rejects geometry


def boundary_color_mismatch(image, mask, top_left, ring_width=6):
    tx, ty = top_left
    ch, cw = mask.shape[:2]
    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    ty1, tx1 = min(ty + ch, image.shape[0]), min(tx + cw, image.shape[1])
    ch2, cw2 = ty1 - ty, tx1 - tx
    full_mask[ty:ty1, tx:tx1] = mask[:ch2, :cw2]
    kernel = np.ones((ring_width, ring_width), np.uint8)
    inside_ring = (full_mask > 0) & (cv2.erode(full_mask, kernel) == 0)
    outside_ring = (cv2.dilate(full_mask, kernel) > 0) & (full_mask == 0)
    if inside_ring.sum() == 0 or outside_ring.sum() == 0:
        return 0.0
    inside_mean = image[inside_ring].mean(axis=0)
    outside_mean = image[outside_ring].mean(axis=0)
    return float(np.linalg.norm(inside_mean.astype(np.float32) - outside_mean.astype(np.float32)))


def mask_to_polygon(local_mask, epsilon_frac=0.003):
    contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon_frac * peri, True)
    poly = approx.reshape(-1, 2).astype(np.float64)
    if len(poly) < 3:
        poly = largest.reshape(-1, 2).astype(np.float64)
    return poly if len(poly) >= 3 else None


def bbox_overlaps_any(cand_box, existing_boxes):
    cx0, cy0, cx1, cy1 = cand_box
    for (ex0, ey0, ex1, ey1) in existing_boxes:
        if cx0 < ex1 and cx1 > ex0 and cy0 < ey1 and cy1 > ey0:
            return True
    return False


def random_paste_location(rng, crop_shape, target_shape):
    ch, cw = crop_shape[:2]
    th, tw = target_shape[:2]
    if ch >= th or cw >= tw:
        return None
    return (rng.randint(0, tw - cw), rng.randint(0, th - ch))


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    scale_lo, scale_hi = (float(x) for x in args.scale_jitter.split(","))

    seg_img_dir = args.seg_src / "images" / "train"
    seg_lbl_dir = args.seg_src / "labels" / "train"
    donor_dir = args.seg_src / "donor_eligibility" / "train"
    det_img_dir = args.detect_src / "images" / "train"
    det_lbl_dir = args.detect_src / "labels" / "train"

    for d in (seg_img_dir, seg_lbl_dir, donor_dir, det_img_dir, det_lbl_dir):
        if not d.exists():
            raise SystemExit(f"expected dir does not exist: {d}")

    # ---- index all train frames: real instances (seg + detect) + donor eligibility ----
    frame_stems = sorted(p.stem for p in seg_img_dir.glob("frame_*.jpg"))
    class_real_counts = Counter()
    donors_by_class = defaultdict(list)  # class_id -> list of (stem, poly_pixel_coords)
    frame_wh = {}

    for stem in frame_stems:
        img_path = seg_img_dir / f"{stem}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARNING: unreadable image, skipping: {img_path}")
            continue
        h, w = img.shape[:2]
        frame_wh[stem] = (w, h)
        seg_instances = load_yolo_seg_labels(seg_lbl_dir / f"{stem}.txt", w, h)
        donor_flags = (donor_dir / f"{stem}.txt").read_text().splitlines() if (donor_dir / f"{stem}.txt").exists() else []
        for i, (cls, poly) in enumerate(seg_instances):
            class_real_counts[cls] += 1
            eligible = i < len(donor_flags) and donor_flags[i].strip() == "1"
            if eligible and cv2.contourArea(poly.astype(np.int32)) >= args.min_instance_area:
                donors_by_class[cls].append((stem, poly))

    print(f"{len(frame_stems)} train frames indexed, {sum(class_real_counts.values())} real instances, "
          f"{sum(len(v) for v in donors_by_class.values())} eligible donor instances "
          f"across {len(donors_by_class)} classes")

    # ---- compute per-class quota ----
    quota = {}
    if args.mode == "equalize":
        target = args.target_count or max(class_real_counts.values())
        for cls, n in class_real_counts.items():
            quota[cls] = max(0, target - n)
        print(f"equalize mode: target={target} instances per class")
    else:
        for cls, n in class_real_counts.items():
            quota[cls] = args.minority_cap if n < args.minority_threshold else 0
        print(f"minority-cap mode: threshold={args.minority_threshold}, cap=+{args.minority_cap}")

    capped_quota = {}
    for cls, q in quota.items():
        if q <= 0:
            continue
        n_donors = len(donors_by_class.get(cls, []))
        cap = args.max_donor_reuse * n_donors
        capped_quota[cls] = min(q, cap)
        if n_donors == 0 and q > 0:
            print(f"  class {cls}: wants +{q} but has ZERO eligible donors -- skipped entirely")
        elif capped_quota[cls] < q:
            print(f"  class {cls}: wants +{q}, capped to +{capped_quota[cls]} "
                  f"({n_donors} donors x max_donor_reuse={args.max_donor_reuse})")

    # ---- generate ----
    suffix = args.synth_suffix or args.mode.replace("-", "")
    synth_dirname = f"train_synth_{suffix}"
    synth_seg_img = args.seg_src / "images" / synth_dirname
    synth_seg_lbl = args.seg_src / "labels" / synth_dirname
    synth_det_img = args.detect_src / "images" / synth_dirname
    synth_det_lbl = args.detect_src / "labels" / synth_dirname
    for d in (synth_seg_img, synth_seg_lbl, synth_det_img, synth_det_lbl):
        d.mkdir(parents=True, exist_ok=True)

    donor_use_count = Counter()
    added_count = Counter()
    report_rows = []
    synth_idx = 0

    # cache decoded target images + their real label lines (re-read cheaply; train sets here are a few hundred-thousand px images, fine to re-read per paste)
    for cls in sorted(capped_quota, key=lambda c: capped_quota[c], reverse=True):
        need = capped_quota[cls]
        pool = donors_by_class.get(cls, [])
        if not pool or need <= 0:
            continue
        made = 0
        stall = 0
        while made < need and stall < need * 4 + 20:
            stall += 1
            donor_stem, donor_poly = rng.choice(pool)
            if donor_use_count[(donor_stem, tuple(map(tuple, donor_poly.round(1))))] >= args.max_donor_reuse:
                continue
            donor_img = cv2.imread(str(seg_img_dir / f"{donor_stem}.jpg"))
            dw, dh = frame_wh[donor_stem]
            donor_mask = polygon_to_mask(donor_poly, (dh, dw))
            crop, crop_mask = crop_instance(donor_img, donor_mask, donor_poly)

            scale = rng.uniform(scale_lo, scale_hi)
            angle = rng.uniform(-args.rotate_jitter_deg, args.rotate_jitter_deg)
            jcrop, jmask = jitter_crop(crop, crop_mask, scale, angle)
            if jcrop is None:
                continue

            tgt_stem = rng.choice(frame_stems)
            tgt_img = cv2.imread(str(seg_img_dir / f"{tgt_stem}.jpg"))
            tw, th = frame_wh[tgt_stem]
            existing_seg = load_yolo_seg_labels(seg_lbl_dir / f"{tgt_stem}.txt", tw, th)
            existing_boxes = [cv2.boundingRect(p.astype(np.int32)) for _, p in existing_seg]
            existing_boxes = [(x, y, x + w, y + h) for (x, y, w, h) in existing_boxes]

            accepted = None
            for _ in range(args.max_target_attempts):
                top_left = random_paste_location(rng, jcrop.shape, tgt_img.shape)
                if top_left is None:
                    continue
                cx0, cy0 = top_left
                ch, cw = jmask.shape
                if bbox_overlaps_any((cx0, cy0, cx0 + cw, cy0 + ch), existing_boxes):
                    continue
                if args.blend == "poisson":
                    pasted = paste_poisson(tgt_img, jcrop, jmask, top_left)
                else:
                    pasted = paste_matched(tgt_img, jcrop, jmask, top_left)
                mismatch = boundary_color_mismatch(pasted, jmask, top_left, ring_width=args.reject_ring)
                if mismatch <= args.reject_threshold:
                    accepted = (pasted, top_left, mismatch)
                    break
            if accepted is None:
                continue  # couldn't find a clean placement this round -- try a different donor/target next loop

            pasted, top_left, mismatch = accepted
            full_mask = np.zeros((th, tw), dtype=np.uint8)
            tx, ty = top_left
            ch, cw = jmask.shape
            full_mask[ty:ty + ch, tx:tx + cw] = jmask
            new_poly_local = mask_to_polygon(full_mask[ty:ty + ch, tx:tx + cw])
            if new_poly_local is None:
                continue
            new_poly_full = new_poly_local + np.array([tx, ty])
            new_poly_norm = np.clip(new_poly_full / np.array([tw, th]), 0.0, 1.0)

            x, y, w, h = cv2.boundingRect(full_mask)
            xc, yc, wn, hn = (x + w / 2) / tw, (y + h / 2) / th, w / tw, h / th

            stem = f"synth_{cls}_{synth_idx:06d}"
            synth_idx += 1
            cv2.imwrite(str(synth_seg_img / f"{stem}.jpg"), pasted)
            cv2.imwrite(str(synth_det_img / f"{stem}.jpg"), pasted)

            real_seg_lines = (seg_lbl_dir / f"{tgt_stem}.txt").read_text().splitlines()
            real_det_lines = (det_lbl_dir / f"{tgt_stem}.txt").read_text().splitlines()
            new_seg_line = f"{cls} " + " ".join(f"{px:.6f} {py:.6f}" for px, py in new_poly_norm)
            new_det_line = f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"
            (synth_seg_lbl / f"{stem}.txt").write_text("\n".join(real_seg_lines + [new_seg_line]) + "\n")
            (synth_det_lbl / f"{stem}.txt").write_text("\n".join(real_det_lines + [new_det_line]) + "\n")

            donor_use_count[(donor_stem, tuple(map(tuple, donor_poly.round(1))))] += 1
            made += 1
            added_count[cls] += 1
            report_rows.append({
                "synth_stem": stem, "class_id": cls, "donor_frame": donor_stem, "target_frame": tgt_stem,
                "blend": args.blend, "scale": round(scale, 3), "rotate_deg": round(angle, 2),
                "boundary_mismatch": round(mismatch, 1),
            })

        if made < need:
            print(f"  class {cls}: only generated {made}/{need} -- ran out of clean placements "
                  f"(rejection filter + overlap-avoidance too strict for available donors/targets)")

    print(f"\ngenerated {synth_idx} synthetic images "
          f"({', '.join(f'class {c}: +{n}' for c, n in sorted(added_count.items()))})")
    print(f"wrote synthetic seg data to {synth_seg_img} / {synth_seg_lbl}")
    print(f"wrote synthetic detect data to {synth_det_img} / {synth_det_lbl}")

    if args.report_csv and report_rows:
        with open(args.report_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"wrote per-instance report to {args.report_csv}")

    synth_path_line = f"  - images/{synth_dirname}\n"
    for yaml_path in (args.seg_yaml, args.detect_yaml):
        if yaml_path is None or not yaml_path.exists():
            continue
        text = yaml_path.read_text()
        if synth_path_line.strip() in text:
            continue  # this exact synth folder already added, don't duplicate
        if "train:\n  - " in text:
            # already a list (from a previous, differently-suffixed augment run) -- append a line
            text = text.replace(
                "train:\n  - images/train\n",
                "train:\n  - images/train\n" + synth_path_line,
                1,
            )
        else:
            text = text.replace(
                "train: images/train\n",
                "train:\n  - images/train\n" + synth_path_line,
                1,
            )
        yaml_path.write_text(text)
        print(f"patched {yaml_path} to also train on images/{synth_dirname}")


if __name__ == "__main__":
    main()
