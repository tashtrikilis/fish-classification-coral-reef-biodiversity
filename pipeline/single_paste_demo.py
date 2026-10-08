#!/usr/bin/env python3
"""
Deterministic single-instance copy-paste demo, for visually showing exactly
what the augmentation pipeline does to one chosen fish in one chosen frame,
at a location you pick yourself -- rather than the randomized batch process
copy_paste_augment.py normally runs.

Reuses the EXACT SAME blending/eligibility/scale-jitter functions from
copy_paste_augment.py (imported, not reimplemented) so this demo is a true
representation of what the real pipeline does, not an approximation.

Run this from the same aqua_yolo/ folder as copy_paste_augment.py (it must
be importable).

Usage
-----
    # 1. Find a donor frame/instance to use -- e.g. pick a Yellow Tang
    #    instance from instance_meta_v3.csv (donor_eligibility="1" there
    #    should match what this script reports for consistency).
    #
    # 2. Paste it into a target frame at an exact pixel location you choose:
    python3 single_paste_demo.py \\
        --seg-src dataset_seg_top10_v3 \\
        --donor-frame frame_001234 --donor-instance-index 0 \\
        --target-frame frame_005678 \\
        --x 400 --y 200 \\
        --blend matched \\
        --out-dir paste_demo_out

Outputs (in --out-dir):
    donor_crop.jpg     -- the cropped source fish (post scale/rotate jitter)
    target_before.jpg  -- the clean target frame, untouched
    target_after.jpg   -- the target frame with the fish pasted in at (x, y)
    demo_report.txt    -- eligibility check + boundary-color-mismatch score

--donor-instance-index is the 0-based line number of the instance in that
frame's label file (labels/train/<donor-frame>.txt), same indexing used by
donor_eligibility/train/<donor-frame>.txt -- print a frame's label file
alongside its donor_eligibility file if you're not sure which index is which
fish.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from copy_paste_augment import (
    load_yolo_seg_labels, polygon_to_mask, crop_instance, jitter_crop,
    paste_matched, paste_poisson, boundary_color_mismatch,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seg-src", required=True, type=Path,
                   help="Dataset root containing images/train, labels/train, donor_eligibility/train "
                        "(e.g. dataset_seg_top10_v3).")
    p.add_argument("--donor-frame", required=True, type=str, help="Stem of the donor frame, e.g. frame_001234")
    p.add_argument("--donor-instance-index", required=True, type=int,
                   help="0-based index of the instance within the donor frame's label file.")
    p.add_argument("--target-frame", required=True, type=str, help="Stem of the target frame to paste into.")
    p.add_argument("--x", type=int, required=True, help="Top-left x pixel coordinate to paste at.")
    p.add_argument("--y", type=int, required=True, help="Top-left y pixel coordinate to paste at.")
    p.add_argument("--scale", type=float, default=1.0, help="Scale jitter to apply (1.0 = none, for a clean demo).")
    p.add_argument("--rotate-deg", type=float, default=0.0, help="Rotation jitter in degrees (0 = none).")
    p.add_argument("--blend", choices=["matched", "poisson"], default="matched")
    p.add_argument("--out-dir", type=Path, default=Path("paste_demo_out"))
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    img_dir = args.seg_src / "images" / "train"
    lbl_dir = args.seg_src / "labels" / "train"
    donor_dir = args.seg_src / "donor_eligibility" / "train"

    donor_img_path = img_dir / f"{args.donor_frame}.jpg"
    donor_lbl_path = lbl_dir / f"{args.donor_frame}.txt"
    if not donor_img_path.exists():
        raise SystemExit(f"donor image not found: {donor_img_path}")
    donor_img = cv2.imread(str(donor_img_path))
    dh, dw = donor_img.shape[:2]

    instances = load_yolo_seg_labels(donor_lbl_path, dw, dh)
    if args.donor_instance_index >= len(instances):
        raise SystemExit(f"--donor-instance-index {args.donor_instance_index} out of range "
                          f"({len(instances)} instances in {donor_lbl_path})")
    cls, poly = instances[args.donor_instance_index]

    # ---- eligibility check, exactly as copy_paste_augment.py reads it ----
    donor_flags_path = donor_dir / f"{args.donor_frame}.txt"
    donor_flags = donor_flags_path.read_text().splitlines() if donor_flags_path.exists() else []
    eligible = (args.donor_instance_index < len(donor_flags)
                and donor_flags[args.donor_instance_index].strip() == "1")

    donor_mask = polygon_to_mask(poly, (dh, dw))
    crop, crop_mask = crop_instance(donor_img, donor_mask, poly)
    jcrop, jmask = jitter_crop(crop, crop_mask, args.scale, args.rotate_deg)
    if jcrop is None:
        raise SystemExit("jitter_crop degenerated to an empty mask -- try scale closer to 1.0")

    target_img_path = img_dir / f"{args.target_frame}.jpg"
    if not target_img_path.exists():
        raise SystemExit(f"target image not found: {target_img_path}")
    target_img = cv2.imread(str(target_img_path))
    th, tw = target_img.shape[:2]
    ch, cw = jmask.shape
    if args.y + ch > th or args.x + cw > tw:
        raise SystemExit(f"paste location ({args.x},{args.y}) + crop size ({cw}x{ch}) "
                          f"exceeds target frame bounds ({tw}x{th}) -- pick a smaller x/y")

    top_left = (args.x, args.y)
    if args.blend == "poisson":
        pasted = paste_poisson(target_img, jcrop, jmask, top_left)
    else:
        pasted = paste_matched(target_img, jcrop, jmask, top_left)
    mismatch = boundary_color_mismatch(pasted, jmask, top_left)

    cv2.imwrite(str(args.out_dir / "donor_crop.jpg"), jcrop)
    cv2.imwrite(str(args.out_dir / "target_before.jpg"), target_img)
    cv2.imwrite(str(args.out_dir / "target_after.jpg"), pasted)

    report = (
        f"donor: {args.donor_frame} instance #{args.donor_instance_index} (class {cls})\n"
        f"donor eligibility flag: {'ELIGIBLE (not occluded, not edge-cut)' if eligible else 'NOT ELIGIBLE'}\n"
        f"  -- copy_paste_augment.py would {'have used' if eligible else 'have SKIPPED'} this instance as a donor\n"
        f"target: {args.target_frame}\n"
        f"paste location (top-left px): ({args.x}, {args.y})\n"
        f"scale jitter: {args.scale}, rotate jitter: {args.rotate_deg} deg\n"
        f"blend mode: {args.blend}\n"
        f"boundary color mismatch score: {mismatch:.1f} "
        f"(pipeline's --reject-threshold default is 80.0 -- above that, this placement "
        f"would have been rejected and retried elsewhere)\n"
    )
    (args.out_dir / "demo_report.txt").write_text(report)
    print(report)
    print(f"wrote donor_crop.jpg, target_before.jpg, target_after.jpg, demo_report.txt to {args.out_dir}/")


if __name__ == "__main__":
    main()
