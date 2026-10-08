#!/usr/bin/env python3
"""
Convert a CVAT-for-images 1.1 XML export that uses the mask/brush annotation
tool (RLE-encoded <mask> elements, with a per-instance `occluded` attribute)
into flat YOLO label directories -- one .txt per frame, `frame_<N>.txt` --
for both instance segmentation (polygon) and bounding-box detection, plus a
CSV of per-instance metadata (occluded flag, "cut off by frame edge" flag,
bbox) that a copy-paste augmentation step can use to decide which instances
are safe to use as paste donors.

Important: this writes the COMPLETE, unfiltered label set -- every real
instance, including occluded ones and ones cut off by the frame edge, stays
in frame_<N>.txt exactly as CVAT recorded it. Nothing is dropped from the
ground-truth labels. occluded/edge-cut only matter for a *separate*
donor-eligibility decision (used later by the copy-paste script), not for
what counts as ground truth here.

CVAT mask RLE format: `rle="c0, c1, c2, ..."` is a list of run-lengths over
the mask's bounding box (given by left/top/width/height on the same
element), in row-major order, alternating background/foreground starting
with background (run c0 = background pixels, c1 = foreground, c2 =
background, ...). Decoding is representation-agnostic to whether the object
comes out as one blob or several disconnected pieces (can happen with noisy
manual brushing) -- see --report for how many instances that affected.

Frame numbering: CVAT image names look like
`nelha-trimmed_mp4-<N>_jpg.rf.<hash>.jpg` -- `<N>` is extracted and used as
the `frame_<N>` stem, matching the naming already used by dataset_seg /
dataset_detect / dataset_seg_v2 / dataset_detect_v2 on Aqua.

Usage
-----
    python3 cvat_masks_to_yolo.py \\
        --xml annotations.xml \\
        --seg-out dataset_seg_labels_v3 \\
        --detect-out dataset_detect_labels_v3 \\
        --meta-csv instance_meta_v3.csv \\
        --edge-margin 2
"""
import argparse
import csv
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Same 58-species taxonomy / id order as data_seg_v2.yaml, data_detect_v2.yaml,
# and restratify_split.py's NAMES dict -- kept identical on purpose so class ids
# stay stable across the old and new label sets.
NAMES = [
    "Acanthuridae", "Black Triggerfish", "Blackfin Chromis", "Blackspot Sergeant", "Blue Goatfish",
    "Bluefin Trevally", "Bluelined Surgeonfish", "Bluespine Unicornfish", "Bluestripe Snapper", "Bridled Triggerfish",
    "Brown Surgeonfish", "Bullethead Unicornfish", "Bullethead Parrotfish - Female", "Chevron Tang", "Chinese Trumpetfish",
    "Convict Tang", "Doublebar Goatfish", "Forcepsfish", "Fourspot Butterflyfish", "Goldring Surgeonfish",
    "Hawaiian Cleaner Wrasse", "Hawaiian Gregory", "Hawaiian Sergeant", "Hawaiian Whitespotted Toby", "Highfin Chub",
    "Indo-Pacific Sergeant", "Lei Triggerfish", "Lined Butterflyfish", "Lognnose Butterflyfish", "Lowfin Chub",
    "Manybar Goatfish", "Moorish Idol", "Multiband Butterflyfish", "Orangeband Surgeonfish", "Orangespine Unicornfish",
    "Ornate Butterfly Fish", "Oval Chromis", "Palenose Parrotfish", "Palenose Parrotfish - Male", "Peacock Grouper",
    "Pinktail Triggerfish", "Pyramid Butterflyfish", "Raccoon Butterflyfish", "Redlip Parrotfish - Male", "Redlip Parrotfish - Female",
    "Speckled Parrotfish - Male", "Spotted Boxfish", "Stareye Parrotfish - Male", "Stripebelly Puffer", "Threadfin Butterflyfish",
    "Whitebar Surgeonfish", "Whitecheek Surgeonfish", "Whitespotted Surgeonfish", "Yellow Tang", "Yellowfin Goatfish",
    "Yellowtail Coris", "Ringtail Surgeonfish", "Saddle Wrasse - Male",
]
NAME_TO_ID = {n: i for i, n in enumerate(NAMES)}

FRAME_RE = re.compile(r"nelha-trimmed_mp4-(\d+)_jpg")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", required=True, type=Path, help="CVAT annotations.xml (mask export).")
    p.add_argument("--seg-out", required=True, type=Path, help="Output dir for YOLO-seg polygon labels.")
    p.add_argument("--detect-out", required=True, type=Path, help="Output dir for YOLO bbox labels.")
    p.add_argument("--donor-out", required=True, type=Path,
                   help="Output dir for donor-eligibility sidecar files: frame_<N>.txt with one line "
                        "per instance ('1'=eligible copy-paste donor, '0'=occluded or edge-cut), in the "
                        "SAME line order as the seg/detect label files -- this positional alignment is "
                        "what copy_paste_augment.py and filter_topk_classes.py rely on, since it survives "
                        "class-subset filtering as long as filtering preserves relative line order.")
    p.add_argument("--meta-csv", required=True, type=Path, help="Output CSV of per-instance metadata (for reporting).")
    p.add_argument("--edge-margin", type=int, default=2,
                   help="Pixels of slack when deciding if a bbox touches the frame edge (default: 2).")
    p.add_argument("--poly-epsilon-frac", type=float, default=0.003,
                   help="cv2.approxPolyDP epsilon as a fraction of the contour's arc length "
                        "(default 0.003 -- small, keeps shape detail).")
    p.add_argument("--min-frame", type=int, default=None, help="Only process frames >= this number.")
    p.add_argument("--max-frame", type=int, default=None, help="Only process frames <= this number.")
    return p.parse_args()


def decode_cvat_rle(rle, width, height):
    """CVAT mask RLE -> (height, width) uint8 binary mask, values in {0, 1}."""
    decoded = np.zeros(width * height, dtype=np.uint8)
    idx = 0
    value = 0
    for run in rle:
        if run:
            decoded[idx:idx + run] = value
        idx += run
        value = 1 - value
    return decoded.reshape(height, width)


def mask_to_polygon(local_mask, epsilon_frac):
    """Binary mask (local to its bbox crop) -> largest-contour polygon, as an
    (N, 2) float array of (x, y) pixel coords still local to that crop.
    Returns (polygon, n_contours_found)."""
    contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0
    largest = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon_frac * peri, True)
    poly = approx.reshape(-1, 2).astype(np.float64)
    if len(poly) < 3:
        # degenerate (near-1px) instance -- fall back to the unsimplified contour
        poly = largest.reshape(-1, 2).astype(np.float64)
    return poly, len(contours)


def main():
    args = parse_args()

    args.seg_out.mkdir(parents=True, exist_ok=True)
    args.detect_out.mkdir(parents=True, exist_ok=True)
    args.donor_out.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(args.xml)
    root = tree.getroot()

    xml_labels = [l.find("name").text for l in root.iter("label")]
    if xml_labels != NAMES:
        missing = set(NAMES) - set(xml_labels)
        extra = set(xml_labels) - set(NAMES)
        raise SystemExit(
            f"XML label taxonomy doesn't match the expected 58-species list!\n"
            f"  in expected but not XML: {missing}\n"
            f"  in XML but not expected: {extra}\n"
            f"Fix NAMES in this script (or re-export) before continuing -- class ids "
            f"would silently misalign otherwise."
        )

    meta_rows = []
    class_totals = Counter()
    class_eligible = Counter()  # non-occluded, non-edge-cut
    n_frames_written = 0
    n_frames_with_instances = 0
    n_multi_contour = 0
    n_degenerate = 0
    skipped_frames = 0

    for img in root.findall("image"):
        name = img.get("name")
        m = FRAME_RE.search(name)
        if not m:
            print(f"WARNING: couldn't parse a frame number from image name: {name!r} -- skipping")
            continue
        frame_num = int(m.group(1))
        if args.min_frame is not None and frame_num < args.min_frame:
            skipped_frames += 1
            continue
        if args.max_frame is not None and frame_num > args.max_frame:
            skipped_frames += 1
            continue

        img_w, img_h = int(img.get("width")), int(img.get("height"))

        seg_lines = []
        det_lines = []
        donor_lines = []
        for line_idx, mask_el in enumerate(img.findall("mask")):
            label = mask_el.get("label")
            cls_id = NAME_TO_ID[label]
            occluded = mask_el.get("occluded") == "1"
            left = int(mask_el.get("left"))
            top = int(mask_el.get("top"))
            bw = int(mask_el.get("width"))
            bh = int(mask_el.get("height"))
            rle = [int(x) for x in mask_el.get("rle").split(",")]

            edge_cut = (
                left <= args.edge_margin
                or top <= args.edge_margin
                or (left + bw) >= (img_w - args.edge_margin)
                or (top + bh) >= (img_h - args.edge_margin)
            )

            local_mask = decode_cvat_rle(rle, bw, bh)
            poly_local, n_contours = mask_to_polygon(local_mask, args.poly_epsilon_frac)
            if n_contours > 1:
                n_multi_contour += 1
            if poly_local is None or len(poly_local) < 3:
                n_degenerate += 1
                continue  # can't make a usable polygon or bbox from an empty/degenerate mask

            # shift polygon from bbox-local coords to full-frame pixel coords, normalize
            poly_full = poly_local + np.array([left, top])
            poly_norm = poly_full / np.array([img_w, img_h])
            poly_norm = np.clip(poly_norm, 0.0, 1.0)
            seg_lines.append(
                f"{cls_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poly_norm)
            )

            xc = (left + bw / 2) / img_w
            yc = (top + bh / 2) / img_h
            wn = bw / img_w
            hn = bh / img_h
            det_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

            eligible = (not occluded) and (not edge_cut)
            donor_lines.append("1" if eligible else "0")

            class_totals[cls_id] += 1
            if eligible:
                class_eligible[cls_id] += 1

            meta_rows.append({
                "frame": frame_num, "line_index": len(seg_lines) - 1,
                "class_id": cls_id, "class_name": label,
                "occluded": int(occluded), "edge_cut": int(edge_cut),
                "left": left, "top": top, "width": bw, "height": bh,
                "n_contours": n_contours,
            })

        stem = f"frame_{frame_num}"
        (args.seg_out / f"{stem}.txt").write_text("\n".join(seg_lines) + ("\n" if seg_lines else ""))
        (args.detect_out / f"{stem}.txt").write_text("\n".join(det_lines) + ("\n" if det_lines else ""))
        (args.donor_out / f"{stem}.txt").write_text("\n".join(donor_lines) + ("\n" if donor_lines else ""))
        n_frames_written += 1
        if seg_lines:
            n_frames_with_instances += 1

    with open(args.meta_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()) if meta_rows else
                                 ["frame", "line_index", "class_id", "class_name", "occluded",
                                  "edge_cut", "left", "top", "width", "height", "n_contours"])
        writer.writeheader()
        writer.writerows(meta_rows)

    print(f"frames processed: {n_frames_written} ({skipped_frames} skipped by --min-frame/--max-frame)")
    print(f"frames with >=1 instance: {n_frames_with_instances}")
    print(f"total instances written: {sum(class_totals.values())}")
    print(f"instances with a degenerate/empty mask (dropped): {n_degenerate}")
    print(f"instances whose mask decoded to >1 disconnected blob (kept, largest contour used): {n_multi_contour}")
    print(f"\n{'class':<6}{'name':<32}{'total':<8}{'eligible_donor':<16}")
    for cid in range(len(NAMES)):
        if class_totals[cid]:
            print(f"{cid:<6}{NAMES[cid]:<32}{class_totals[cid]:<8}{class_eligible[cid]:<16}")
    print(f"\nTOTAL{'':<32}{sum(class_totals.values()):<8}{sum(class_eligible.values()):<16}")
    print(f"\nwrote seg labels to {args.seg_out}, detect labels to {args.detect_out}, metadata to {args.meta_csv}")


if __name__ == "__main__":
    main()
