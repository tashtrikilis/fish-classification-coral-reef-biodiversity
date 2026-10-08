#!/usr/bin/env python3
"""
Build dataset_seg_v3 / dataset_detect_v3 from the CVAT mask export (already
converted by cvat_masks_to_yolo.py into flat dataset_seg_labels_v3 /
dataset_detect_labels_v3 / dataset_donor_v3 label dirs) plus whatever raw
frame images are actually available on this machine.

Unlike restratify_split.py (which re-splits an EXISTING train/val split),
this builds train/val from scratch out of a flat pool of frames, using the
same block-stratification approach: group frames into contiguous blocks,
force every class spanning 2+ blocks onto both sides, then fill the
remaining val quota by shuffled block assignment.

Frame-availability handling: --min-frame/--max-frame define the "reviewed"
frame range (default 0-1911, matching the CVAT export's actual labelled
span -- frames from 1912-2816 exist in the video but have no evidence of
being reviewed, so they're excluded entirely, not treated as background).
Within that range, a frame is only included if its image file actually
exists in --images-dir -- if it doesn't, the frame is skipped and reported,
never silently treated as background. Run this and read the "missing
images" count before trusting the resulting dataset: if it's large, you're
training on a smaller span than you think and should transfer the missing
frames first.

Usage
-----
    python3 build_v3_dataset.py \\
        --images-dir /path/to/full_dataset \\
        --seg-labels dataset_seg_labels_v3 --detect-labels dataset_detect_labels_v3 \\
        --donor dataset_donor_v3 \\
        --seg-dst dataset_seg_v3 --detect-dst dataset_detect_v3 \\
        --seg-yaml data_seg_v3.yaml --detect-yaml data_detect_v3.yaml \\
        --min-frame 0 --max-frame 1911
"""
import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

random.seed(42)

NAMES = {
    0: "Acanthuridae", 1: "Black Triggerfish", 2: "Blackfin Chromis", 3: "Blackspot Sergeant",
    4: "Blue Goatfish", 5: "Bluefin Trevally", 6: "Bluelined Surgeonfish", 7: "Bluespine Unicornfish",
    8: "Bluestripe Snapper", 9: "Bridled Triggerfish", 10: "Brown Surgeonfish", 11: "Bullethead Unicornfish",
    12: "Bullethead Parrotfish - Female", 13: "Chevron Tang", 14: "Chinese Trumpetfish", 15: "Convict Tang",
    16: "Doublebar Goatfish", 17: "Forcepsfish", 18: "Fourspot Butterflyfish", 19: "Goldring Surgeonfish",
    20: "Hawaiian Cleaner Wrasse", 21: "Hawaiian Gregory", 22: "Hawaiian Sergeant", 23: "Hawaiian Whitespotted Toby",
    24: "Highfin Chub", 25: "Indo-Pacific Sergeant", 26: "Lei Triggerfish", 27: "Lined Butterflyfish",
    28: "Lognnose Butterflyfish", 29: "Lowfin Chub", 30: "Manybar Goatfish", 31: "Moorish Idol",
    32: "Multiband Butterflyfish", 33: "Orangeband Surgeonfish", 34: "Orangespine Unicornfish",
    35: "Ornate Butterfly Fish", 36: "Oval Chromis", 37: "Palenose Parrotfish", 38: "Palenose Parrotfish - Male",
    39: "Peacock Grouper", 40: "Pinktail Triggerfish", 41: "Pyramid Butterflyfish", 42: "Raccoon Butterflyfish",
    43: "Redlip Parrotfish - Male", 44: "Redlip Parrotfish - Female", 45: "Speckled Parrotfish - Male",
    46: "Spotted Boxfish", 47: "Stareye Parrotfish - Male", 48: "Stripebelly Puffer", 49: "Threadfin Butterflyfish",
    50: "Whitebar Surgeonfish", 51: "Whitecheek Surgeonfish", 52: "Whitespotted Surgeonfish", 53: "Yellow Tang",
    54: "Yellowfin Goatfish", 55: "Yellowtail Coris", 56: "Ringtail Surgeonfish", 57: "Saddle Wrasse - Male",
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True, type=Path, help="Flat dir of frame_<N>.jpg images.")
    p.add_argument("--seg-labels", required=True, type=Path, help="dataset_seg_labels_v3 (from cvat_masks_to_yolo.py).")
    p.add_argument("--detect-labels", required=True, type=Path, help="dataset_detect_labels_v3.")
    p.add_argument("--donor", required=True, type=Path, help="dataset_donor_v3.")
    p.add_argument("--seg-dst", required=True, type=Path)
    p.add_argument("--detect-dst", required=True, type=Path)
    p.add_argument("--seg-yaml", type=Path, default=None)
    p.add_argument("--detect-yaml", type=Path, default=None)
    p.add_argument("--min-frame", type=int, default=0)
    p.add_argument("--max-frame", type=int, default=1911)
    p.add_argument("--block-size", type=int, default=12)
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--priority-classes", type=str, default=None,
                   help="Comma-separated class ids. If given, the 'force every class spanning 2+ "
                        "blocks onto both sides' step only considers THESE classes -- freeing up "
                        "blocks that would otherwise get consumed forcing coverage for classes you "
                        "don't actually care about in this dataset variant (e.g. when building a "
                        "split meant to feed a top-10 subset). All 58 classes' instances are still "
                        "written to the output labels either way -- this only changes which classes' "
                        "coverage the greedy assignment prioritizes. Omit for the default 'whole "
                        "dataset' behavior (prioritize every class).")
    return p.parse_args()


def build_blocks(frame_nums, block_size):
    frame_nums = sorted(frame_nums)
    return [frame_nums[i:i + block_size] for i in range(0, len(frame_nums), block_size)]


def stratify(frame_classes, blocks, val_fraction, priority_classes=None):
    class_blocks = defaultdict(set)
    class_total = defaultdict(int)
    for bi, block in enumerate(blocks):
        for fn in block:
            for c in frame_classes[fn]:
                class_blocks[c].add(bi)
                class_total[c] += 1

    n_frames_total = sum(len(b) for b in blocks)
    target_val_frames = int(round(n_frames_total * val_fraction))

    force_classes = set(class_total) if priority_classes is None else (set(class_total) & priority_classes)

    # Force every priority class spanning 2+ blocks onto both sides. Prefer a block
    # nobody has claimed yet over reassigning one already forced for a different
    # class -- the original (unconditional-overwrite) version could silently steal
    # a block back from an earlier class's only val/train coverage when a later
    # class's requirement happened to overlap it. This doesn't make the greedy
    # assignment fully conflict-free (two classes confined to the exact same block
    # pair can still collide), but it removes the common case where a free
    # alternative block existed and just wasn't tried first.
    assignment = {}
    for c in sorted(force_classes, key=lambda c: class_total[c]):
        bset = class_blocks[c]
        if len(bset) < 2:
            continue
        if "val" not in [assignment.get(bi) for bi in bset]:
            unclaimed = [bi for bi in bset if bi not in assignment]
            candidates = unclaimed or [bi for bi in bset if assignment.get(bi) != "val"]
            candidates.sort(key=lambda bi: len(blocks[bi]))
            if candidates:
                assignment[candidates[0]] = "val"
        if "train" not in [assignment.get(bi) for bi in bset]:
            unclaimed = [bi for bi in bset if bi not in assignment]
            candidates = unclaimed or [bi for bi in bset if assignment.get(bi) != "train"]
            candidates.sort(key=lambda bi: len(blocks[bi]))
            if candidates:
                assignment[candidates[0]] = "train"

    val_frames_so_far = sum(len(blocks[bi]) for bi, s in assignment.items() if s == "val")
    unassigned = [bi for bi in range(len(blocks)) if bi not in assignment]
    random.shuffle(unassigned)
    for bi in unassigned:
        if val_frames_so_far < target_val_frames:
            assignment[bi] = "val"
            val_frames_so_far += len(blocks[bi])
        else:
            assignment[bi] = "train"

    return assignment, class_blocks, class_total


def main():
    args = parse_args()

    all_frame_nums = list(range(args.min_frame, args.max_frame + 1))
    included = []
    missing_images = []
    frame_classes = {}   # frame_num -> list of class ids present (for stratification)

    for fn in all_frame_nums:
        img_path = args.images_dir / f"frame_{fn}.jpg"
        seg_path = args.seg_labels / f"frame_{fn}.txt"
        if not img_path.exists():
            missing_images.append(fn)
            continue
        classes = []
        if seg_path.exists():
            for line in seg_path.read_text().splitlines():
                parts = line.split()
                if parts:
                    classes.append(int(parts[0]))
        frame_classes[fn] = classes
        included.append(fn)

    print(f"reviewed frame range: {args.min_frame}-{args.max_frame} ({len(all_frame_nums)} frames)")
    print(f"images found: {len(included)}")
    print(f"images MISSING (excluded entirely, not treated as background): {len(missing_images)}")
    if missing_images:
        print(f"  missing frame numbers (first 20): {missing_images[:20]}"
              + (" ..." if len(missing_images) > 20 else ""))
        print("  -> if this is a lot, transfer those frames from your full-resolution image pool before "
              "trusting this dataset for training.")

    if not included:
        raise SystemExit("no images found at all -- check --images-dir")

    priority_classes = None
    if args.priority_classes:
        priority_classes = {int(x) for x in args.priority_classes.split(",") if x.strip() != ""}
        print(f"prioritizing coverage for {len(priority_classes)} classes: {sorted(priority_classes)}")

    blocks = build_blocks(included, args.block_size)
    assignment, class_blocks, class_total = stratify(frame_classes, blocks, args.val_fraction, priority_classes)

    train_frames, val_frames = set(), set()
    for bi, split in assignment.items():
        for fn in blocks[bi]:
            (train_frames if split == "train" else val_frames).add(fn)

    both, train_only, val_only, single_block = [], [], [], []
    for c in sorted(class_total):
        in_val = any(c in frame_classes[fn] for fn in val_frames)
        in_train = any(c in frame_classes[fn] for fn in train_frames)
        label = NAMES.get(c, str(c))
        if len(class_blocks[c]) < 2:
            single_block.append((c, label, class_total[c]))
        elif in_train and in_val:
            both.append((c, label, class_total[c]))
        elif in_train:
            train_only.append((c, label, class_total[c]))
        elif in_val:
            val_only.append((c, label, class_total[c]))

    print(f"\ntotal usable frames: {len(included)}  ->  train: {len(train_frames)}  val: {len(val_frames)}")
    print(f"classes present in BOTH train and val: {len(both)}")
    print(f"classes TRAIN-only (multi-block, still not in val): {len(train_only)}")
    for c, label, n in train_only:
        print(f"    class {c:2d} {label:35s} n={n}")
    print(f"classes VAL-only (multi-block, still not in train): {len(val_only)}")
    for c, label, n in val_only:
        print(f"    class {c:2d} {label:35s} n={n}")
    print(f"classes confined to a single block (can't be split): {len(single_block)}")
    for c, label, n in single_block:
        print(f"    class {c:2d} {label:35s} n={n}")

    for split, frame_set in (("train", train_frames), ("val", val_frames)):
        (args.seg_dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.seg_dst / "labels" / split).mkdir(parents=True, exist_ok=True)
        (args.seg_dst / "donor_eligibility" / split).mkdir(parents=True, exist_ok=True)
        (args.detect_dst / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.detect_dst / "labels" / split).mkdir(parents=True, exist_ok=True)

    for fn in included:
        split = "train" if fn in train_frames else "val"
        stem = f"frame_{fn}"
        img_src = args.images_dir / f"{stem}.jpg"
        shutil.copy2(img_src, args.seg_dst / "images" / split / f"{stem}.jpg")
        shutil.copy2(img_src, args.detect_dst / "images" / split / f"{stem}.jpg")

        seg_src = args.seg_labels / f"{stem}.txt"
        det_src = args.detect_labels / f"{stem}.txt"
        donor_src = args.donor / f"{stem}.txt"
        (args.seg_dst / "labels" / split / f"{stem}.txt").write_text(seg_src.read_text() if seg_src.exists() else "")
        (args.detect_dst / "labels" / split / f"{stem}.txt").write_text(det_src.read_text() if det_src.exists() else "")
        (args.seg_dst / "donor_eligibility" / split / f"{stem}.txt").write_text(
            donor_src.read_text() if donor_src.exists() else ""
        )

    print(f"\nwrote {args.seg_dst} and {args.detect_dst}")

    for yaml_path, dst_name in ((args.seg_yaml, args.seg_dst.name), (args.detect_yaml, args.detect_dst.name)):
        if yaml_path is None:
            continue
        names_block = "\n".join(f"  {i}: {NAMES[i]}" for i in range(58))
        yaml_path.write_text(
            f"# Auto-generated by build_v3_dataset.py from the CVAT mask export "
            f"(frames {args.min_frame}-{args.max_frame}, block-stratified split, block_size={args.block_size}).\n"
            f"# {len(missing_images)} frames in that range had no image file and were excluded, not treated as background.\n\n"
            f"path: {dst_name}\ntrain: images/train\nval: images/val\n\nnames:\n{names_block}\n"
        )
        print(f"wrote {yaml_path}")


if __name__ == "__main__":
    main()
