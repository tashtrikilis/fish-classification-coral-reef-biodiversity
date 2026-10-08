#!/usr/bin/env python3
"""
Rebuild the train/val split for dataset_seg (and, applying the identical
frame assignment, dataset_detect) so that classes with enough instances to
be split are represented in BOTH train and val -- instead of the original
single temporal cutoff, which happened to separate several species from
their own examples entirely (see conversation notes: 5 species were val-only
with zero training exposure, 20 species were train-only and never evaluated).

Approach: group the temporally-ordered frames into small contiguous BLOCKS
(so we still avoid the near-duplicate-consecutive-frame leakage a pure
temporal split was meant to prevent), then greedily assign blocks to val so
that every class with instances spread across 2+ blocks gets at least one
block on each side, before filling the rest of the val budget by shuffled
assignment. Classes confined to a single block can't be split without
breaking temporal locality -- those are reported, not silently ignored.

Writes fresh dataset_seg_v2/ and dataset_detect_v2/ folders (originals are
left untouched) plus data_seg_v2.yaml / data_detect_v2.yaml.

Usage:
    python3 restratify_split.py            # dry run: report only, no files written
    python3 restratify_split.py --write    # actually build the v2 dataset folders
"""
import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

random.seed(42)

BASE = Path(__file__).resolve().parent
SEG_SRC = BASE / "dataset_seg"
DET_SRC = BASE / "dataset_detect"
SEG_DST = BASE / "dataset_seg_v2"
DET_DST = BASE / "dataset_detect_v2"

BLOCK_SIZE = 12          # consecutive frames per block
TARGET_VAL_FRACTION = 0.20


def load_frames(dataset_dir):
    """Return {frame_num: {"img": path, "lbl": path, "classes": set, "counts": Counter}}"""
    frames = {}
    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("frame_*.jpg")):
            frame_num = int(img_path.stem.split("_")[1])
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            classes = []
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.split()
                    if parts:
                        classes.append(int(parts[0]))
            frames[frame_num] = {"img": img_path, "lbl": lbl_path if lbl_path.exists() else None, "classes": classes}
    return frames


def build_blocks(frame_nums, block_size):
    frame_nums = sorted(frame_nums)
    return [frame_nums[i : i + block_size] for i in range(0, len(frame_nums), block_size)]


def stratify(frames, blocks):
    """Greedily assign each block to 'train' or 'val'. Returns dict block_idx -> split,
    plus a report of which classes ended up covered on both sides / one side only."""
    class_blocks = defaultdict(set)   # class -> set of block indices containing it
    class_total = defaultdict(int)    # class -> total instance count
    for bi, block in enumerate(blocks):
        for fn in block:
            for c in frames[fn]["classes"]:
                class_blocks[c].add(bi)
                class_total[c] += 1

    n_frames_total = sum(len(b) for b in blocks)
    target_val_frames = int(round(n_frames_total * TARGET_VAL_FRACTION))

    assignment = {}  # block_idx -> "train" | "val"

    # Rarest classes first: force at least one block to val and one to train
    # for every class that spans 2+ blocks.
    for c in sorted(class_total, key=lambda c: class_total[c]):
        bset = class_blocks[c]
        if len(bset) < 2:
            continue  # can't split a single-block class without breaking temporal locality
        assigned_here = [assignment.get(bi) for bi in bset]
        if "val" not in assigned_here:
            # pick the smallest unassigned (or already-train) block containing c for val
            candidates = [bi for bi in bset if assignment.get(bi) != "val"]
            candidates.sort(key=lambda bi: len(blocks[bi]))
            pick = candidates[0]
            assignment[pick] = "val"
        if "train" not in assigned_here + [assignment.get(bi) for bi in bset]:
            candidates = [bi for bi in bset if assignment.get(bi) != "train"]
            candidates.sort(key=lambda bi: len(blocks[bi]))
            pick = candidates[0]
            assignment[pick] = "train"

    # Fill remaining blocks to hit the overall val-fraction target
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


def report(assignment, blocks, frames, class_blocks, class_total, names=None):
    train_frames, val_frames = set(), set()
    for bi, split in assignment.items():
        for fn in blocks[bi]:
            (train_frames if split == "train" else val_frames).add(fn)

    both, train_only, val_only, single_block = [], [], [], []
    for c in sorted(class_total):
        in_val = any(c in frames[fn]["classes"] for fn in val_frames)
        in_train = any(c in frames[fn]["classes"] for fn in train_frames)
        label = names[c] if names else str(c)
        if len(class_blocks[c]) < 2:
            single_block.append((c, label, class_total[c]))
        elif in_train and in_val:
            both.append((c, label, class_total[c]))
        elif in_train:
            train_only.append((c, label, class_total[c]))
        elif in_val:
            val_only.append((c, label, class_total[c]))

    print(f"\ntotal frames: {len(frames)}  ->  train: {len(train_frames)}  val: {len(val_frames)}")
    print(f"classes present in BOTH train and val: {len(both)}")
    print(f"classes TRAIN-only (multi-block, still not in val): {len(train_only)}")
    for c, label, n in train_only:
        print(f"    class {c:2d} {label:35s} n={n}")
    print(f"classes VAL-only (multi-block, still not in train): {len(val_only)}")
    for c, label, n in val_only:
        print(f"    class {c:2d} {label:35s} n={n}")
    print(f"classes confined to a single block (can't be split at all): {len(single_block)}")
    for c, label, n in single_block:
        print(f"    class {c:2d} {label:35s} n={n}")

    return train_frames, val_frames


def write_dataset(frames, train_frames, val_frames, src_dir, dst_dir):
    for split, frame_set in (("train", train_frames), ("val", val_frames)):
        (dst_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    for fn, info in frames.items():
        split = "train" if fn in train_frames else "val"
        img_dst = dst_dir / "images" / split / info["img"].name
        shutil.copy2(info["img"], img_dst)
        lbl_name = info["img"].stem + ".txt"
        lbl_dst = dst_dir / "labels" / split / lbl_name
        if info["lbl"] is not None:
            shutil.copy2(info["lbl"], lbl_dst)
        else:
            lbl_dst.write_text("")  # background frame, empty label


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="actually build dataset_seg_v2/dataset_detect_v2")
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    args = ap.parse_args()

    seg_frames = load_frames(SEG_SRC)
    print(f"loaded {len(seg_frames)} frames from {SEG_SRC}")

    blocks = build_blocks(list(seg_frames.keys()), args.block_size)
    print(f"{len(blocks)} blocks of ~{args.block_size} frames each")

    assignment, class_blocks, class_total = stratify(seg_frames, blocks)
    train_frames, val_frames = report(assignment, blocks, seg_frames, class_blocks, class_total, names=NAMES)

    if not args.write:
        print("\n(dry run -- rerun with --write to actually build dataset_seg_v2/dataset_detect_v2)")
        return

    print(f"\nwriting {SEG_DST} ...")
    write_dataset(seg_frames, train_frames, val_frames, SEG_SRC, SEG_DST)

    if DET_SRC.exists():
        det_frames = load_frames(DET_SRC)
        # sanity check: detect dataset should have the exact same frame set as seg
        missing = set(seg_frames) - set(det_frames)
        if missing:
            print(f"WARNING: {len(missing)} frames present in dataset_seg but missing from dataset_detect: {sorted(missing)[:10]}...")
        print(f"writing {DET_DST} ...")
        write_dataset(det_frames, train_frames, val_frames, DET_SRC, DET_DST)

    print("done.")


if __name__ == "__main__":
    main()
