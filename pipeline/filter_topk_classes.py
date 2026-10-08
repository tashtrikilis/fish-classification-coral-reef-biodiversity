#!/usr/bin/env python3
"""
Build a filtered dataset containing only a chosen subset of species (e.g. the
top 10 most common), remapped to contiguous class ids 0..N-1, reusing the
existing v2 restratified train/val split -- it doesn't re-derive a split from
scratch, it just filters and renumbers the labels of dataset_seg_v2 /
dataset_detect_v2 in place into a new folder, copying images across
unchanged.

Why reuse the v2 split rather than rebuild one: restratify_split.py's block
stratification guarantees train+val coverage for every class that spans 2+
blocks. The species being kept here are by construction the most frequent in
the dataset (tens to hundreds of instances each), so they are essentially
guaranteed to already span many blocks and have solid coverage on both sides
under the existing v2 assignment -- rebuilding the split from scratch for a
10-class subset would just reinvent work restratify_split.py already did.
This script's summary table is the check on that assumption: it prints
train/val instance counts for every kept class so you can confirm none of
them accidentally ended up 0 on either side before you spend GPU time on it.

Works for both the segmentation dataset (polygon labels: "<cls> x1 y1 x2 y2
...") and the detection dataset (bbox labels: "<cls> xc yc w h") -- both are
plain-text YOLO label files where only the first token (class id) needs to
change, so the same filtering logic handles either without touching
coordinates.

Frames that lose every instance (because their only labelled fish weren't in
the keep-list) are kept as background frames with an empty label file, not
deleted -- that matches how the original datasets already represent
frames with no labelled fish, and gives the model extra true-negative
examples for free rather than shrinking the dataset further.

Usage
-----
    python3 filter_topk_classes.py \\
        --src dataset_seg_v2 --dst dataset_seg_top10 \\
        --keep 53,33,10,2,19,31,35,0,29,34 \\
        --out-yaml data_seg_top10.yaml

    python3 filter_topk_classes.py \\
        --src dataset_detect_v2 --dst dataset_detect_top10 \\
        --keep 53,33,10,2,19,31,35,0,29,34 \\
        --out-yaml data_detect_top10.yaml

--keep is a comma-separated list of the ORIGINAL class ids to keep, in the
order you want them renumbered: the first id becomes new class 0, the second
becomes new class 1, and so on. Pass the exact same --keep list (same ids,
same order) to both the seg and detect calls so the two new datasets share
one class taxonomy.
"""
import argparse
import shutil
from collections import Counter
from pathlib import Path

# Same 58-species taxonomy as data_seg_v2.yaml / data_detect_v2.yaml / restratify_split.py.
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
    p.add_argument("--src", required=True, type=Path,
                   help="Existing dataset dir with images/{train,val} and labels/{train,val} "
                        "(e.g. dataset_seg_v2 or dataset_detect_v2).")
    p.add_argument("--dst", required=True, type=Path,
                   help="Output dataset dir to create (e.g. dataset_seg_top10).")
    p.add_argument("--keep", required=True, type=str,
                   help="Comma-separated ORIGINAL class ids to keep, in the order they should be "
                        "renumbered to 0..N-1, e.g. '53,33,10,2,19,31,35,0,29,34'.")
    p.add_argument("--out-yaml", type=Path, default=None,
                   help="If given, write a YOLO data.yaml here pointing at --dst with the "
                         "remapped class names.")
    p.add_argument("--img-glob", type=str, default="frame_*.jpg",
                   help="Glob for image files under src/images/<split>. Default: frame_*.jpg")
    p.add_argument("--donor-src", type=Path, default=None,
                   help="Optional: a donor_eligibility dir (from build_v3_dataset.py) paired with --src, "
                        "e.g. dataset_seg_v3/donor_eligibility. If given, its per-line eligibility flags "
                        "are filtered in lockstep with the label lines (same lines kept, same order) and "
                        "written to --donor-dst -- this is what keeps donor eligibility usable after "
                        "narrowing down to a class subset like top10.")
    p.add_argument("--donor-dst", type=Path, default=None,
                   help="Output dir for the filtered donor_eligibility files. Required if --donor-src is given.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite --dst if it already exists (default: refuse).")
    return p.parse_args()


def filter_split(src, dst, split, old_to_new, img_glob, donor_src=None, donor_dst=None):
    img_src_dir = src / "images" / split
    lbl_src_dir = src / "labels" / split
    img_dst_dir = dst / "images" / split
    lbl_dst_dir = dst / "labels" / split
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)
    donor_src_dir = donor_dst_dir = None
    if donor_src is not None:
        donor_src_dir = donor_src / split
        donor_dst_dir = donor_dst / split
        donor_dst_dir.mkdir(parents=True, exist_ok=True)

    if not img_src_dir.exists():
        return Counter(), Counter(), 0, 0

    inst_counts = Counter()      # new_class_id -> instance count
    frame_counts = Counter()     # new_class_id -> number of frames containing it
    n_frames = 0
    n_background = 0

    for img_path in sorted(img_src_dir.glob(img_glob)):
        n_frames += 1
        shutil.copy2(img_path, img_dst_dir / img_path.name)

        lbl_src = lbl_src_dir / f"{img_path.stem}.txt"
        kept_lines = []
        kept_donor_lines = []
        classes_in_frame = set()
        donor_lines_all = None
        if donor_src_dir is not None:
            donor_file = donor_src_dir / f"{img_path.stem}.txt"
            donor_lines_all = donor_file.read_text().splitlines() if donor_file.exists() else []
        if lbl_src.exists():
            # enumerate() gives the line's position in the ORIGINAL (unfiltered) label file,
            # which is exactly the index into donor_lines_all -- both were written in the same
            # per-instance order by cvat_masks_to_yolo.py / build_v3_dataset.py, and this loop
            # never skips a non-blank line without also skipping its donor entry.
            for orig_idx, line in enumerate(lbl_src.read_text().splitlines()):
                parts = line.split()
                if not parts:
                    continue
                old_cls = int(parts[0])
                if old_cls not in old_to_new:
                    continue  # drop instances of species not in the keep-list
                new_cls = old_to_new[old_cls]
                kept_lines.append(" ".join([str(new_cls)] + parts[1:]))
                inst_counts[new_cls] += 1
                classes_in_frame.add(new_cls)
                if donor_lines_all is not None:
                    kept_donor_lines.append(donor_lines_all[orig_idx] if orig_idx < len(donor_lines_all) else "0")

        for c in classes_in_frame:
            frame_counts[c] += 1
        if not kept_lines:
            n_background += 1

        (lbl_dst_dir / f"{img_path.stem}.txt").write_text(
            "\n".join(kept_lines) + ("\n" if kept_lines else "")
        )
        if donor_dst_dir is not None:
            (donor_dst_dir / f"{img_path.stem}.txt").write_text(
                "\n".join(kept_donor_lines) + ("\n" if kept_donor_lines else "")
            )

    return inst_counts, frame_counts, n_frames, n_background


def main():
    args = parse_args()

    keep_ids = [int(x) for x in args.keep.split(",") if x.strip() != ""]
    if len(set(keep_ids)) != len(keep_ids):
        raise SystemExit(f"--keep has duplicate ids: {args.keep}")
    old_to_new = {old: new for new, old in enumerate(keep_ids)}
    new_names = {new: NAMES.get(old, f"class_{old}") for old, new in old_to_new.items()}

    if args.dst.exists():
        if not args.force:
            raise SystemExit(f"{args.dst} already exists -- pass --force to overwrite into it")
    args.dst.mkdir(parents=True, exist_ok=True)

    if not args.src.exists():
        raise SystemExit(f"--src {args.src} does not exist")
    if (args.donor_src is None) != (args.donor_dst is None):
        raise SystemExit("--donor-src and --donor-dst must be given together")

    print(f"keeping {len(keep_ids)} classes, renumbered 0..{len(keep_ids)-1}:")
    for old, new in old_to_new.items():
        print(f"    {old:2d} ({NAMES.get(old, '?'):30s}) -> {new}")
    print()

    totals_inst = Counter()
    totals_frame = Counter()
    per_split = {}
    for split in ("train", "val"):
        inst_counts, frame_counts, n_frames, n_background = filter_split(
            args.src, args.dst, split, old_to_new, args.img_glob,
            donor_src=args.donor_src, donor_dst=args.donor_dst,
        )
        per_split[split] = (inst_counts, frame_counts, n_frames, n_background)
        for c, n in inst_counts.items():
            totals_inst[c] += n
        for c, n in frame_counts.items():
            totals_frame[c] += n
        print(f"{split}: {n_frames} frames copied, {n_background} now background "
              f"(no kept-class instances), {sum(inst_counts.values())} kept instances")

    print(f"\n{'class':<6}{'name':<30}{'train_inst':<12}{'val_inst':<10}{'train_frm':<11}{'val_frm':<10}")
    problems = []
    for new in range(len(keep_ids)):
        tr_i = per_split["train"][0].get(new, 0)
        va_i = per_split["val"][0].get(new, 0)
        tr_f = per_split["train"][1].get(new, 0)
        va_f = per_split["val"][1].get(new, 0)
        print(f"{new:<6}{new_names[new]:<30}{tr_i:<12}{va_i:<10}{tr_f:<11}{va_f:<10}")
        if tr_i == 0 or va_i == 0:
            problems.append((new, new_names[new], tr_i, va_i))

    if problems:
        print(f"\n!!! {len(problems)} class(es) have ZERO instances in train or val after filtering:")
        for new, name, tr_i, va_i in problems:
            print(f"    class {new} ({name}): train={tr_i} val={va_i}")
        print("    This subset can't reuse the v2 split as-is for those classes -- either drop them,")
        print("    swap in a different species, or rebuild a split scoped to just this keep-list.")
    else:
        print("\nall kept classes have >=1 instance in both train and val -- the v2 split carries over cleanly.")

    if args.out_yaml is not None:
        names_block = "\n".join(f"  {i}: {new_names[i]}" for i in range(len(keep_ids)))
        yaml_text = (
            f"# Auto-generated by filter_topk_classes.py from {args.src} -- {len(keep_ids)}-species subset,\n"
            f"# reusing the v2 restratified train/val split (only the label files were filtered +\n"
            f"# renumbered; the image/split assignment is unchanged from {args.src}).\n"
            f"# Original -> new class id map: "
            + ", ".join(f"{old}->{new}" for old, new in old_to_new.items()) + "\n\n"
            f"path: {args.dst.name}\n"
            f"train: images/train\n"
            f"val: images/val\n\n"
            f"names:\n{names_block}\n"
        )
        args.out_yaml.write_text(yaml_text)
        print(f"\nwrote {args.out_yaml}")


if __name__ == "__main__":
    main()
