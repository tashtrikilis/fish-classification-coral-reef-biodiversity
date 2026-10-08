"""
The "what you'd actually ship" gallery: matched blend (the production default), donors
filtered to min_dim>=20px (the fix established in the notebook's Moorish Idol investigation --
Section 11a/11c), targets from the standard unrestricted sparse pool (standard copy-paste
practice). This is deliberately NOT a diagnostic/stress-test folder like the others -- it's
meant to show what the real augmentation pipeline's output looks like once the known fix is
applied, for a fast gut-check against deadline pressure.

Same composite format as the other review folders (full scene + zoom/outline), but a smaller
N (30/species, matching the notebook's own gallery size) since this is a quick look, not an
exhaustive review.
"""
import random

import cv2

import blend_review_generate as base

SPECIES_LIST = ["Raccoon Butterflyfish", "Forcepsfish", "Ornate Butterfly Fish", "Moorish Idol"]
MIN_DIM = 20
N_PER_SPECIES = 30
OUT_DIR = base.ROOT / "final_recommended_matched_sizefiltered"


def main():
    dataset = base.parse_cvat_annotations(base.ANNOTATIONS_XML)
    frame_index = base.build_frame_index(base.IMAGES_DIR)
    print(f"Parsed {len(dataset)} images. Indexed {len(frame_index)} real frames.")
    if not frame_index:
        print("No real frames found -- aborting.")
        return

    targets = base.sparse_target_frames(dataset, frame_index)
    print(f"{len(targets)} sparse target frames available locally.")
    if not targets:
        print("No target frames -- aborting.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for species in SPECIES_LIST:
        all_donors = [(n, inst) for n, inst in base.eligible_donors(dataset, species)
                      if dataset[n]["frame"] in frame_index]
        donors = [(n, inst) for n, inst in all_donors
                  if min(inst["bbox"][2], inst["bbox"][3]) >= MIN_DIM]
        print(f"\n{species}: {len(donors)}/{len(all_donors)} donors kept at min_dim>={MIN_DIM}px")
        if not donors:
            print(f"{species}: nothing left after the size filter -- skipping")
            continue

        out_dir = OUT_DIR / species.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(f"{base.RNG_SEED}:final:{species}")

        written = 0
        attempts = 0
        max_attempts = N_PER_SPECIES * 25
        while written < N_PER_SPECIES and attempts < max_attempts:
            attempts += 1
            donor_name, donor_inst = rng.choice(donors)
            target_name, target_d = rng.choice(targets)

            donor_patch, donor_mask = base.get_donor_patch(dataset, frame_index, donor_name, donor_inst)
            target_img = base.get_frame(target_d["frame"], frame_index)
            if donor_patch is None or target_img is None:
                continue
            mh, mw = donor_mask.shape
            if mh >= target_img.shape[0] or mw >= target_img.shape[1]:
                continue

            existing = [inst["bbox"] for inst in target_d["instances"]]
            x, y = base.find_placement(target_img.shape, (mh, mw), existing, rng=rng)

            result = base.paste_matched_feathered(target_img, donor_patch, donor_mask, x, y)

            pad = max(30, int(0.6 * max(mh, mw)))
            overlay = base.mask_contour_overlay(result, donor_mask, x, y)
            zoom_outline_crop = base.crop_with_padding(overlay, x, y, mh, mw, pad)
            composite = base.compose_side_by_side(result, zoom_outline_crop)

            donor_frame = dataset[donor_name]["frame"]
            target_frame = target_d["frame"]
            fname = f"{written:03d}_donor{donor_frame}_target{target_frame}.png"
            cv2.imwrite(str(out_dir / fname), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            written += 1

        print(f"{species}: wrote {written} composite images -> {out_dir}")

    print(f"\nDone. See {OUT_DIR}")


if __name__ == "__main__":
    main()
