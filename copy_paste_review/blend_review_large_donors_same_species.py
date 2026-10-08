"""
Combines the two follow-up ideas so far: poisson/seamless-clone, restricted to the LARGEST
eligible donors per species (small donors blend badly under poisson -- see
blend_review/poisson_seamless_clone/), AND targets restricted to frames that already contain
that species (blend_review_same_species_target/'s hypothesis -- similar lighting/water
conditions may help colour matching). Poisson only, per your last answer -- naive/matched keep
using the full donor range in the other review folders.

Run across all 4 species. Each species only has a handful of donors large enough to qualify,
so (as with the earlier Raccoon Butterflyfish-only version) a random rotation (0/90/180/270)
and/or horizontal/vertical flip is applied to the donor patch+mask before each paste, to get a
decent sample size out of a small donor pool -- recorded in the filename.

Imports straight from blend_review_generate.py (must be in the same folder).
"""
import random

import cv2
import numpy as np

import blend_review_generate as base
from blend_review_large_donors import random_transform

SPECIES_LIST = ["Raccoon Butterflyfish", "Forcepsfish", "Ornate Butterfly Fish", "Moorish Idol"]
N_LARGEST_DONORS = 10
N_EXAMPLES_PER_SPECIES = 100
OUT_DIR = base.ROOT / "poisson_large_donors_same_species_target"


def main():
    dataset = base.parse_cvat_annotations(base.ANNOTATIONS_XML)
    frame_index = base.build_frame_index(base.IMAGES_DIR)
    print(f"Parsed {len(dataset)} images. Indexed {len(frame_index)} real frames.")
    if not frame_index:
        print("No real frames found -- aborting.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for species in SPECIES_LIST:
        donors = [(n, inst) for n, inst in base.eligible_donors(dataset, species)
                  if dataset[n]["frame"] in frame_index]
        if not donors:
            print(f"{species}: no local donors -- skipping")
            continue
        donors.sort(key=lambda nd: min(nd[1]["bbox"][2], nd[1]["bbox"][3]), reverse=True)
        largest = donors[:N_LARGEST_DONORS]
        print(f"\n{species}: using {len(largest)} largest donors (by bbox shorter side):")
        for n, inst in largest:
            w, h = inst["bbox"][2], inst["bbox"][3]
            print(f"  frame {dataset[n]['frame']}: {w}x{h}px")

        targets = base.species_target_frames(dataset, frame_index, species)
        print(f"{species}: {len(targets)} target frames already containing this species.")
        if not targets:
            print(f"{species}: no valid target frames -- skipping")
            continue

        out_dir = OUT_DIR / species.replace(" ", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(f"{base.RNG_SEED}:large_donor_poisson_same_species:{species}")

        written = 0
        attempts = 0
        max_attempts = N_EXAMPLES_PER_SPECIES * 25
        while written < N_EXAMPLES_PER_SPECIES and attempts < max_attempts:
            attempts += 1
            donor_name, donor_inst = rng.choice(largest)
            target_name, target_d = rng.choice(targets)

            donor_patch, donor_mask = base.get_donor_patch(dataset, frame_index, donor_name, donor_inst)
            target_img = base.get_frame(target_d["frame"], frame_index)
            if donor_patch is None or target_img is None:
                continue

            donor_patch, donor_mask, tag = random_transform(donor_patch, donor_mask, rng)
            mh, mw = donor_mask.shape
            if mh >= target_img.shape[0] or mw >= target_img.shape[1]:
                continue

            existing = [inst["bbox"] for inst in target_d["instances"]]
            x, y = base.find_placement(target_img.shape, (mh, mw), existing, rng=rng)

            result = base.paste_poisson(target_img, donor_patch, donor_mask, x, y)

            pad = max(30, int(0.6 * max(mh, mw)))
            overlay = base.mask_contour_overlay(result, donor_mask, x, y)
            zoom_outline_crop = base.crop_with_padding(overlay, x, y, mh, mw, pad)
            composite = base.compose_side_by_side(result, zoom_outline_crop)

            donor_frame = dataset[donor_name]["frame"]
            target_frame = target_d["frame"]
            fname = f"{written:03d}_donor{donor_frame}_target{target_frame}{tag}.png"
            cv2.imwrite(str(out_dir / fname), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
            written += 1

        print(f"{species}: wrote {written} composite images -> {out_dir}")

    print(f"\nDone. See {OUT_DIR}")


if __name__ == "__main__":
    main()
