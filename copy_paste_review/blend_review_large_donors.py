"""
Follow-up to blend_review_generate.py: poisson/seamless-clone specifically on the LARGEST
Raccoon Butterflyfish donors, after noticing small donors blend badly under poisson (see
blend_review/poisson_seamless_clone/Raccoon_Butterflyfish/) while large ones look fine.

Only ~N_LARGEST_DONORS donors are big enough to be worth testing this way, so to get a decent
sample size each paste applies a random SMALL-ANGLE rotation (+/-MAX_ROTATION_DEG, continuous,
not 90-degree steps -- a reef fish is basically never oriented exactly vertically, so rotating
a full 90 would just be mislabelling a bad pose as a bad blend) and/or a horizontal flip (a
realistic left/right swim-direction mirror) to the donor patch+mask before blending -- same
donor pixels, a slightly different pose each time. Vertical flip is deliberately NOT used --
fish don't swim upside-down. The transform applied is recorded in the filename.

Imports straight from blend_review_generate.py (must be in the same folder) to reuse its CVAT
parsing, image loading, and blend functions rather than duplicating them.
"""
import random

import cv2
import numpy as np

import blend_review_generate as base

SPECIES = "Raccoon Butterflyfish"
N_LARGEST_DONORS = 10
N_EXAMPLES = 100
MAX_ROTATION_DEG = 20  # a realistic amount of roll/pitch variation, not a full quarter-turn
OUT_DIR = base.ROOT / "poisson_large_donors_review" / SPECIES.replace(" ", "_")


def random_transform(patch: np.ndarray, mask: np.ndarray, rng: random.Random,
                      max_angle: float = MAX_ROTATION_DEG):
    """Randomly rotate a donor patch+mask by a small continuous angle (+/-max_angle degrees,
    NOT a 90-degree step) and/or flip horizontally. Returns the transformed patch, mask, and a
    filename tag recording what was applied."""
    angle = rng.uniform(-max_angle, max_angle)
    h, w = mask.shape

    # Pad out to the rotated diagonal first so the rotation doesn't clip the fish's corners,
    # then warpAffine (mask with nearest-neighbour so it stays strictly binary), then re-tighten
    # the crop to the rotated mask's actual extent (otherwise every patch would carry a growing
    # border of blank padding).
    diag = int(np.ceil(np.hypot(h, w))) + 4
    pad_h, pad_w = (diag - h) // 2 + 2, (diag - w) // 2 + 2
    patch_p = cv2.copyMakeBorder(patch, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    mask_p = cv2.copyMakeBorder(mask, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    ph, pw = mask_p.shape

    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    patch_r = cv2.warpAffine(patch_p, M, (pw, ph), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_r = cv2.warpAffine(mask_p, M, (pw, ph), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    ys, xs = np.where(mask_r > 0)
    if len(ys) == 0:
        # degenerate (shouldn't happen at these small angles) -- fall back to the untouched donor
        return patch, mask, "_orig"
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch_r = patch_r[y0:y1, x0:x1]
    mask_r = mask_r[y0:y1, x0:x1]

    label_parts = [f"rot{angle:+.0f}"]
    if rng.random() < 0.5:
        patch_r = np.fliplr(patch_r)
        mask_r = np.fliplr(mask_r)
        label_parts.append("flipH")

    patch_r = np.ascontiguousarray(patch_r)
    mask_r = np.ascontiguousarray(mask_r)
    tag = "_" + "_".join(label_parts)
    return patch_r, mask_r, tag


def main():
    dataset = base.parse_cvat_annotations(base.ANNOTATIONS_XML)
    frame_index = base.build_frame_index(base.IMAGES_DIR)
    print(f"Parsed {len(dataset)} images. Indexed {len(frame_index)} real frames.")
    if not frame_index:
        print("No real frames found -- aborting.")
        return

    donors = [(n, inst) for n, inst in base.eligible_donors(dataset, SPECIES)
              if dataset[n]["frame"] in frame_index]
    if not donors:
        print(f"No local donors for {SPECIES} -- aborting.")
        return

    donors.sort(key=lambda nd: min(nd[1]["bbox"][2], nd[1]["bbox"][3]), reverse=True)
    largest = donors[:N_LARGEST_DONORS]
    print(f"Using the {len(largest)} largest {SPECIES} donors (by bbox shorter side):")
    for n, inst in largest:
        w, h = inst["bbox"][2], inst["bbox"][3]
        print(f"  frame {dataset[n]['frame']}: {w}x{h}px")

    targets_all = base.sparse_target_frames(dataset, frame_index)  # excludes the pole/open-water view
    print(f"{len(targets_all)} sparse target frames available locally.")
    if not targets_all:
        print("No local target frames -- aborting.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"{base.RNG_SEED}:large_donor_poisson")

    written = 0
    attempts = 0
    max_attempts = N_EXAMPLES * 25
    while written < N_EXAMPLES and attempts < max_attempts:
        attempts += 1
        donor_name, donor_inst = rng.choice(largest)
        target_name, target_d = rng.choice(targets_all)

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
        cv2.imwrite(str(OUT_DIR / fname), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        written += 1

    print(f"\nDone. Wrote {written} composite images -> {OUT_DIR}")


if __name__ == "__main__":
    main()
