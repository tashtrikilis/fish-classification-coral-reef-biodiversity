"""
Generates a folder of copy-paste blend examples for visual review: for each of several
species, the SAME set of (donor, target, placement) combinations is rendered under three
blend methods -- naive ("pure paste"), matched (colour-corrected + feathered), and poisson
(cv2.seamlessClone) -- so the same paste can be compared apples-to-apples across methods.

For each paste, ONE image is written: a side-by-side composite, left half the whole scene
(the full target frame, no outline), right half a closer crop around the paste with the
donor's mask boundary outlined in magenta. Filenames encode a running index plus the donor
and target frame numbers, so a specific example can be reported back by name.

RESTRICT_TARGETS_TO_SAME_SPECIES: when True (the current setting), target frames are
restricted to frames that already contain >=1 real instance of the species being pasted --
NOT standard copy-paste practice (the literature usually pastes onto arbitrary frames for
context diversity), but tried here on the hypothesis that a frame already containing the
species was shot under similar lighting/water conditions, which may help `matched`-mode
colour correction specifically. This does NOT also require the target to be sparse (<=1
total instance) -- that combination was checked and is nearly empty for these species (0
frames for Ornate Butterfly Fish), so crowded multi-fish target frames are allowed. Set to
False to go back to the original unrestricted sparse-frame pool (see blend_review/, generated
that way, for comparison).

Run from the folder containing annotations.xml / data.yaml (copy_paste_species_demo/), with
the real frames one level up in full_dataset/ -- same layout the notebook itself assumes.
"""
import csv
import random
import re
from pathlib import Path

import cv2
import numpy as np

RNG_SEED = 42
MIN_FRAME = 1000
N_PER_SPECIES = 100
SPECIES_LIST = ["Raccoon Butterflyfish", "Forcepsfish", "Ornate Butterfly Fish", "Moorish Idol"]
METHODS = [("pure_paste", "naive"), ("matched_colour_corrected", "matched"),
           ("poisson_seamless_clone", "poisson")]
RESTRICT_TARGETS_TO_SAME_SPECIES = True

ROOT = Path(__file__).resolve().parent
ANNOTATIONS_XML = ROOT / "annotations.xml"
IMAGES_DIR = ROOT.parent / "full_dataset"
OUT_DIR = ROOT / ("blend_review_same_species_target" if RESTRICT_TARGETS_TO_SAME_SPECIES else "blend_review")

PANEL_HEIGHT = 480     # both panels of the side-by-side composite are resized to this height
SEPARATOR_PX = 6       # width of the white divider between the two panels
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------------------------------------------------------------------------
# CVAT parsing (same approach as copy_paste_demo.ipynb / notebook_source.py)
# ---------------------------------------------------------------------------
_IMAGE_BLOCK_RE = re.compile(r"<image\b[^>]*>.*?</image>|<image\b[^>]*/>", re.S)
_IMAGE_ATTR_RE = re.compile(r'id="(\d+)" name="([^"]+)"[^>]*width="(\d+)" height="(\d+)"')
_FRAME_RE = re.compile(r"mp4-(\d+)_jpg")
_MASK_RE = re.compile(
    r'<mask label="([^"]+)" source="[^"]*" occluded="(\d)" rle="([^"]*)" '
    r'left="(\d+)" top="(\d+)" width="(\d+)" height="(\d+)"'
)


def decode_cvat_rle(rle_str: str, width: int, height: int) -> np.ndarray:
    counts = [int(x) for x in rle_str.split(",")]
    total = width * height
    flat = np.zeros(total, dtype=np.uint8)
    pos, val = 0, 0
    for c in counts:
        if val == 1:
            flat[pos:pos + c] = 1
        pos += c
        val = 1 - val
    assert pos == total, f"RLE decoded {pos} pixels, expected {total} -- malformed mask"
    return flat.reshape((height, width))


def parse_cvat_annotations(xml_path: Path) -> dict:
    content = xml_path.read_text()
    images = {}
    for block in _IMAGE_BLOCK_RE.findall(content):
        attr = _IMAGE_ATTR_RE.search(block)
        if not attr:
            continue
        _, name, iw, ih = attr.group(1), attr.group(2), int(attr.group(3)), int(attr.group(4))
        fm = _FRAME_RE.search(name)
        frame = int(fm.group(1)) if fm else None
        instances = []
        for mm in _MASK_RE.finditer(block):
            label, occluded, rle, left, top, w, h = mm.groups()
            left, top, w, h = int(left), int(top), int(w), int(h)
            edge_touch = (left <= 0) or (top <= 0) or (left + w >= iw) or (top + h >= ih)
            instances.append({
                "label": label, "occluded": int(occluded), "rle": rle,
                "bbox": (left, top, w, h), "edge": edge_touch,
            })
        images[name] = {"frame": frame, "iw": iw, "ih": ih, "instances": instances}
    return images


def eligible_donors(dataset: dict, species: str, min_frame: int = MIN_FRAME):
    out = []
    for name, d in dataset.items():
        if d["frame"] is None or d["frame"] < min_frame:
            continue
        for inst in d["instances"]:
            if inst["label"] == species and inst["occluded"] == 0 and not inst["edge"]:
                out.append((name, inst))
    return out


# ---------------------------------------------------------------------------
# Image loading (real frames only -- frame_<N>.jpg under IMAGES_DIR)
# ---------------------------------------------------------------------------
def build_frame_index(images_dir: Path) -> dict:
    index = {}
    if images_dir.exists():
        for p in images_dir.glob("frame_*.jpg"):
            m = re.match(r"frame_(\d+)\.jpg$", p.name)
            if m:
                index[int(m.group(1))] = p
    return index


_FRAME_CACHE = {}


def get_frame(frame_no: int, frame_index: dict):
    if frame_no in _FRAME_CACHE:
        return _FRAME_CACHE[frame_no]
    p = frame_index.get(frame_no)
    if p is None:
        _FRAME_CACHE[frame_no] = None
        return None
    img = cv2.imread(str(p))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    _FRAME_CACHE[frame_no] = img
    return img


def get_donor_patch(dataset: dict, frame_index: dict, image_name: str, instance: dict):
    d = dataset[image_name]
    left, top, w, h = instance["bbox"]
    mask = decode_cvat_rle(instance["rle"], w, h)
    img = get_frame(d["frame"], frame_index)
    if img is None:
        return None, None
    patch = img[top:top + h, left:left + w]
    return patch, mask


# ---------------------------------------------------------------------------
# Blending (verbatim from copy_paste_demo.ipynb / notebook_source.py)
# ---------------------------------------------------------------------------
def color_transfer_lab(patch: np.ndarray, mask: np.ndarray, target_region: np.ndarray) -> np.ndarray:
    patch_lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_region, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = mask.astype(bool)
    if m.sum() < 5:
        return patch
    out = patch_lab.copy()
    channel_strength = (0.45, 0.15, 0.15)
    for c in range(3):
        src_vals = patch_lab[:, :, c][m]
        tgt_vals = target_lab[:, :, c]
        src_mean, src_std = src_vals.mean(), src_vals.std() + 1e-6
        tgt_mean, tgt_std = tgt_vals.mean(), tgt_vals.std() + 1e-6
        s = channel_strength[c]
        std_ratio = 1 + s * (np.clip(tgt_std / src_std, 0.8, 1.25) - 1)
        shift = np.clip(s * (tgt_mean - src_mean), -0.5 * src_std, 0.5 * src_std)
        out[:, :, c][m] = (src_vals - src_mean) * std_ratio + src_mean + shift
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def feather_mask(mask: np.ndarray, feather_px: int = 5) -> np.ndarray:
    m = (mask * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(m, (0, 0), sigmaX=feather_px)
    return (blurred.astype(np.float32) / 255.0)


def paste_naive(target: np.ndarray, patch: np.ndarray, mask: np.ndarray, x: int, y: int) -> np.ndarray:
    out = target.copy()
    h, w = mask.shape
    region = out[y:y + h, x:x + w]
    m = mask.astype(bool)
    region[m] = patch[m]
    out[y:y + h, x:x + w] = region
    return out


def paste_matched_feathered(target: np.ndarray, patch: np.ndarray, mask: np.ndarray,
                             x: int, y: int, feather_px: int = None) -> np.ndarray:
    h, w = mask.shape
    if feather_px is None:
        feather_px = max(1, min(5, round(0.06 * min(h, w))))
    local_bg = target[y:y + h, x:x + w]
    matched = color_transfer_lab(patch, mask, local_bg)
    alpha = feather_mask(mask, feather_px)[..., None]
    out = target.copy()
    region = out[y:y + h, x:x + w].astype(np.float32)
    blended = region * (1 - alpha) + matched.astype(np.float32) * alpha
    out[y:y + h, x:x + w] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def paste_poisson(target: np.ndarray, patch: np.ndarray, mask: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = mask.shape
    center = (x + w // 2, y + h // 2)
    mask255 = (mask * 255).astype(np.uint8)
    if mask255.sum() == 0:
        return target.copy()
    try:
        out = cv2.seamlessClone(patch, target, mask255, center, cv2.NORMAL_CLONE)
    except cv2.error:
        # seamlessClone can fail on tiny/degenerate masks -- fall back to the feathered blend,
        # same as the notebook. If a poisson example looks identical to its matched sibling,
        # this is why: seamlessClone errored on that particular placement.
        out = paste_matched_feathered(target, patch, mask, x, y)
    return out


def find_placement(target_shape, mask_shape, existing_bboxes, margin: int = 2, tries: int = 30,
                    rng: random.Random = random):
    th, tw = target_shape[:2]
    mh, mw = mask_shape
    x = y = margin
    for _ in range(tries):
        x = rng.randint(margin, max(tw - mw - margin, margin))
        y = rng.randint(margin, max(th - mh - margin, margin))
        new_box = (x, y, x + mw, y + mh)
        ok = True
        for (bx, by, bw, bh) in existing_bboxes:
            ex_box = (bx, by, bx + bw, by + bh)
            if not (new_box[2] < ex_box[0] or new_box[0] > ex_box[2] or
                    new_box[3] < ex_box[1] or new_box[1] > ex_box[3]):
                ok = False
                break
        if ok:
            return x, y
    return x, y


def mask_contour_overlay(image: np.ndarray, mask: np.ndarray, x: int, y: int,
                          color=(255, 0, 255), thickness: int = 2) -> np.ndarray:
    out = image.copy()
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    shifted = [c + np.array([[x, y]]) for c in contours]
    cv2.drawContours(out, shifted, -1, color, thickness)
    return out


def crop_with_padding(img: np.ndarray, x: int, y: int, mh: int, mw: int, pad: int):
    cy0, cy1 = max(y - pad, 0), min(y + mh + pad, img.shape[0])
    cx0, cx1 = max(x - pad, 0), min(x + mw + pad, img.shape[1])
    return img[cy0:cy1, cx0:cx1]


def resize_to_height(img: np.ndarray, target_h: int = PANEL_HEIGHT):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    scale = target_h / h
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(img, (max(1, int(round(w * scale))), target_h), interpolation=interp)


def label_panel(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a small label with a dark backing box into the top-left corner of `img`."""
    out = img.copy()
    (tw, th), baseline = cv2.getTextSize(text, LABEL_FONT, 0.6, 2)
    pad = 6
    cv2.rectangle(out, (0, 0), (tw + 2 * pad, th + baseline + 2 * pad), (0, 0, 0), -1)
    cv2.putText(out, text, (pad, th + pad), LABEL_FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def compose_side_by_side(full_scene: np.ndarray, zoom_outline: np.ndarray) -> np.ndarray:
    left = label_panel(resize_to_height(full_scene), "FULL SCENE")
    right = label_panel(resize_to_height(zoom_outline), "ZOOM + OUTLINE")
    sep = np.full((PANEL_HEIGHT, SEPARATOR_PX, 3), 255, dtype=np.uint8)
    return np.hstack([left, sep, right])


# ---------------------------------------------------------------------------
# Acceptor (paste target) view filter
# ---------------------------------------------------------------------------
# The camera pans slowly and periodically faces the mooring pole / open water: a paler,
# washed-out, emptier view whose colour differs from everything else, so nothing may be
# pasted onto it (Tash's rule, 2026-10-07). Tash reviewed one frame per minute of footage
# (acceptor_gallery in donor_acceptor_review.ipynb) and marked each kept or dropped.
# A frame is an allowed acceptor only if the reviewed samples on BOTH sides of it were kept
# (or it IS a kept sample). Frames between a kept and a dropped sample fall in a minute where
# the camera was mid-pan and are excluded rather than guessed. Frames before the first or
# after the last sample are excluded too.
ACCEPTOR_REVIEW_SAMPLES = [  # (frame, kept?) from the reviewed acceptor gallery, #0..#60
    (1016, 0), (1054, 0), (1075, 0), (1105, 0), (1140, 1), (1159, 1), (1243, 1), (1264, 1),
    (1285, 1), (1315, 1), (1349, 1), (1380, 1), (1404, 0), (1457, 0), (1464, 0), (1495, 0),
    (1528, 0), (1555, 0), (1587, 0), (1615, 1), (1645, 1), (1674, 1), (1705, 1), (1735, 1),
    (1764, 1), (1795, 0), (1824, 0), (1855, 0), (1884, 0), (1915, 1), (1945, 1), (1974, 1),
    (2002, 0), (2036, 1), (2062, 1), (2091, 1), (2118, 1), (2158, 1), (2182, 1), (2215, 0),
    (2245, 0), (2274, 0), (2303, 0), (2333, 0), (2365, 0), (2395, 1), (2420, 1), (2454, 1),
    (2485, 1), (2515, 1), (2545, 1), (2575, 1), (2605, 0), (2635, 0), (2665, 0), (2695, 0),
    (2725, 1), (2755, 1), (2785, 1), (2815, 1), (2816, 1),
]
EXCLUDE_DULL_VIEW = True


def acceptor_view_allowed(frame: int) -> bool:
    if not EXCLUDE_DULL_VIEW:
        return True
    samples = ACCEPTOR_REVIEW_SAMPLES
    for f, kept in samples:
        if f == frame:
            return bool(kept)
    for (f0, k0), (f1, k1) in zip(samples, samples[1:]):
        if f0 < frame < f1:
            return bool(k0 and k1)
    return False


def sparse_target_frames(dataset: dict, frame_index: dict, min_frame: int = MIN_FRAME,
                         view_filter: bool = True):
    """The original, unrestricted target pool: any frame with <=1 total instance (any
    species, or empty), regardless of what that one instance (if any) is. Excludes the
    pole/open-water camera view (see acceptor_view_allowed)."""
    return [(n, d) for n, d in dataset.items()
            if d["frame"] is not None and d["frame"] >= min_frame
            and len(d["instances"]) <= 1 and d["frame"] in frame_index
            and (not view_filter or acceptor_view_allowed(d["frame"]))]


def species_target_frames(dataset: dict, frame_index: dict, species: str, min_frame: int = MIN_FRAME,
                          view_filter: bool = True):
    """Frames that already contain >=1 real instance of `species` -- NOT restricted to
    sparse frames (that combination is nearly empty for these species, see module docstring),
    so a target frame here may already hold the species plus other fish. Excludes the
    pole/open-water camera view (see acceptor_view_allowed)."""
    return [(n, d) for n, d in dataset.items()
            if d["frame"] is not None and d["frame"] >= min_frame and d["frame"] in frame_index
            and (not view_filter or acceptor_view_allowed(d["frame"]))
            and any(inst["label"] == species for inst in d["instances"])]


def main():
    dataset = parse_cvat_annotations(ANNOTATIONS_XML)
    frame_index = build_frame_index(IMAGES_DIR)
    print(f"Parsed {len(dataset)} images. Indexed {len(frame_index)} real frames under {IMAGES_DIR}.")
    if not frame_index:
        print("No real frames found -- check IMAGES_DIR. Aborting.")
        return

    if RESTRICT_TARGETS_TO_SAME_SPECIES:
        print("Target mode: same-species only (each species paired against its own frames).")
    else:
        targets_all = sparse_target_frames(dataset, frame_index)
        print(f"Target mode: unrestricted sparse pool -- {len(targets_all)} target frames available locally.")

    summary_rows = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for species in SPECIES_LIST:
        donors = [(n, inst) for n, inst in eligible_donors(dataset, species)
                  if dataset[n]["frame"] in frame_index]
        targets_for_species = (species_target_frames(dataset, frame_index, species)
                                if RESTRICT_TARGETS_TO_SAME_SPECIES else targets_all)
        if RESTRICT_TARGETS_TO_SAME_SPECIES:
            print(f"{species}: {len(targets_for_species)} target frames already containing this species.")
        if not donors or not targets_for_species:
            print(f"{species}: not enough local donors/targets (donors={len(donors)}, "
                  f"targets={len(targets_for_species)}) -- skipping")
            continue

        rng = random.Random(f"{RNG_SEED}:{species}")
        placements = []
        attempts = 0
        max_attempts = N_PER_SPECIES * 25
        while len(placements) < N_PER_SPECIES and attempts < max_attempts:
            attempts += 1
            donor_name, donor_inst = rng.choice(donors)
            target_name, target_d = rng.choice(targets_for_species)
            target_img = get_frame(target_d["frame"], frame_index)
            if target_img is None:
                continue
            left, top, mw, mh = donor_inst["bbox"]
            if mh >= target_img.shape[0] or mw >= target_img.shape[1]:
                continue
            existing = [inst["bbox"] for inst in target_d["instances"]]
            x, y = find_placement(target_img.shape, (mh, mw), existing, rng=rng)
            placements.append((donor_name, donor_inst, target_name, target_d, x, y))

        print(f"{species}: {len(placements)}/{N_PER_SPECIES} placements ready "
              f"(from {len(donors)} eligible donors, {len(targets_for_species)} target frames)")

        for method_folder, method in METHODS:
            out_dir = OUT_DIR / method_folder / species.replace(" ", "_")
            out_dir.mkdir(parents=True, exist_ok=True)
            written = 0
            for i, (donor_name, donor_inst, target_name, target_d, x, y) in enumerate(placements):
                donor_patch, donor_mask = get_donor_patch(dataset, frame_index, donor_name, donor_inst)
                target_img = get_frame(target_d["frame"], frame_index)
                if donor_patch is None or target_img is None:
                    continue

                if method == "naive":
                    result = paste_naive(target_img, donor_patch, donor_mask, x, y)
                elif method == "poisson":
                    result = paste_poisson(target_img, donor_patch, donor_mask, x, y)
                else:
                    result = paste_matched_feathered(target_img, donor_patch, donor_mask, x, y)

                mh, mw = donor_mask.shape
                pad = max(30, int(0.6 * max(mh, mw)))
                overlay = mask_contour_overlay(result, donor_mask, x, y)
                zoom_outline_crop = crop_with_padding(overlay, x, y, mh, mw, pad)

                composite = compose_side_by_side(result, zoom_outline_crop)

                donor_frame = dataset[donor_name]["frame"]
                target_frame = target_d["frame"]
                base = f"{i:03d}_donor{donor_frame}_target{target_frame}"
                cv2.imwrite(str(out_dir / f"{base}.png"), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
                written += 1

                summary_rows.append({
                    "species": species, "method": method_folder, "index": i,
                    "donor_frame": donor_frame, "target_frame": target_frame,
                })
            print(f"  {method_folder}: wrote {written} composite images -> {out_dir}")

    with open(OUT_DIR / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["species", "method", "index", "donor_frame", "target_frame"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    n_pastes = len(summary_rows)
    print(f"\nDone. {n_pastes} composite images (one per paste example) under {OUT_DIR}. "
          f"Index: {OUT_DIR / 'index.csv'}")


if __name__ == "__main__":
    main()
