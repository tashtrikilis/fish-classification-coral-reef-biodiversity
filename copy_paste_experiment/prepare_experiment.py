#!/usr/bin/env python3
"""
Helpers used by cv_prepare.py: reading the CVAT export, box/polygon conversion, random
rotation/flip, finding a free spot and the edge-only Poisson paste.
"""
import os
import re
import shutil
import sys

import cv2
import numpy as np


# ---- labels

def read_names(yaml_path):
    names = {}
    for line in yaml_path.read_text().splitlines():
        m = re.match(r"\s+(\d+):\s*(.+?)\s*$", line)
        if m:
            names[int(m.group(1))] = m.group(2).strip().strip("'\"")
    if not names:
        sys.exit(f"no class names found in {yaml_path}")
    return names


FRAME_RE = re.compile(r"mp4-(\d+)_jpg")


def read_export_labels(labels_dir):
    """frame -> list of (class_id, Nx2 normalised polygon). Export files are named like
    nelha-trimmed_mp4-<frame>_jpg.rf.<hash>.txt"""
    out = {}
    for f in labels_dir.glob("*.txt"):
        m = FRAME_RE.search(f.name)
        if not m:
            continue
        rows = []
        for line in f.read_text().splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append((int(parts[0]), np.array(parts[1:], dtype=np.float64).reshape(-1, 2)))
        out[int(m.group(1))] = rows
    if not out:
        sys.exit(f"no label files matching *mp4-<frame>_jpg* found in {labels_dir}")
    return out


def poly_bbox_px(poly, w, h):
    xy = poly * [w, h]
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    return float(x0), float(y0), float(x1 - x0), float(y1 - y0)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def seg_line(cls, poly_norm):
    return f"{cls} " + " ".join(f"{v:.6f}" for v in np.clip(poly_norm, 0, 1).ravel())


def det_line(cls, poly_norm):
    x0, y0 = np.clip(poly_norm.min(0), 0, 1)
    x1, y1 = np.clip(poly_norm.max(0), 0, 1)
    return f"{cls} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {x1 - x0:.6f} {y1 - y0:.6f}"


# ---- augmentation and blending (images are RGB numpy arrays)

def color_transfer_lab(patch, mask, target_region):
    patch_lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_region, cv2.COLOR_RGB2LAB).astype(np.float32)
    m = mask.astype(bool)
    if m.sum() < 5:
        return patch
    out = patch_lab.copy()
    for c, s in enumerate((0.45, 0.15, 0.15)):
        src_vals = patch_lab[:, :, c][m]
        tgt_vals = target_lab[:, :, c]
        src_mean, src_std = src_vals.mean(), src_vals.std() + 1e-6
        tgt_mean, tgt_std = tgt_vals.mean(), tgt_vals.std() + 1e-6
        std_ratio = 1 + s * (np.clip(tgt_std / src_std, 0.8, 1.25) - 1)
        shift = np.clip(s * (tgt_mean - src_mean), -0.5 * src_std, 0.5 * src_std)
        out[:, :, c][m] = (src_vals - src_mean) * std_ratio + src_mean + shift
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def paste_matched_feathered(target, patch, mask, x, y):
    h, w = mask.shape
    feather_px = max(1, min(5, round(0.06 * min(h, w))))
    matched = color_transfer_lab(patch, mask, target[y:y + h, x:x + w])
    alpha = cv2.GaussianBlur((mask * 255).astype(np.uint8), (0, 0), sigmaX=feather_px).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    out = target.copy()
    region = out[y:y + h, x:x + w].astype(np.float32)
    out[y:y + h, x:x + w] = np.clip(region * (1 - alpha) + matched.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return out


def paste_poisson(target, patch, mask, x, y):
    """Returns (image, used_fallback)."""
    h, w = mask.shape
    mask255 = (mask * 255).astype(np.uint8)
    try:
        return cv2.seamlessClone(patch, target, mask255, (x + w // 2, y + h // 2), cv2.NORMAL_CLONE), False
    except cv2.error:
        return paste_matched_feathered(target, patch, mask, x, y), True


def paste_poisson_edge(target, patch, mask, x, y, ring_px=3.0):
    """Edge-only Poisson: the seamlessClone result in a thin ring around the mask edge, original donor
    pixels inside. Full Poisson recolours the whole fish from the background (yellow Forcepsfish came
    out grey-blue on open-water frames) and loses thin fins. Returns (image, used_fallback)."""
    pois, fb = paste_poisson(target, patch, mask, x, y)
    if fb:
        return pois, fb
    h, w = mask.shape
    m = (mask > 0).astype(np.uint8)
    dist = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    alpha = (np.clip(dist / ring_px, 0, 1) * m)[..., None].astype(np.float32)
    region = pois[y:y + h, x:x + w].astype(np.float32)
    out = pois.copy()
    out[y:y + h, x:x + w] = np.clip(alpha * patch.astype(np.float32) + (1 - alpha) * region, 0, 255).astype(np.uint8)
    return out, False


def random_transform(patch, mask, rng, max_angle, hflip_prob):
    angle = rng.uniform(-max_angle, max_angle)
    h, w = mask.shape
    diag = int(np.ceil(np.hypot(h, w))) + 4
    pad_h, pad_w = (diag - h) // 2 + 2, (diag - w) // 2 + 2
    patch_p = cv2.copyMakeBorder(patch, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    mask_p = cv2.copyMakeBorder(mask, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
    ph, pw = mask_p.shape
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    patch_r = cv2.warpAffine(patch_p, M, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_r = cv2.warpAffine(mask_p, M, (pw, ph), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.where(mask_r > 0)
    if len(ys) == 0:
        return patch, mask, 0.0, False
    patch_r = patch_r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    mask_r = mask_r[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    flipped = rng.random() < hflip_prob
    if flipped:
        patch_r, mask_r = np.fliplr(patch_r), np.fliplr(mask_r)
    return np.ascontiguousarray(patch_r), np.ascontiguousarray(mask_r), angle, flipped


def find_free_spot(rng, img_shape, mask_shape, existing_boxes, margin=3, tries=60):
    th, tw = img_shape[:2]
    mh, mw = mask_shape
    if mh + 2 * margin >= th or mw + 2 * margin >= tw:
        return None
    for _ in range(tries):
        x = rng.randint(margin, tw - mw - margin)
        y = rng.randint(margin, th - mh - margin)
        if all(x + mw < bx or x > bx + bw or y + mh < by or y > by + bh for bx, by, bw, bh in existing_boxes):
            return x, y
    return None


def mask_to_polygon_norm(mask, x, y, w_img, h_img):
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, max(0.5, 0.004 * cv2.arcLength(c, True)), True).reshape(-1, 2).astype(np.float64)
    if len(c) < 3:
        return None
    c[:, 0] = (c[:, 0] + x) / w_img
    c[:, 1] = (c[:, 1] + y) / h_img
    return c


# ---- files

def link_or_copy(src, dst, copy):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(os.path.abspath(src), dst)
