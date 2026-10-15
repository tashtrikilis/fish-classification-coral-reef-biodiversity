#!/usr/bin/env python3
"""
Automatically list copy-paste donor candidates (Orangespine Unicornfish and Convict Tang), and draw a
contact sheet of them for a visual quality check.

A candidate is a labelled instance of the species, in ANY labelled frame of the CVAT export, that is:
  - big enough:   shorter bbox side >= --min-side px (default 30)
  - whole:        bbox at least --edge-margin px from every frame edge
  - fish-shaped:  polygon area / bbox area >= --min-fill (blocky masks on small/far fish fail this)
  - unobstructed: its bbox doesn't overlap any other labelled fish's bbox (proxy for occlusion)

Writes
  cv_donor_candidates.csv            one row per candidate (species, idx, frame, bbox, size, fill, view)
  qa_preview/cv_donors_<species>.jpg  numbered crops with the label outline, for the visual check

Candidates that fail the visual check are listed in cv_donor_veto.txt as "<species>,<idx>,<idx>..." lines
(or "<species>,all-except,<idx>,<idx>..."). cv_prepare.py combines: Forcepsfish = experiment_donors.csv
(from an earlier visual review), other species = candidates minus those exclusions.
The check was done once, before any cross-validation run, and applies to every fold and model.

Usage (from copy_paste_experiment/):
    python3 cv_select_donors.py --images-dir ../full_dataset
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from prepare_experiment import read_names, read_export_labels, poly_bbox_px

HERE = Path(__file__).resolve().parent
W, H = 1280, 720


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--labels-dir", type=Path, default=HERE / "labels_export" / "labels" / "train")
    p.add_argument("--names-yaml", type=Path, default=HERE / "labels_export" / "data.yaml")
    p.add_argument("--acceptor-csv", type=Path, default=HERE / "approved_acceptor_frames.csv")
    p.add_argument("--species", default="Orangespine Unicornfish,Convict Tang")
    p.add_argument("--min-side", type=float, default=30)
    p.add_argument("--edge-margin", type=float, default=3)
    p.add_argument("--min-fill", type=float, default=0.45)
    p.add_argument("--out-csv", type=Path, default=HERE / "cv_donor_candidates.csv")
    p.add_argument("--sheet-dir", type=Path, default=HERE / "qa_preview")
    return p.parse_args()


def overlaps(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def main():
    args = parse_args()
    names = read_names(args.names_yaml)
    ids = {v: k for k, v in names.items()}
    species = [s.strip() for s in args.species.split(",") if s.strip()]
    for s in species:
        if s not in ids:
            raise SystemExit(f"'{s}' not in class list")
    labels = read_export_labels(args.labels_dir)
    allowed = {}
    if args.acceptor_csv.exists():
        for r in csv.DictReader(open(args.acceptor_csv)):
            allowed[int(r["frame"])] = r["allowed_as_acceptor"] == "1"

    rows = []
    for s in species:
        cid = ids[s]
        cands = []
        for fr in sorted(labels):
            insts = labels[fr]
            boxes = [poly_bbox_px(p, W, H) for _, p in insts]
            for i, (c, poly) in enumerate(insts):
                if c != cid:
                    continue
                x, y, w, h = boxes[i]
                side = min(w, h)
                if side < args.min_side:
                    continue
                if x < args.edge_margin or y < args.edge_margin or x + w > W - args.edge_margin or y + h > H - args.edge_margin:
                    continue
                area = cv2.contourArea((poly * [W, H]).astype(np.float32))
                fill = area / max(1.0, w * h)
                if fill < args.min_fill:
                    continue
                if any(overlaps(boxes[i], boxes[j]) for j in range(len(insts)) if j != i):
                    continue
                cands.append((fr, i, x, y, w, h, side, fill))
        for k, (fr, i, x, y, w, h, side, fill) in enumerate(cands):
            view = {True: "allowed", False: "pole/open-water or turning"}.get(allowed.get(fr), "not reviewed")
            rows.append({"species": s, "idx": k, "frame": fr, "instance": i, "bbox_x": round(x), "bbox_y": round(y),
                         "bbox_w": round(w), "bbox_h": round(h), "min_side_px": round(side, 1),
                         "fill": round(fill, 2), "view": view})
        print(f"{s:26s} {len(cands):3d} candidates from {len({c[0] for c in cands})} frames")

    with open(args.out_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    # contact sheets
    args.sheet_dir.mkdir(exist_ok=True)
    T = 200
    for s in species:
        rs = [r for r in rows if r["species"] == s]
        tiles = []
        for r in rs:
            img = cv2.imread(str(args.images_dir / f"frame_{r['frame']}.jpg"))
            if img is None:
                continue
            poly = labels[r["frame"]][r["instance"]][1] * [W, H]
            x, y, w, h = r["bbox_x"], r["bbox_y"], r["bbox_w"], r["bbox_h"]
            pad = int(max(w, h) * 0.35) + 6
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
            crop = img[y0:y1, x0:x1].copy()
            pts = (poly - [x0, y0]).astype(np.int32)
            sc = T / max(crop.shape[:2])
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * sc)), max(1, int(crop.shape[0] * sc))))
            cv2.polylines(crop, [(pts * sc).astype(np.int32)], True, (255, 0, 255), 1, cv2.LINE_AA)
            tile = np.zeros((T + 34, T, 3), np.uint8)
            tile[34:34 + crop.shape[0], :crop.shape[1]] = crop
            col = (255, 255, 255) if r["view"] == "allowed" else (80, 180, 255)
            cv2.putText(tile, f"#{r['idx']}  f{r['frame']}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            cv2.putText(tile, f"{int(r['min_side_px'])}px  fill {r['fill']}", (4, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            tiles.append(tile)
        if not tiles:
            continue
        cols = 8
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
        out = args.sheet_dir / f"cv_donors_{s.replace(' ', '_')}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"wrote {out}  (orange text = frame not in an approved camera view)")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
