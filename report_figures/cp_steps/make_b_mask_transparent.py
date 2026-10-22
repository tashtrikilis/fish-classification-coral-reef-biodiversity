"""Transparent versions of panel b (donor Forcepsfish, frame 1668): the cut-out fish and the white mask,
no black background. Same crop and 3x nearest-neighbour scaling as b_mask.png."""
import sys, numpy as np, cv2
from pathlib import Path
E = Path.home()/'mnt/EGH490-2_scripts/copy_paste_experiment'; sys.path.insert(0, str(E))
from prepare_experiment import read_export_labels, read_names, poly_bbox_px
W, H = 1280, 720; DON = 1668
names = read_names(E/'labels_export/data.yaml'); cid = {v: k for k, v in names.items()}['Forcepsfish']
labels = read_export_labels(E/'labels_export/labels/train')
src = cv2.imread(str(E.parent/'full_dataset'/f'frame_{DON}.jpg'))
polys = [p for c, p in labels[DON] if c == cid]
poly = max(polys, key=lambda p: poly_bbox_px(p, W, H)[2]*poly_bbox_px(p, W, H)[3])
x0, y0, bw, bh = (int(round(v)) for v in poly_bbox_px(poly, W, H)); x1, y1 = min(x0+bw+1, W), min(y0+bh+1, H)
full = np.zeros((H, W), np.uint8); cv2.fillPoly(full, [np.round(poly*[W, H]).astype(np.int32)], 255)
patch, mask = src[y0:y1, x0:x1], full[y0:y1, x0:x1]
pad = 12
fish = np.zeros((bh+2*pad, bw+2*pad, 4), np.uint8); fish[pad:pad+mask.shape[0], pad:pad+mask.shape[1], :3] = patch
fish[pad:pad+mask.shape[0], pad:pad+mask.shape[1], 3] = mask
mk = np.zeros_like(fish); mk[..., :3] = 255; mk[..., 3] = fish[..., 3]
up = lambda a: cv2.resize(a, (a.shape[1]*3, a.shape[0]*3), interpolation=cv2.INTER_NEAREST)
cv2.imwrite('b_cutout_transparent.png', up(fish)); cv2.imwrite('b_mask_transparent.png', up(mk))
# smooth-edged version for slides
sm = cv2.resize(fish, (fish.shape[1]*3, fish.shape[0]*3), interpolation=cv2.INTER_LANCZOS4)
cv2.imwrite('b_cutout_transparent_smooth.png', sm)
