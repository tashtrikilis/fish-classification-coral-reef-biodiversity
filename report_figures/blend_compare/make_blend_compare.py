"""Before/after edge-only Poisson blending for the copy-paste example in report_final/figs/cp_steps
(donor frame 1668, receiver frame 1950, seed 3, size 50 - the same paste as d_augmented.png).
(a) naive paste: mask pixels copied directly; (b) edge-only Poisson blend used in the experiment."""
import sys, random, numpy as np, cv2
from pathlib import Path
E = Path.home()/'mnt/EGH490-2_scripts/copy_paste_experiment'; sys.path.insert(0, str(E))
from prepare_experiment import read_export_labels, read_names, poly_bbox_px, paste_poisson_edge, random_transform, find_free_spot
IMG = E.parent/'full_dataset'; W, H = 1280, 720
DON, RECV, SIZE, SEED = 1668, 1950, int(sys.argv[1]) if len(sys.argv) > 1 else 50, 3
names = read_names(E/'labels_export/data.yaml'); cid = {v:k for k,v in names.items()}['Forcepsfish']
labels = read_export_labels(E/'labels_export/labels/train')
src = cv2.imread(str(IMG/f'frame_{DON}.jpg'))
polys = [p for c, p in labels[DON] if c == cid]
poly = max(polys, key=lambda p: poly_bbox_px(p, W, H)[2]*poly_bbox_px(p, W, H)[3])
x0, y0, bw, bh = (int(round(v)) for v in poly_bbox_px(poly, W, H)); x1, y1 = min(x0+bw+1, W), min(y0+bh+1, H)
full = np.zeros((H, W), np.uint8); cv2.fillPoly(full, [np.round(poly*[W, H]).astype(np.int32)], 1)
patch, mask = src[y0:y1, x0:x1], full[y0:y1, x0:x1]
rng = random.Random(SEED)
p2, m2, ang, fl = random_transform(patch, mask, rng, 20, 0.5)
scale = min(1.0, SIZE/max(1, min(m2.shape)))
nw, nh = max(1, round(m2.shape[1]*scale)), max(1, round(m2.shape[0]*scale))
p2 = cv2.resize(p2, (nw, nh), interpolation=cv2.INTER_AREA)
m2 = (cv2.resize(m2.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
ys, xs = np.where(m2 > 0); p2 = np.ascontiguousarray(p2[ys.min():ys.max()+1, xs.min():xs.max()+1]); m2 = np.ascontiguousarray(m2[ys.min():ys.max()+1, xs.min():xs.max()+1])
recv = cv2.imread(str(IMG/f'frame_{RECV}.jpg'))
boxes = [poly_bbox_px(p, W, H) for _, p in labels.get(RECV, [])]
while True:
    spot = find_free_spot(rng, recv.shape, m2.shape, boxes, margin=120)
    if spot: break
x, y = spot
blend, fb = paste_poisson_edge(recv, p2, m2, x, y, ring_px=float(np.clip(0.1*min(m2.shape), 1.0, 3.0)))
naive = recv.copy(); h, w = m2.shape
naive[y:y+h, x:x+w][m2 > 0] = p2[m2 > 0]
ref = cv2.imread(str(Path.home()/'mnt/EGH490-2_scripts/report_final/figs/cp_steps/d_augmented.png'))
reg = (slice(y-5, y+h+5), slice(x-5, x+w+5))
print('size', SIZE, 'spot', (x, y), 'shape', m2.shape, 'fallback', fb,
      'match to d_augmented (mean abs diff in paste region):', np.abs(blend[reg].astype(int) - ref[reg].astype(int)).mean())
cx, cy = x + w//2, y + h//2; hw, hh = 72, 40; Z = 7
def crop(img):
    c = img[cy-hh:cy+hh, cx-hw:cx+hw]
    return cv2.resize(c, (c.shape[1]*Z, c.shape[0]*Z), interpolation=cv2.INTER_CUBIC)
cv2.imwrite('a_naive_paste.png', crop(naive)); cv2.imwrite('b_poisson_edge.png', crop(blend))
cv2.imwrite('a_naive_paste_full.png', naive); cv2.imwrite('b_poisson_edge_full.png', blend)
