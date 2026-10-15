import sys, csv, random, numpy as np, cv2
from pathlib import Path
E = Path.home()/'mnt/EGH490-2_scripts/copy_paste_experiment'
sys.path.insert(0, str(E))
from prepare_experiment import read_export_labels, read_names, poly_bbox_px, paste_poisson_edge, random_transform, find_free_spot, mask_to_polygon_norm
IMG = E.parent/'full_dataset'
names = read_names(E/'labels_export/data.yaml'); cid = {v:k for k,v in names.items()}['Forcepsfish']
labels = read_export_labels(E/'labels_export/labels/train')
W, H = 1280, 720
DON, RECV, SIZE, SEED = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
src = cv2.imread(str(IMG/f'frame_{DON}.jpg'))
# donor polygon = largest Forcepsfish in donor frame
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
# keep paste away from frame edges so the inset is readable
while True:
    spot = find_free_spot(rng, recv.shape, m2.shape, boxes, margin=120)
    if spot: break
x, y = spot
out, fb = paste_poisson_edge(recv, p2, m2, x, y, ring_px=float(np.clip(0.1*min(m2.shape), 1.0, 3.0)))
npoly = mask_to_polygon_norm(m2, x, y, W, H)
print('donor', DON, (x0, y0, bw, bh), 'angle %.1f flip %s scale %.2f -> %dx%d at (%d,%d) fallback %s' % (ang, fl, scale, m2.shape[1], m2.shape[0], x, y, fb))
Y = (0, 215, 255); G = (80, 200, 80)
def inset(img, cx, cy, half, z, corner='tr'):
    a = img.copy()
    xa, ya = max(0, cx-half), max(0, cy-half); xb, yb = min(W, cx+half), min(H, cy+half)
    crop = cv2.resize(img[ya:yb, xa:xb], ((xb-xa)*z, (yb-ya)*z), interpolation=cv2.INTER_CUBIC)
    ch, cw = crop.shape[:2]
    ox, oy = (W-cw-12, 12) if corner == 'tr' else (12, 12)
    if corner == 'br': ox, oy = W-cw-12, H-ch-12
    if corner == 'bl': ox, oy = 12, H-ch-12
    cv2.rectangle(a, (xa, ya), (xb, yb), (255, 255, 255), 2)
    a[oy:oy+ch, ox:ox+cw] = crop; cv2.rectangle(a, (ox, oy), (ox+cw, oy+ch), (255, 255, 255), 3)
    return a
def pick_corner(cx, cy):
    return ('b' if cy < H/2 else 't') + ('l' if cx > W/2 else 'r')
pts = np.round(poly*[W, H]).astype(np.int32)
a = src.copy(); cv2.polylines(a, [pts], True, Y, 2)
dcx, dcy = x0+bw//2, y0+bh//2
a = inset(a, dcx, dcy, 110, 2, pick_corner(dcx, dcy).replace('tl','tl').replace('tr','tr')[0:2] if False else {'bl':'bl','br':'br','tl':'tl','tr':'tr'}[pick_corner(dcx, dcy)] if pick_corner(dcx, dcy) != 'tl' else 'tl')
cv2.imwrite(str(Path.home()/'cpfig/a_donor.png'), a)
# mask panel: donor crop with mask, side by side: cut-out on black
pad = 12; mk = np.zeros((bh+2*pad, bw+2*pad), np.uint8); mk[pad:pad+mask.shape[0], pad:pad+mask.shape[1]] = mask*255
cut = np.zeros((bh+2*pad, bw+2*pad, 3), np.uint8); cut[pad:pad+mask.shape[0], pad:pad+mask.shape[1]] = patch*mask[..., None]
b = np.hstack([cv2.cvtColor(mk, cv2.COLOR_GRAY2BGR), np.full((mk.shape[0], 8, 3), 255, np.uint8), cut])
b = cv2.resize(b, (b.shape[1]*3, b.shape[0]*3), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(Path.home()/'cpfig/b_mask.png'), b)
pcx, pcy = x+m2.shape[1]//2, y+m2.shape[0]//2; corner = pick_corner(pcx, pcy)
c = inset(recv, pcx, pcy, 70, 3, corner); cv2.imwrite(str(Path.home()/'cpfig/c_receiver.png'), c)
d0 = out.copy(); dd = inset(d0, pcx, pcy, 70, 3, corner)
# draw new label outline only inside the inset (so the pasted fish itself is visible in the full frame)
q = np.round(npoly*[W, H]).astype(np.int32)
cv2.imwrite(str(Path.home()/'cpfig/d_augmented.png'), dd)
d2 = out.copy(); cv2.polylines(d2, [q], True, G, 1); d2 = inset(d2, pcx, pcy, 70, 3, corner)
cv2.imwrite(str(Path.home()/'cpfig/d_augmented_label.png'), d2)
