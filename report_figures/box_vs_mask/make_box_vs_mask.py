import sys, numpy as np, cv2
from pathlib import Path
E = Path(__file__).resolve().parents[3] / 'copy_paste_experiment'; sys.path.insert(0, str(E))
from prepare_experiment import read_export_labels, read_names, poly_bbox_px
FR = int(sys.argv[1]) if len(sys.argv) > 1 else 2189
OUT = Path(__file__).resolve().parent
L = read_export_labels(E / 'labels_export/labels/train'); N = read_names(E / 'labels_export/data.yaml')
N = {k: v.replace('Bullethead Unicornfish', 'Bullethead Parrotfish').replace('Ornate Butterfly Fish', 'Ornate Butterflyfish') for k, v in N.items()}
img = cv2.imread(str(E.parent / 'full_dataset' / f'frame_{FR}.jpg')); H, W = img.shape[:2]
rows = L[FR]
PAL = [(200,110,30),(0,128,255),(60,170,60),(40,40,220),(170,60,140),(200,200,0),(0,200,235),(40,90,150),(120,120,120),(255,120,200)]
cls = sorted({c for c, _ in rows}); col = {c: PAL[i % len(PAL)] for i, c in enumerate(cls)}
boxes = [poly_bbox_px(p, W, H) for _, p in rows]
x0 = int(max(0, min(b[0] for b in boxes) - 40)); y0 = int(max(0, min(b[1] for b in boxes) - 40))
x1 = int(min(W, max(b[0] + b[2] for b in boxes) + 40)); y1 = int(min(H, max(b[1] + b[3] for b in boxes) + 40))
X1 = None
def tag(im, text, x, y, c):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y = max(th + 4, y)
    x = min(x, X1 - tw - 6)
    cv2.rectangle(im, (x, y - th - 5), (x + tw + 4, y), c, -1)
    cv2.putText(im, text, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
X1 = x1
a = img.copy()
for (c, p), (bx, by, bw, bh) in zip(rows, boxes):
    cv2.rectangle(a, (int(bx), int(by)), (int(bx + bw), int(by + bh)), col[c], 2)
b = img.copy(); ov = img.copy()
for c, p in rows:
    pts = np.round(p * [W, H]).astype(np.int32); cv2.fillPoly(ov, [pts], col[c])
b = cv2.addWeighted(ov, 0.35, b, 0.65, 0)
for c, p in rows:
    cv2.polylines(b, [np.round(p * [W, H]).astype(np.int32)], True, col[c], 2, cv2.LINE_AA)
for im in (a, b):
    for (c, p), (bx, by, bw, bh) in zip(rows, boxes):
        if min(bw, bh) >= 20: tag(im, N[c], int(bx), int(by) - 2, col[c])
for name, im in (('a_boxes', a), ('b_masks', b)):
    cv2.imwrite(str(OUT / f'{name}.png'), im[y0:y1, x0:x1])
print(FR, len(rows), 'fish; crop', (x0, y0, x1, y1), sorted({N[c] for c, _ in rows}))
