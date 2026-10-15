"""Side-by-side video of out-of-fold TEST detections: baseline vs copy-paste (medium segmentation, seed 0, boxes).
Each frame is predicted by the fold model whose test set contains it, so no model ever saw the frame in training."""
import csv, json, sys, subprocess, numpy as np, cv2
from pathlib import Path
B = Path.home() / 'vid/analysis_bundle'
IMG = Path.home() / 'mnt/EGH490-2_scripts/full_dataset'
OUT = Path.home() / 'mnt/EGH490-2_scripts/report_final/video'
TMP = Path.home() / 'vid/frames'; TMP.mkdir(exist_ok=True)
SETUP, SEED, TAU, FPS = 'segment_m', 0, 0.25, 2
SEGMENTS = [(1601, 1609, 'Forcepsfish encounter'), (2093, 2101, 'Forcepsfish encounter'), (2130, 2136, 'Forcepsfish encounter'),
            (2186, 2194, 'Convict Tang and Orangespine Unicornfish'), (2282, 2288, 'Convict Tang encounter')]
G = json.load(open(B / 'gt.json'))
NAMES = {int(k): v.replace('Bullethead Unicornfish', 'Bullethead Parrotfish').replace('Ornate Butterfly Fish', 'Ornate Butterflyfish').replace('Lognnose', 'Longnose') for k, v in G['names'].items()}
FOCUS = {next(k for k, v in NAMES.items() if v == s): col for s, col in zip(G['focus'], [(0, 200, 255), (255, 0, 220), (255, 230, 0)])}
GT = {int(f): [r for r in v] for f, v in G['gt'].items()}
FOLD = {int(r['frame']): int(r['fold']) for r in csv.DictReader(open(B / 'folds.csv'))}
P = {}
for arm in ('baseline', 'copypaste'):
    P[arm] = {}
    for k in range(5):
        z = np.load(B / f'preds/{SETUP}_{arm}_f{k}_s{SEED}.npz')
        for f in np.unique(z['frame']):
            i = np.where((z['frame'] == f) & (z['conf'] >= TAU))[0]
            P[arm][int(f)] = (z['cls'][i].astype(int), z['conf'][i], z['box'][i])
FONT = cv2.FONT_HERSHEY_SIMPLEX
def label(im, text, x, y, c, s=0.55, t=1):
    (tw, th), _ = cv2.getTextSize(text, FONT, s, t); y = max(th + 6, y); x = min(max(0, x), im.shape[1] - tw - 6)
    cv2.rectangle(im, (x, y - th - 6), (x + tw + 6, y), c, -1)
    cv2.putText(im, text, (x + 3, y - 4), FONT, s, (0, 0, 0) if sum(c) > 400 else (255, 255, 255), t, cv2.LINE_AA)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    return I / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I + 1e-9)
def found(f, arm):
    cls, conf, box = P[arm].get(f, (np.zeros(0, int), np.zeros(0), np.zeros((0, 4))))
    g = [r for r in GT.get(f, []) if int(r[0]) in FOCUS]; used = set(); tp = 0
    for r in g:
        for j in range(len(cls)):
            if j not in used and cls[j] == int(r[0]) and iou(r[1:], box[j]) >= 0.5: used.add(j); tp += 1; break
    return tp, len(g)
def panel(f, arm, title):
    im = cv2.imread(str(IMG / f'frame_{f}.jpg'))
    cls, conf, box = P[arm].get(f, (np.zeros(0, int), np.zeros(0), np.zeros((0, 4))))
    order = sorted(range(len(cls)), key=lambda i: cls[i] in FOCUS)  # focus species drawn last (on top)
    for i in order:
        c = int(cls[i]); x0, y0, x1, y1 = map(int, box[i]); foc = c in FOCUS
        col = FOCUS.get(c, (235, 235, 235))
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 4 if foc else 2)
        if foc or (x1 - x0) >= 30:
            label(im, f'{NAMES[c]} {conf[i]:.2f}', x0, y0 - 3, col, 0.65 if foc else 0.5, 2 if foc else 1)
    im = cv2.resize(im, (960, 540), interpolation=cv2.INTER_AREA)
    bar = np.full((48, 960, 3), 25, np.uint8)
    cv2.putText(bar, title, (14, 33), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    tp, n = found(f, arm)
    cv2.putText(bar, f'focus fish found: {tp}/{n}' if n else 'no focus fish', (260, 32), FONT, 0.7, (0, 230, 120) if n and tp == n else (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(bar, f'frame {f}   test fold {FOLD[f]}', (700, 32), FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, im])
def legend():
    lg = np.full((44, 1928, 3), 25, np.uint8); x = 14
    for c, col in FOCUS.items():
        cv2.rectangle(lg, (x, 12), (x + 28, 32), col, -1); cv2.putText(lg, NAMES[c], (x + 36, 30), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA); x += 300
    cv2.rectangle(lg, (x, 12), (x + 28, 32), (235, 235, 235), 2); cv2.putText(lg, 'other species', (x + 36, 30), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(lg, f'Out-of-fold test predictions, medium segmentation (boxes), seed 0, confidence >= {TAU}', (1230, 30), FONT, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    return lg
LG = legend()
def card(lines, n):
    c = np.full((48 + 540 + 44, 1928, 3), 20, np.uint8)
    for j, (t, s) in enumerate(lines):
        (tw, _), _ = cv2.getTextSize(t, FONT, s, 2); cv2.putText(c, t, ((1928 - tw) // 2, 250 + 60 * j), FONT, s, (255, 255, 255), 2, cv2.LINE_AA)
    return [c] * n
frames = card([('Test detections: baseline vs copy-paste augmentation', 1.3), ('Medium YOLO26 segmentation (boxes shown), seed 0', 0.9),
               ('Every frame is predicted by the cross-validation model that never trained on it', 0.8)], 3 * FPS)
for a, b, what in SEGMENTS:
    frames += card([(what, 1.2), (f'frames {a}-{b} (about {2 * (b - a + 1)} s of footage)', 0.8)], FPS)
    for f in range(a, b + 1):
        row = np.hstack([panel(f, 'baseline', 'Baseline'), np.full((588, 8, 3), 255, np.uint8), panel(f, 'copypaste', 'Copy-paste')])
        frames.append(np.vstack([row, LG]))
for i, fr in enumerate(frames): cv2.imwrite(str(TMP / f'{i:05d}.jpg'), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
out = OUT / 'test_detections_baseline_vs_copypaste.mp4'
subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS), '-i', str(TMP / '%05d.jpg'), '-vf', 'scale=1928:-2',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', str(out)], check=True)
print(out, len(frames), 'frames', len(frames) / FPS, 's')
