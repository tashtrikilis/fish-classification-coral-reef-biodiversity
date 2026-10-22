"""Makes the demo videos in this folder: baseline | augmented | ground truth, one per focus species.

All frames are test frames, predicted by the fold model that has that frame in its test set.
Detection M, seed 0, boxes with conf >= 0.25. Focus species in orange, everything else in white.

The clips were picked from encounters where the baseline misses the fish and the augmented model
gets it, so they're examples rather than a random sample (see ../results for the actual numbers).

Usage (needs the dataset zip and analysis_bundle_public.tgz from Releases):
  python make_demo_videos.py --images ../dataset/kona_reef_fish_dataset/images \
      --bundle ../supplementary_analysis/scripts/analysis_bundle --fonts fonts
"""
import argparse, json, subprocess, tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument('--images', type=Path, default=Path('../dataset/kona_reef_fish_dataset/images'))
ap.add_argument('--bundle', type=Path, default=Path('../supplementary_analysis/scripts/analysis_bundle'))
ap.add_argument('--fonts', type=Path, default=Path('fonts'), help='folder with Inter-400.ttf and Inter-700.ttf')
ap.add_argument('--out', type=Path, default=Path(__file__).resolve().parent)
ap.add_argument('--seconds-per-frame', type=float, default=1.5)
a = ap.parse_args()

SETUP, SEED, TAU, MODEL = 'detect_m', 0, 0.25, 'YOLO26 Detection M'
CLIPS = {  # (first, last) frame of each clip; frames without the species are skipped
    'Forcepsfish': [(1603, 1607), (2095, 2100), (2159, 2161)],
    'Orangespine Unicornfish': [(1011, 1011), (1025, 1027), (1259, 1259), (1805, 1805), (2166, 2169), (2191, 2192)],
    'Convict Tang': [(2188, 2190), (2425, 2427), (2456, 2456)],
}
W, H, S = 960, 540, 0.75                     # pane size and scale from 1280 x 720
BAR, GAP = 64, 10
NAVY, ORANGE, WHITE = (11, 37, 69), (242, 109, 61), (240, 240, 240)
F_TITLE = ImageFont.truetype(str(a.fonts / 'Inter-700.ttf'), 28)
F_LAB = ImageFont.truetype(str(a.fonts / 'Inter-700.ttf'), 17)
F_SMALL = ImageFont.truetype(str(a.fonts / 'Inter-700.ttf'), 13)

G = json.load(open(a.bundle / 'gt.json'))
FIX = {'Bullethead Unicornfish': 'Bullethead Parrotfish', 'Lognnose Butterflyfish': 'Longnose Butterflyfish',
       'Ornate Butterfly Fish': 'Ornate Butterflyfish'}
NAMES = {int(k): FIX.get(v, v) for k, v in G['names'].items()}
GT = {int(f): v for f, v in G['gt'].items()}


def load(arm):
    P = {}
    for k in range(5):
        z = np.load(a.bundle / f'preds/{SETUP}_{arm}_f{k}_s{SEED}.npz')
        m = z['conf'] >= TAU
        for f, c, s, b in zip(z['frame'][m], z['cls'][m], z['conf'][m], z['box'][m]):
            P.setdefault(int(f), []).append((int(c), float(s), b))
    return P


PRED = {'baseline': load('baseline'), 'copypaste': load('copypaste')}


def iou(p, q):
    ix = max(0, min(p[2], q[2]) - max(p[0], q[0])); iy = max(0, min(p[3], q[3]) - max(p[1], q[1])); i = ix * iy
    return i / ((p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - i + 1e-9)


def label(d, x, y, text, bg, fg, font=F_LAB, pad=5):
    l, t, r, b = d.textbbox((0, 0), text, font=font); tw, th = r - l, b - t
    x = min(max(0, x), W - tw - 2 * pad); y = y - th - 2 * pad if y - th - 2 * pad >= 0 else y + 2
    d.rectangle([x, y, x + tw + 2 * pad, y + th + 2 * pad], fill=bg)
    d.text((x + pad - l, y + pad - t), text, font=font, fill=fg)


def pane(f, arm, cid, title):
    """arm = 'baseline' / 'copypaste' (predictions with confidence) or 'gt' (manual labels, name only)"""
    im = Image.open(a.images / f'frame_{f}.jpg').convert('RGB').resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    preds = [(int(r[0]), None, r[1:]) for r in GT.get(f, [])] if arm == 'gt' else PRED[arm].get(f, [])
    txt = lambda c, s: NAMES[c] if s is None else f'{NAMES[c]} {s:.2f}'
    others = [(c, s, b) for c, s, b in preds if c != cid]
    mine = [(c, s, b) for c, s, b in preds if c == cid]
    for c, s, b in others:                                   # other species first, so the focus species is on top
        x0, y0, x1, y1 = (float(v) * S for v in b)
        d.rectangle([x0, y0, x1, y1], outline=WHITE, width=2)
        label(d, int(x0), int(y0), txt(c, s), WHITE, NAVY, F_SMALL, 3)
    for c, s, b in mine:
        x0, y0, x1, y1 = (float(v) * S for v in b)
        d.rectangle([x0, y0, x1, y1], outline=ORANGE, width=4)
        label(d, int(x0), int(y0), txt(c, s), ORANGE, WHITE)
    bar = Image.new('RGB', (W, BAR), NAVY); db = ImageDraw.Draw(bar)
    l, t, r, b = db.textbbox((0, 0), title, font=F_TITLE)
    db.text(((W - (r - l)) / 2 - l, (BAR - (b - t)) / 2 - t), title, font=F_TITLE, fill=WHITE)
    out = Image.new('RGB', (W, BAR + H), NAVY); out.paste(bar, (0, 0)); out.paste(im, (0, BAR))
    return out


for sp, clips in CLIPS.items():
    cid = next(k for k, v in NAMES.items() if v == sp)
    with tempfile.TemporaryDirectory() as tmp:
        n = 0
        for f0, f1 in clips:
            for f in range(f0, f1 + 1):
                if not any(r[0] == cid for r in GT.get(f, [])):
                    continue
                canvas = Image.new('RGB', (3 * W + 2 * GAP, BAR + H), NAVY)   # three panes in a row
                canvas.paste(pane(f, 'baseline', cid, f'Baseline ({MODEL})'), (0, 0))
                canvas.paste(pane(f, 'copypaste', cid, f'Augmented ({MODEL})'), (W + GAP, 0))
                canvas.paste(pane(f, 'gt', cid, 'Ground truth (manual labels)'), (2 * (W + GAP), 0))
                canvas.save(f'{tmp}/{n:05d}.png'); n += 1
        out = a.out / f"demo_{sp.lower().replace(' ', '_')}.mp4"
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(1 / a.seconds_per_frame),
                        '-i', f'{tmp}/%05d.png', '-r', '30', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
                        '-movflags', '+faststart', str(out)], check=True)
        print(f'{out.name}: {n} frames, {n * a.seconds_per_frame:.0f} s')
