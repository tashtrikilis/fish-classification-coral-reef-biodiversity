import json, csv
import numpy as np
from pathlib import Path
B = Path(__file__).resolve().parent / 'analysis_bundle'   # extract analysis_bundle.tgz here
SETUPS = ['detect_s','detect_m','detect_l','segment_s','segment_m','segment_l']
ARMS = ['baseline','copypaste']
SEEDS = [0,1]
G = json.load(open(B/'gt.json'))
NAMES = {int(k): v for k, v in G['names'].items()}
FOCUS = G['focus']
FOCUS_IDS = [next(k for k, v in NAMES.items() if v == s) for s in FOCUS]
GT = {int(f): np.array(v, float).reshape(-1, 5) for f, v in G['gt'].items()}
FOLDS = {}
for r in csv.DictReader(open(B/'folds.csv')):
    FOLDS[int(r['frame'])] = (int(r['fold']), int(r['block']))
BLOCKS = sorted({b for _, b in FOLDS.values()})
BIDX = {b: i for i, b in enumerate(BLOCKS)}
NB = len(BLOCKS)

_cache = {}
def load(setup, arm, seed):
    """pooled test predictions over the 5 folds: dict frame -> (cls, conf, box(n,4), miou(n,n_gt) or None)"""
    key = (setup, arm, seed)
    if key in _cache: return _cache[key]
    out = {}
    for k in range(5):
        z = np.load(B/'preds'/f'{setup}_{arm}_f{k}_s{seed}.npz')
        fr, cls, conf, box = z['frame'], z['cls'].astype(int), z['conf'].astype(float), z['box'].astype(float)
        miou, mlen = z['miou'], z['miou_len']
        has_m = len(mlen) == len(fr) and len(fr) > 0
        offs = np.concatenate([[0], np.cumsum(mlen)]) if has_m else None
        for f in np.unique(fr):
            idx = np.where(fr == f)[0]
            assert FOLDS[int(f)][0] == k, 'prediction outside its test fold'
            M = None
            if has_m:
                ng = len(GT[int(f)])
                M = np.array([miou[offs[i]:offs[i+1]] for i in idx], float).reshape(len(idx), ng)
            out[int(f)] = (cls[idx], conf[idx], box[idx], M)
    for f in FOLDS:
        if f not in out:
            out[f] = (np.zeros(0, int), np.zeros(0), np.zeros((0, 4)), np.zeros((0, len(GT[f]))) )
    _cache[key] = out
    return out
