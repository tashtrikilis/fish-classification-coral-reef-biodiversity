"""Shared bootstrap engine: AP curves for many block-weight vectors at once (same resamples as cv_evaluate.py)."""
import numpy as np, pickle, time, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import common as C, cv_evaluate as E

R = 2000
rng = np.random.default_rng(0)                      # same generator/seed as cv_evaluate.py
W = np.stack([np.bincount(rng.integers(0, C.NB, C.NB), minlength=C.NB).astype(float) for _ in range(R)])
W = np.vstack([np.ones(C.NB), W])                    # row 0 = point estimate (every block once)

def ap_rows(cd, t, Wm, chunk=250):
    """101-pt AP for class data cd at IoU index t, for every weight row in Wm -> (rows,)"""
    out = np.full(len(Wm), np.nan)
    if cd is None: return out
    tp = cd.tp[t].astype(float); blk = cd.block; rs = np.linspace(0, 1, 101)
    for s in range(0, len(Wm), chunk):
        w = Wm[s:s+chunk]
        ngt = w @ cd.gt
        if len(blk) == 0:
            out[s:s+chunk] = np.where(ngt > 0, 0.0, np.nan); continue
        pw = w[:, blk]
        ctp = np.cumsum(pw * tp, 1); cfp = np.cumsum(pw * (1 - tp), 1)
        prec = ctp / np.maximum(ctp + cfp, 1e-9)
        prec = np.maximum.accumulate(prec[:, ::-1], 1)[:, ::-1]
        for i in range(len(w)):
            if ngt[i] <= 0: continue
            rec = ctp[i] / ngt[i]
            idx = np.searchsorted(rec, rs, side='left')
            ok = idx < prec.shape[1]
            out[s+i] = (prec[i, idx[ok]].sum()) / 101.0
    return out

def classdata(setup, arm, seed, iou):
    P = C.load(setup, arm, seed)
    preds = {f: [[int(c), float(s), *map(float, b)] + ([list(M[i])] if M is not None else [])
                 for i, (c, s, b) in enumerate(zip(*P[f][:3]))] for f, (_, _, _, M) in P.items()}
    gt = {f: [(int(r[0]), *r[1:]) for r in C.GT[f]] for f in C.FOLDS}
    classes = sorted({int(r[0]) for f in gt for r in gt[f]})
    return E.build_classdata(C.FOLDS, gt, preds, classes, C.BIDX, C.NB, iou)
