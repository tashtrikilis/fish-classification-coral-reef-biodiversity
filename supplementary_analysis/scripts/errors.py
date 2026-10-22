"""Fish-level error analysis at a fixed operating point (no retraining, saved CV test predictions only)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import common as C
from cv_evaluate import iou_matrix

TAU = 0.25        # confidence threshold (Ultralytics default; same as the reported P/R)
IOU_T = 0.50      # match threshold (box IoU)
IOU_LOC = 0.10    # lower bound of the "poorly localised" band
GAP = 15          # frames: instances of a species <= 15 frames apart belong to one encounter

# encounters per focus species (computed on the ground truth of the whole evaluated footage)
ENC = {}
for c in C.FOCUS_IDS:
    frames = sorted(f for f in C.FOLDS if (C.GT[f][:, 0] == c).any())
    e, last = -1, None
    for f in frames:
        if last is None or f - last > GAP: e += 1
        ENC[(c, f)] = e; last = f

def greedy(pc, ps, pb, gc, gb):
    """class-aware greedy matching, highest confidence first; returns gt->pred and pred->gt maps"""
    g2p, p2g = {}, {}
    if len(pb) == 0 or len(gb) == 0: return g2p, p2g, np.zeros((len(pb), len(gb)))
    M = iou_matrix(pb, gb)
    for i in np.argsort(-ps, kind='stable'):
        cand = [j for j in range(len(gb)) if j not in g2p and gc[j] == pc[i] and M[i, j] >= IOU_T]
        if cand:
            j = max(cand, key=lambda j: M[i, j]); g2p[j] = i; p2g[i] = j
    return g2p, p2g, M

def analyse(setup, arm, seed):
    P = C.load(setup, arm, seed)
    fish, fps = [], []
    for f, (k, b) in C.FOLDS.items():
        g = C.GT[f]; gc = g[:, 0].astype(int); gb = g[:, 1:5]
        cls, conf, box, MI = P[f]
        hi = conf >= TAU
        pc, ps, pb = cls[hi], conf[hi], box[hi]
        g2p, p2g, M = greedy(pc, ps, pb, gc, gb)
        Mall = iou_matrix(box, gb) if len(box) and len(gb) else np.zeros((len(box), len(gb)))
        mi_hi = MI[hi] if MI is not None and len(MI) else None
        for j in range(len(gb)):
            if gc[j] not in C.FOCUS_IDS: continue
            w_, h_ = gb[j, 2] - gb[j, 0], gb[j, 3] - gb[j, 1]
            row = dict(frame=f, block=C.BIDX[b], fold=k, cls=gc[j], size=min(w_, h_), enc=ENC[(gc[j], f)],
                       cat=None, pred_cls=-1, conf=np.nan, box_iou=np.nan, mask_iou=np.nan, sub=None)
            if j in g2p:
                i = g2p[j]
                row.update(cat='TP', pred_cls=int(pc[i]), conf=float(ps[i]), box_iou=float(M[i, j]))
                if mi_hi is not None: row['mask_iou'] = float(mi_hi[i, j])
            else:
                free = [i for i in range(len(pb)) if i not in p2g]
                def best(cond):
                    c_ = [i for i in free if cond(i)]
                    return max(c_, key=lambda i: M[i, j]) if c_ else None
                i = best(lambda i: pc[i] != gc[j] and M[i, j] >= IOU_T)
                if i is not None:
                    row.update(cat='wrong_species', pred_cls=int(pc[i]), conf=float(ps[i]), box_iou=float(M[i, j]))
                else:
                    i = best(lambda i: pc[i] == gc[j] and IOU_LOC <= M[i, j] < IOU_T)
                    if i is not None:
                        row.update(cat='poor_loc_same', pred_cls=int(pc[i]), conf=float(ps[i]), box_iou=float(M[i, j]))
                    else:
                        i = best(lambda i: pc[i] != gc[j] and IOU_LOC <= M[i, j] < IOU_T)
                        if i is not None:
                            row.update(cat='poor_loc_other', pred_cls=int(pc[i]), conf=float(ps[i]), box_iou=float(M[i, j]))
                        else:
                            row['cat'] = 'missed'
                            # sub-threshold evidence only (never counted as a classification)
                            lo = (conf < TAU)
                            same = lo & (cls == gc[j]) & (Mall[:, j] >= IOU_T) if len(cls) else np.zeros(0, bool)
                            other = lo & (cls != gc[j]) & (Mall[:, j] >= IOU_T) if len(cls) else np.zeros(0, bool)
                            if same.any():
                                row.update(sub='below_tau_same', conf=float(conf[same].max()))
                            elif other.any():
                                row.update(sub='below_tau_other', conf=float(conf[other].max()))
                            else:
                                row['sub'] = 'no_prediction'
            fish.append(row)
        # false positives of focus classes
        for i in range(len(pb)):
            if pc[i] not in C.FOCUS_IDS or i in p2g: continue
            ious = M[i] if len(gb) else np.zeros(0)
            r = dict(frame=f, block=C.BIDX[b], fold=k, pred_cls=int(pc[i]), conf=float(ps[i]), true_cls=-1, cat=None)
            same_hit = [j for j in range(len(gb)) if gc[j] == pc[i] and ious[j] >= IOU_T]
            other_hit = [j for j in range(len(gb)) if gc[j] != pc[i] and ious[j] >= IOU_T]
            if same_hit: r['cat'] = 'duplicate'
            elif other_hit:
                j = max(other_hit, key=lambda j: ious[j]); r.update(cat='wrong_species', true_cls=int(gc[j]))
            elif len(gb) and ious.max() >= IOU_LOC:
                j = int(np.argmax(ious)); r.update(cat='poor_loc', true_cls=int(gc[j]))
            else: r['cat'] = 'no_labelled_fish'
            fps.append(r)
    F = pd.DataFrame(fish); Q = pd.DataFrame(fps)
    for D in (F, Q): D['setup'], D['arm'], D['seed'] = setup, arm, seed
    return F, Q

if __name__ == '__main__':
    allF, allQ = [], []
    for s in C.SETUPS:
        for a in C.ARMS:
            for sd in C.SEEDS:
                F, Q = analyse(s, a, sd); allF.append(F); allQ.append(Q)
                print(s, a, sd, len(F), (F.cat == 'TP').sum(), len(Q), flush=True)
    pd.concat(allF).to_csv('fish_level.csv', index=False)
    pd.concat(allQ).to_csv('fp_level.csv', index=False)
