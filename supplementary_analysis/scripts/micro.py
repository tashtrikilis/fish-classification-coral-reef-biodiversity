"""Post-hoc MICRO-averaged metrics: every fish counts equally (species-aware matching, all classes pooled).
Same block-bootstrap resamples as cv_evaluate.py (rng default_rng(0), 2000 resamples); seeds averaged inside each resample."""
import sys, pickle, time, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import engine as G, common as C
TAU = 0.25
class Merged:  # pooled across all species classes (excludes the class-agnostic -1 entry)
    def __init__(self, cd):
        ks = [c for c in cd if c != -1]
        conf = np.concatenate([cd[c].conf for c in ks]); blk = np.concatenate([cd[c].block for c in ks])
        tp = np.concatenate([cd[c].tp for c in ks], 1); o = np.argsort(-conf, kind='stable')
        self.conf, self.block, self.tp = conf[o], blk[o], tp[:, o]
        self.gt = np.sum([cd[c].gt for c in ks], 0)
def pr_rows(m, W):
    sel = m.conf >= TAU; b = m.block[sel]; t = m.tp[0][sel].astype(float)
    tpb = np.bincount(b, weights=t, minlength=C.NB); fpb = np.bincount(b, weights=1 - t, minlength=C.NB)
    tp, fp, ngt = W @ tpb, W @ fpb, W @ m.gt
    P = tp / np.maximum(tp + fp, 1e-9); R = tp / ngt
    return P, R, 2 * P * R / np.maximum(P + R, 1e-9)
res = {}
jobs = [(s, a, sd, 'box') for s in C.SETUPS for a in C.ARMS for sd in C.SEEDS] + \
       [(s, a, sd, 'mask') for s in C.SETUPS if s.startswith('segment') for a in C.ARMS for sd in C.SEEDS]
for (s, a, sd, iou) in jobs:
    t0 = time.time()
    m = Merged(G.classdata(s, a, sd, iou))
    P, R, F = pr_rows(m, G.W)
    res[(s, a, sd, iou)] = {'AP50': G.ap_rows(m, 0, G.W), 'P': P, 'R': R, 'F1': F, 'n_gt': int(m.gt.sum())}
    print(s, a, sd, iou, f"AP50={res[(s,a,sd,iou)]['AP50'][0]:.3f} P={P[0]:.3f} R={R[0]:.3f}", f'{time.time()-t0:.0f}s', flush=True)
    pickle.dump(res, open('micro.pkl', 'wb'))
print('done')
