import sys, warnings, numpy as np, pandas as pd, json
warnings.filterwarnings("ignore")
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import common as C
rng = np.random.default_rng(0)
W = np.vstack([np.ones(C.NB)] + [np.bincount(rng.integers(0, C.NB, C.NB), minlength=C.NB).astype(float) for _ in range(2000)])
F = pd.read_csv('fish_level.csv'); Q = pd.read_csv('fp_level.csv')
SP = {c: C.NAMES[c] for c in C.FOCUS_IDS}
NMOD = 12  # 6 set-ups x 2 seeds

def per_block(df, mask):
    return np.bincount(df.loc[mask, 'block'], minlength=C.NB).astype(float)

def ci(x):
    b = x[1:][~np.isnan(x[1:])]   # resamples that contain no fish of the species are dropped (as in cv_evaluate.py)
    return x[0], np.percentile(b, 2.5), np.percentile(b, 97.5), min(1, 2 * min((b <= 0).mean(), (b >= 0).mean()))

def stats(Fs, Qs):
    """block-weighted quantities for one arm (pooled over the models in Fs/Qs) -> dict name -> (R+1,)"""
    n = W @ per_block(Fs, Fs.index == Fs.index)
    out = {}
    for cat in ['TP', 'wrong_species', 'poor_loc_same', 'poor_loc_other', 'missed']:
        out[cat] = (W @ per_block(Fs, Fs.cat == cat)) / n
    out['poor_loc'] = out['poor_loc_same'] + out['poor_loc_other']
    for sub in ['below_tau_same', 'below_tau_other', 'no_prediction']:
        out['missed:' + sub] = (W @ per_block(Fs, Fs['sub'] == sub)) / n
    tp = W @ per_block(Fs, Fs.cat == 'TP'); fp = W @ per_block(Qs, Qs.index == Qs.index)
    out['recall'] = tp / n
    out['precision'] = tp / np.maximum(tp + fp, 1e-9)
    loc = W @ per_block(Fs, Fs.cat.isin(['TP', 'wrong_species']))
    out['localised'] = loc / n                                   # any-class box IoU >= 0.5 at conf >= 0.25
    out['id_given_localised'] = tp / np.maximum(loc, 1e-9)       # correct species among localised
    for cat in ['duplicate', 'wrong_species', 'poor_loc', 'no_labelled_fish']:
        out['FP_' + cat] = (W @ per_block(Qs, Qs.cat == cat)) / NMOD_cur[0]
    out['FP_total'] = fp / NMOD_cur[0]
    out['n_fish'] = n
    return out

rows, deltas = [], []
NMOD_cur = [NMOD]
for c in list(C.FOCUS_IDS) + ['all3']:
    for scope in ['all'] + C.SETUPS:
        sel = (F.setup == scope) if scope != 'all' else (F.setup == F.setup)
        selq = (Q.setup == scope) if scope != 'all' else (Q.setup == Q.setup)
        if c != 'all3':
            sel &= F.cls == c; selq &= Q.pred_cls == c
        NMOD_cur[0] = NMOD if scope == 'all' else 2
        st = {a: stats(F[sel & (F.arm == a)], Q[selq & (Q.arm == a)]) for a in C.ARMS}
        for k in st['baseline']:
            if k == 'n_fish': continue
            b, cp = st['baseline'][k], st['copypaste'][k]
            d0, lo, hi, p = ci(cp - b)
            rows.append(dict(species=SP.get(c, 'Focus species (pooled)'), scope=scope, quantity=k,
                             baseline=b[0], copypaste=cp[0], delta=d0, lo=lo, hi=hi, p=p))
T = pd.DataFrame(rows)
T.to_csv('err_summary_all.csv', index=False)
print(T[(T.scope == 'all')].round(3).to_string())
