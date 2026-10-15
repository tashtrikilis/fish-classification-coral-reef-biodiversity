import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import common as C
F = pd.read_csv('fish_level.csv')
rng = np.random.default_rng(0)
W = np.vstack([np.ones(C.NB)] + [np.bincount(rng.integers(0, C.NB, C.NB), minlength=C.NB).astype(float) for _ in range(2000)])
BINS = [0, 16, 24, 40, 1e9]; LAB = ['<16', '16–24', '24–40', '≥40']
F['sbin'] = pd.cut(F['size'], BINS, right=False, labels=LAB)
def rate(df, mask):
    n = W @ np.bincount(df.block, minlength=C.NB); k = W @ np.bincount(df.loc[mask, 'block'], minlength=C.NB)
    return k / n
rows = []
for c in C.FOCUS_IDS:
    for b in LAB:
        sub = {a: F[(F.cls == c) & (F.sbin == b) & (F.arm == a)] for a in C.ARMS}
        nf = len(sub['baseline']) / 12
        r = {a: rate(sub[a], sub[a].cat == 'TP') for a in C.ARMS}
        d = r['copypaste'] - r['baseline']; db = d[1:][~np.isnan(d[1:])]
        mis = {a: (sub[a].cat == 'missed').mean() for a in C.ARMS}
        wr = {a: (sub[a].cat == 'wrong_species').mean() for a in C.ARMS}
        rows.append(dict(species=C.NAMES[c], size_bin=b, n_fish=nf, recall_base=r['baseline'][0], recall_cp=r['copypaste'][0],
                         delta=d[0], lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5),
                         missed_base=mis['baseline'], missed_cp=mis['copypaste'], wrong_base=wr['baseline'], wrong_cp=wr['copypaste']))
S = pd.DataFrame(rows); S.to_csv(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'tables', 'table10_recall_by_size.csv'), index=False); print(S.round(3).to_string())
# encounters (no bootstrap: each encounter is a unit; report recall per encounter, mean over the 12 models)
rows = []
for c in C.FOCUS_IDS:
    for e in sorted(F[F.cls == c].enc.unique()):
        sub = F[(F.cls == c) & (F.enc == e)]
        fr = sub.frame.unique()
        rows.append(dict(species=C.NAMES[c], encounter=int(e), first_frame=int(fr.min()), last_frame=int(fr.max()),
                         n_fish=len(sub) / 24, folds=','.join(map(str, sorted(sub.fold.unique()))),
                         median_size=sub['size'].median(),
                         recall_base=(sub[sub.arm == 'baseline'].cat == 'TP').mean(),
                         recall_cp=(sub[sub.arm == 'copypaste'].cat == 'TP').mean(),
                         missed_base=(sub[sub.arm == 'baseline'].cat == 'missed').mean(),
                         wrong_base=(sub[sub.arm == 'baseline'].cat == 'wrong_species').mean(),
                         wrong_cp=(sub[sub.arm == 'copypaste'].cat == 'wrong_species').mean()))
E = pd.DataFrame(rows); E['delta'] = E.recall_cp - E.recall_base
E.to_csv(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'tables', 'table11_recall_by_encounter.csv'), index=False); print(E.round(2).to_string())
for c in C.FOCUS_IDS:
    e = E[E.species == C.NAMES[c]]
    print(C.NAMES[c], 'encounters improved', (e.delta > 0.02).sum(), 'worse', (e.delta < -0.02).sum(), 'of', len(e))
