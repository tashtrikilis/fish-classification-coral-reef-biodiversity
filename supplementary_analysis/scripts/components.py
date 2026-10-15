import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import common as C
F = pd.read_csv('fish_level.csv')
rng = np.random.default_rng(0)
W = np.vstack([np.ones(C.NB)] + [np.bincount(rng.integers(0, C.NB, C.NB), minlength=C.NB).astype(float) for _ in range(2000)])
def bc(df, m=None, val=None):
    d = df if m is None else df[m]
    return np.bincount(d.block, weights=None if val is None else d[val], minlength=C.NB).astype(float)
def comps(df):
    n = W @ bc(df); tp = W @ bc(df, df.cat == 'TP'); loc = W @ bc(df, df.cat.isin(['TP', 'wrong_species']))
    out = {'recall': tp / n, 'localisation': loc / n, 'identification': tp / np.maximum(loc, 1e-9)}
    t = df[(df.cat == 'TP') & df.mask_iou.notna()]
    if len(t):
        out['mask_IoU_of_TP'] = (W @ bc(t, None, 'mask_iou')) / np.maximum(W @ bc(t), 1e-9)
        out['mask_IoU>=0.75_given_TP'] = (W @ bc(t, t.mask_iou >= 0.75)) / np.maximum(W @ bc(t), 1e-9)
        out['box_IoU_of_TP'] = (W @ bc(t, None, 'box_iou')) / np.maximum(W @ bc(t), 1e-9)
    return out
def summ(x):
    b = x[1:][~np.isnan(x[1:])]
    return x[0], np.percentile(b, 2.5), np.percentile(b, 97.5), min(1, 2 * min((b <= 0).mean(), (b >= 0).mean()))
rows, eff = [], {}
for scope_name, cls in [('Focus species (pooled)', None)] + [(C.NAMES[c], c) for c in C.FOCUS_IDS]:
    for s in C.SETUPS:
        sub = F[F.setup == s] if cls is None else F[(F.setup == s) & (F.cls == cls)]
        cb = comps(sub[sub.arm == 'baseline']); cc = comps(sub[sub.arm == 'copypaste'])
        for k in cb:
            d = cc[k] - cb[k]; eff[(scope_name, s, k)] = d
            e, lo, hi, p = summ(d)
            rows.append(dict(target=scope_name, setup=s, component=k, baseline=cb[k][0], copypaste=cc[k][0], effect=e, lo=lo, hi=hi, p=p))
T = pd.DataFrame(rows); T.to_csv('components_by_setup.csv', index=False)
did = []
for scope_name in ['Focus species (pooled)'] + [C.NAMES[c] for c in C.FOCUS_IDS]:
    for size in 'sml':
        for k in ['recall', 'localisation', 'identification']:
            d = eff[(scope_name, f'segment_{size}', k)] - eff[(scope_name, f'detect_{size}', k)]
            e, lo, hi, p = summ(d); did.append(dict(target=scope_name, size=size, component=k, seg_minus_det=e, lo=lo, hi=hi, p=p))
pd.DataFrame(did).to_csv('components_DiD.csv', index=False)
print(T[T.target == 'Focus species (pooled)'].round(3).to_string())
print(pd.DataFrame(did)[lambda d: d.target == 'Focus species (pooled)'].round(3).to_string())
