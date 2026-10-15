import sys, pickle, json, numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import common as C
D = pickle.load(open('boot.pkl', 'rb')); REL = D['reliable']; NONF = [c for c in REL if c not in C.FOCUS_IDS]
def iou(st): return 'box' if st.startswith('detect') else 'mask'
def arr(st, arm, cls, seed=None):
    ss = C.SEEDS if seed is None else [seed]
    return np.nanmean([D[(st, arm, sd, iou(st))][('AP50', cls)] for sd in ss], 0)
def grp(st, arm, ids, seed=None): return np.nanmean([arr(st, arm, c, seed) for c in ids], 0)
def ci(x): b = x[1:][~np.isnan(x[1:])]; return [float(x[0]), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
out = {'setups': {}, 'species_avg': {}, 'convict_seed_wins': None}
for st in C.SETUPS:
    r = {}
    for arm in C.ARMS:
        r[arm] = dict(focus=float(grp(st, arm, C.FOCUS_IDS)[0]), macro=float(grp(st, arm, REL)[0]),
                      nonfocus=float(grp(st, arm, NONF)[0]), anyfish=float(arr(st, arm, -1)[0]),
                      **{C.NAMES[c]: float(arr(st, arm, c)[0]) for c in C.FOCUS_IDS})
    r['d_focus'] = ci(grp(st, 'copypaste', C.FOCUS_IDS) - grp(st, 'baseline', C.FOCUS_IDS))
    r['d_nonfocus'] = ci(grp(st, 'copypaste', NONF) - grp(st, 'baseline', NONF))
    r['d_anyfish'] = ci(arr(st, 'copypaste', -1) - arr(st, 'baseline', -1))
    out['setups'][st] = r
wins = 0
for c in C.FOCUS_IDS:
    n = C.NAMES[c]
    out['species_avg'][n] = {arm: float(np.mean([out['setups'][st][arm][n] for st in C.SETUPS])) for arm in C.ARMS}
    w = sum(arr(st, 'copypaste', c, sd)[0] > arr(st, 'baseline', c, sd)[0] for st in C.SETUPS for sd in C.SEEDS)
    out['species_avg'][n]['wins_of_12'] = int(w)
json.dump(out, open('pres_data.json', 'w'), indent=1)
for st, r in out['setups'].items():
    print(st, {k: round(v, 3) for k, v in r['baseline'].items()}, '->', round(r['copypaste']['focus'], 3), [round(v, 3) for v in r['d_focus']], 'nf', [round(v, 3) for v in r['d_nonfocus']], 'af', [round(v, 3) for v in r['d_anyfish']])
print(json.dumps(out['species_avg'], indent=0))
