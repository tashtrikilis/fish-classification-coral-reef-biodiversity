import pickle, numpy as np, pandas as pd, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__))); import common as C
M = pickle.load(open('micro.pkl', 'rb')); B = pickle.load(open('boot.pkl', 'rb')); REL = B['reliable']
def seedmean(D, s, a, iou, f):
    return np.nanmean([f(D[(s, a, sd, iou)]) for sd in C.SEEDS], 0)
metrics = {
  'Micro AP50': (M, lambda d: d['AP50']), 'Micro precision': (M, lambda d: d['P']), 'Micro recall': (M, lambda d: d['R']),
  'Macro AP50 (20 classes)': (B, lambda d: np.nanmean(np.stack([d[('AP50', c)] for c in REL]), 0)),
  'Focus mean AP50': (B, lambda d: np.mean(np.stack([d[('AP50', c)] for c in C.FOCUS_IDS]), 0)),
  'Any-fish AP50': (B, lambda d: d[('AP50', -1)]),
}
rows = []
for s in C.SETUPS:
    for iou in (['box'] if s.startswith('detect') else ['box', 'mask']):
        if (s, 'copypaste', 1, iou) not in M: continue
        for name, (D, f) in metrics.items():
            b = seedmean(D, s, 'baseline', iou, f); c = seedmean(D, s, 'copypaste', iou, f); d = (c - b)[1:]; d = d[~np.isnan(d)]
            p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
            rows.append(dict(setup=s, iou=iou, metric=name, baseline=b[0], copypaste=c[0], delta=c[0] - b[0],
                             ci_lo=np.percentile(d, 2.5), ci_hi=np.percentile(d, 97.5), p=p))
T = pd.DataFrame(rows); T.to_csv('micro_results.csv', index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200)
print(T.round(3).to_string(index=False))
