"""Augmentation-vs-capacity comparisons, with segmentation scored by MASK IoU (matches main results)."""
import sys, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import common as C
D = pickle.load(open('boot.pkl', 'rb')); REL = D['reliable']; NONF = [c for c in REL if c not in C.FOCUS_IDS]
def sm(task, size): return f"{'detect' if task == 'det' else 'segment'}_{size}"
def arr(st, arm, iou, cls): return np.nanmean([D[(st, arm, sd, iou)][('AP50', cls)] for sd in C.SEEDS], 0)
def M(task, size, arm, what):
    st = sm(task, size); iou = 'box' if task == 'det' else 'mask'
    ids = {'focus': C.FOCUS_IDS, 'nonfocus': NONF, 'macro': REL, 'anyfish': [-1]}[what]
    return np.nanmean([arr(st, arm, iou, c) for c in ids], 0)
def summ(x):
    b = x[1:][~np.isnan(x[1:])]
    return x[0], np.percentile(b, 2.5), np.percentile(b, 97.5), min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
rows = []
for task in ['det', 'seg']:
    for small, big in [('s', 'm'), ('m', 'l')]:
        for what in ['focus', 'nonfocus', 'anyfish']:
            a, b0, B = M(task, small, 'copypaste', what), M(task, small, 'baseline', what), M(task, big, 'baseline', what)
            for lab, x in [(f'CP {small} - base {big}', a - B), (f'aug gain {small}', a - b0), (f'capacity {small}->{big}', B - b0)]:
                e, lo, hi, p = summ(x); rows.append(dict(task=task, comp=lab, metric=what, est=e, lo=lo, hi=hi, p=p))
    for what in ['focus', 'nonfocus', 'anyfish']:
        e, lo, hi, p = summ(M(task, 'l', 'copypaste', what) - M(task, 'l', 'baseline', what))
        rows.append(dict(task=task, comp='aug gain l', metric=what, est=e, lo=lo, hi=hi, p=p))
R = pd.DataFrame(rows); R.to_csv(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'tables', 'table8_augmentation_vs_capacity.csv'), index=False)
print(R.pivot_table(index=['task', 'comp'], columns='metric', values='est', sort=False).round(3))
print(R[R.task == 'seg'].round(3).to_string())
