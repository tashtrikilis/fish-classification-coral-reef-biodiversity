"""Tables 3-6 and the in-text numbers in the report, from boot.pkl (run run_boot.py first).

AP50 uses box IoU for detection and mask IoU for segmentation (box for both in the baseline seg vs det
comparison). Predictions are pooled over the 5 test folds and seeds are averaged inside each resample.
CIs are the 2.5-97.5 percentiles of 2000 paired block-bootstrap resamples (49 blocks);
p = 2 * min(P(diff <= 0), P(diff >= 0)), Holm-adjusted within each table.
"""
import os, sys, pickle, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common as C

OUT = os.path.join(HERE, '..', 'tables'); os.makedirs(OUT, exist_ok=True)
D = pickle.load(open('boot.pkl', 'rb'))
REL = D['reliable']                                   # the 20 classes with >= 20 labelled instances
NONF = [c for c in REL if c not in C.FOCUS_IDS]       # 17 non-target classes
LAB = {'detect_s': 'Detection S', 'detect_m': 'Detection M', 'detect_l': 'Detection L',
       'segment_s': 'Segmentation S', 'segment_m': 'Segmentation M', 'segment_l': 'Segmentation L'}


def iou_of(setup): return 'box' if setup.startswith('detect') else 'mask'


def ap(setup, arm, cls, iou=None, seeds=C.SEEDS):
    """seed-averaged bootstrap array (index 0 = point estimate, 1.. = resamples)"""
    iou = iou or iou_of(setup)
    return np.nanmean([D[(setup, arm, s, iou)][('AP50', cls)] for s in seeds], 0)


def mean_ap(setup, arm, classes, iou=None, seeds=C.SEEDS):
    return np.nanmean([ap(setup, arm, c, iou, seeds) for c in classes], 0)


def summ(x):
    b = x[1:][~np.isnan(x[1:])]
    return dict(est=x[0], lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                p=min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())), n_boot=len(b))


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); prev = 0
    for rank, i in enumerate(order):
        prev = max(prev, min(1.0, (m - rank) * ps[i])); adj[i] = prev
    return adj


# ---- Table 3: baseline AP50, focus species
rows = []
for s in C.SETUPS:
    r = {'model': LAB[s]}
    for c in C.FOCUS_IDS: r[C.NAMES[c]] = ap(s, 'baseline', c)[0]
    r['focus_mean'] = mean_ap(s, 'baseline', C.FOCUS_IDS)[0]
    rows.append(r)
T3 = pd.DataFrame(rows); T3.to_csv(os.path.join(OUT, 'table3_baseline_focus_ap50.csv'), index=False)
print('\nTable 3  Baseline AP50, focus species\n', T3.round(3).to_string(index=False))

# ---- Table 4: focus-species mean, augmented - baseline
rows = []
for s in C.SETUPS:
    d = mean_ap(s, 'copypaste', C.FOCUS_IDS) - mean_ap(s, 'baseline', C.FOCUS_IDS)
    r = summ(d)
    rows.append(dict(model=LAB[s], baseline=mean_ap(s, 'baseline', C.FOCUS_IDS)[0],
                     augmented=mean_ap(s, 'copypaste', C.FOCUS_IDS)[0], delta=r['est'], ci_lo=r['lo'], ci_hi=r['hi'],
                     p_raw=r['p'], n_boot=r['n_boot']))
T4 = pd.DataFrame(rows); T4['p_holm'] = holm(T4.p_raw.values)
T4.to_csv(os.path.join(OUT, 'table4_focus_mean_effect.csv'), index=False)
print('\nTable 4  Focus-species mean AP50 (Holm across 6; p = 0 means no resample <= 0, i.e. p < 1/2000 before adjustment)\n',
      T4.round(3).to_string(index=False))

# ---- Table 5: per species, mean of the 6 configurations
rows = []
for c in C.FOCUS_IDS:
    b = np.mean([ap(s, 'baseline', c) for s in C.SETUPS], 0); a = np.mean([ap(s, 'copypaste', c) for s in C.SETUPS], 0)
    r = summ(a - b)
    wins = sum(ap(s, 'copypaste', c, seeds=[sd])[0] > ap(s, 'baseline', c, seeds=[sd])[0] for s in C.SETUPS for sd in C.SEEDS)
    rows.append(dict(species=C.NAMES[c], n=int(D['gtcount'][c]) if 'gtcount' in D else np.nan, baseline=b[0], augmented=a[0],
                     delta=r['est'], ci_lo=r['lo'], ci_hi=r['hi'], p_raw=r['p'], runs_improved=f'{wins}/12'))
T5 = pd.DataFrame(rows); T5['p_holm'] = holm(T5.p_raw.values)
T5.to_csv(os.path.join(OUT, 'table5_species_effect.csv'), index=False)
print('\nTable 5  Per species, mean of 6 configurations (Holm across 3)\n', T5.round(3).to_string(index=False))

# ---- Table 6: macro, non-target and any-fish changes (unadjusted)
rows = []
for s in C.SETUPS:
    r = {'model': LAB[s]}
    for name, x in [('macro_mAP50', mean_ap(s, 'copypaste', REL) - mean_ap(s, 'baseline', REL)),
                    ('nontarget_AP50', mean_ap(s, 'copypaste', NONF) - mean_ap(s, 'baseline', NONF)),
                    ('anyfish_AP50', ap(s, 'copypaste', -1) - ap(s, 'baseline', -1))]:
        q = summ(x); r[name] = q['est']; r[name + '_lo'] = q['lo']; r[name + '_hi'] = q['hi']
    rows.append(r)
T6 = pd.DataFrame(rows); T6.to_csv(os.path.join(OUT, 'table6_nontarget_anyfish.csv'), index=False)
print('\nTable 6  Changes (augmented - baseline), unadjusted 95% CI\n', T6.round(3).to_string(index=False))

# ---- in text: baseline seg vs det (box macro mAP50)
rows = []
for size in 'sml':
    q = summ(mean_ap(f'segment_{size}', 'baseline', REL, iou='box') - mean_ap(f'detect_{size}', 'baseline', REL, iou='box'))
    rows.append(dict(size=size.upper(), seg_minus_det=q['est'], ci_lo=q['lo'], ci_hi=q['hi'], p_raw=q['p']))
TS = pd.DataFrame(rows); TS.to_csv(os.path.join(OUT, 'baseline_seg_vs_det_box.csv'), index=False)
print('\nBaseline segmentation - detection, box macro mAP50 (unadjusted)\n', TS.round(3).to_string(index=False))

# ---- in text: focus-mean gain for each configuration and seed
rows = []
for s in C.SETUPS:
    for sd in C.SEEDS:
        q = summ(mean_ap(s, 'copypaste', C.FOCUS_IDS, seeds=[sd]) - mean_ap(s, 'baseline', C.FOCUS_IDS, seeds=[sd]))
        rows.append(dict(model=LAB[s], seed=sd, delta=q['est'], ci_lo=q['lo'], ci_hi=q['hi'], ci_excludes_zero=q['lo'] > 0 or q['hi'] < 0))
TR = pd.DataFrame(rows); TR.to_csv(os.path.join(OUT, 'focus_gain_per_run.csv'), index=False)
print(f'\nPer-run focus-mean gains: {TR.delta.min():+.3f} to {TR.delta.max():+.3f}; '
      f'CI excludes zero in {int(TR.ci_excludes_zero.sum())} of {len(TR)} runs')
