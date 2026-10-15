import sys, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import common as C
D = pickle.load(open('boot.pkl', 'rb'))
W = D['W']; REL = D['reliable']; NONF = [c for c in REL if c not in C.FOCUS_IDS]
SIZES = ['s', 'm', 'l']
def sm(task, size): return f"{'detect' if task == 'det' else 'segment'}_{size}"

def arr(setup, arm, iou, met, cls):
    """seed-averaged bootstrap array (R+1,) for a single class metric"""
    return np.nanmean([D[(setup, arm, sd, iou)][(met, cls)] for sd in C.SEEDS], 0)
def focus(setup, arm, iou, met='AP50', cls=None):
    ids = C.FOCUS_IDS if cls is None else [cls]
    return np.nanmean([arr(setup, arm, iou, met, c) for c in ids], 0)
def macro(setup, arm, iou, classes): return np.nanmean([arr(setup, arm, iou, 'AP50', c) for c in classes], 0)
def anyfish(setup, arm, iou): return arr(setup, arm, iou, 'AP50', -1)

def summ(x):
    b = x[1:][~np.isnan(x[1:])]
    return dict(est=x[0], lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                p=min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())))
def fmt(s): return f"{s['est']:+.3f} [{s['lo']:+.3f}, {s['hi']:+.3f}]"

# =========================== B. detection vs segmentation
rows = []
targets = [('Focus-species mean', None)] + [(C.NAMES[c], c) for c in C.FOCUS_IDS]
for size in SIZES:
    det, seg = sm('det', size), sm('seg', size)
    for met in ['AP50', 'AP50-95']:
        for name, cid in targets:
            eff = {}
            eff['det_box'] = focus(det, 'copypaste', 'box', met, cid) - focus(det, 'baseline', 'box', met, cid)
            eff['seg_box'] = focus(seg, 'copypaste', 'box', met, cid) - focus(seg, 'baseline', 'box', met, cid)
            eff['seg_mask'] = focus(seg, 'copypaste', 'mask', met, cid) - focus(seg, 'baseline', 'mask', met, cid)
            base = {'det_box': focus(det, 'baseline', 'box', met, cid)[0], 'seg_box': focus(seg, 'baseline', 'box', met, cid)[0],
                    'seg_mask': focus(seg, 'baseline', 'mask', met, cid)[0]}
            r = dict(size=size, metric=met, target=name)
            for k, v in eff.items():
                s = summ(v); r[f'{k}_base'] = base[k]; r[f'{k}_effect'] = s['est']; r[f'{k}_lo'] = s['lo']; r[f'{k}_hi'] = s['hi']; r[f'{k}_p'] = s['p']
            for lab, a, b in [('DiD_segbox_minus_detbox', 'seg_box', 'det_box'),
                              ('DiD_segmask_minus_detbox', 'seg_mask', 'det_box'),
                              ('mask_minus_box_within_seg', 'seg_mask', 'seg_box')]:
                s = summ(eff[a] - eff[b]); r[f'{lab}'] = s['est']; r[f'{lab}_lo'] = s['lo']; r[f'{lab}_hi'] = s['hi']; r[f'{lab}_p'] = s['p']
            rows.append(r)
B = pd.DataFrame(rows); B.to_csv('task_comparison.csv', index=False)
# pooled over sizes (mean of the three matched-size DiDs)
pool = []
for met in ['AP50', 'AP50-95']:
    for name, cid in targets:
        dd = {k: [] for k in ['bb', 'mb', 'mm']}
        for size in SIZES:
            det, seg = sm('det', size), sm('seg', size)
            e_db = focus(det, 'copypaste', 'box', met, cid) - focus(det, 'baseline', 'box', met, cid)
            e_sb = focus(seg, 'copypaste', 'box', met, cid) - focus(seg, 'baseline', 'box', met, cid)
            e_sm = focus(seg, 'copypaste', 'mask', met, cid) - focus(seg, 'baseline', 'mask', met, cid)
            dd['bb'].append(e_sb - e_db); dd['mb'].append(e_sm - e_db); dd['mm'].append(e_sm - e_sb)
        r = dict(metric=met, target=name)
        for k, lab in [('bb', 'segbox_minus_detbox'), ('mb', 'segmask_minus_detbox'), ('mm', 'mask_minus_box')]:
            s = summ(np.mean(dd[k], 0)); r.update({lab: s['est'], lab + '_lo': s['lo'], lab + '_hi': s['hi'], lab + '_p': s['p']})
        pool.append(r)
pd.DataFrame(pool).to_csv('task_comparison_pooled_sizes.csv', index=False)

# =========================== C. augmentation vs capacity
rows = []
for task in ['det', 'seg']:
    for size in SIZES:
        st = sm(task, size)
        for arm in C.ARMS:
            r = dict(task=task, size=size, arm=arm,
                     focus_AP50_box=focus(st, arm, 'box')[0],
                     focus_AP50_mask=focus(st, arm, 'mask')[0] if task == 'seg' else np.nan,
                     macro_AP50=macro(st, arm, 'box', REL)[0], nonfocus17_AP50=macro(st, arm, 'box', NONF)[0],
                     anyfish_AP50=anyfish(st, arm, 'box')[0])
            for k in ['focus_AP50_box', 'macro_AP50', 'nonfocus17_AP50', 'anyfish_AP50']:
                x = {'focus_AP50_box': focus(st, arm, 'box'), 'macro_AP50': macro(st, arm, 'box', REL),
                     'nonfocus17_AP50': macro(st, arm, 'box', NONF), 'anyfish_AP50': anyfish(st, arm, 'box')}[k]
                b = x[1:][~np.isnan(x[1:])]; r[k + '_lo'] = np.percentile(b, 2.5); r[k + '_hi'] = np.percentile(b, 97.5)
            rows.append(r)
P = pd.DataFrame(rows); P.to_csv('tradeoff_performance.csv', index=False)

comp = []
def M(task, size, arm, what):
    st = sm(task, size)
    return {'focus': focus(st, arm, 'box'), 'macro': macro(st, arm, 'box', REL), 'nonfocus': macro(st, arm, 'box', NONF),
            'anyfish': anyfish(st, arm, 'box')}[what]
for task in ['det', 'seg']:
    for small, big in [('s', 'm'), ('m', 'l')]:
        for what in ['focus', 'macro', 'nonfocus', 'anyfish']:
            aug_small = M(task, small, 'copypaste', what); base_small = M(task, small, 'baseline', what)
            base_big = M(task, big, 'baseline', what)
            for lab, x in [(f'aug {small} − base {big}', aug_small - base_big),
                           (f'augmentation gain at {small}', aug_small - base_small),
                           (f'capacity gain {small}→{big}', base_big - base_small),
                           (f'aug gain − capacity gain ({small})', (aug_small - base_small) - (base_big - base_small))]:
                s = summ(x); comp.append(dict(task=task, comparison=lab, metric=what, est=s['est'], lo=s['lo'], hi=s['hi'], p=s['p']))
    for size in ['l']:
        for what in ['focus', 'macro', 'nonfocus', 'anyfish']:
            s = summ(M(task, size, 'copypaste', what) - M(task, size, 'baseline', what))
            comp.append(dict(task=task, comparison=f'augmentation gain at {size}', metric=what, est=s['est'], lo=s['lo'], hi=s['hi'], p=s['p']))
pd.DataFrame(comp).to_csv('tradeoff_comparisons.csv', index=False)
print(B[B.metric == 'AP50'][['size', 'target', 'det_box_effect', 'seg_box_effect', 'seg_mask_effect',
      'DiD_segbox_minus_detbox', 'DiD_segbox_minus_detbox_lo', 'DiD_segbox_minus_detbox_hi', 'DiD_segbox_minus_detbox_p',
      'DiD_segmask_minus_detbox', 'DiD_segmask_minus_detbox_lo', 'DiD_segmask_minus_detbox_hi']].round(3).to_string())
print(pd.DataFrame(pool).round(3).to_string())
print(P.round(3).to_string())
print(pd.DataFrame(comp).round(3).to_string())
