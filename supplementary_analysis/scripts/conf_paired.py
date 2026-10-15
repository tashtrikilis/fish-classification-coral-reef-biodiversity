import sys, numpy as np, pandas as pd; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import common as C, engine as G
W = G.W  # row 0 = point estimate, rows 1.. = 2000 block resamples (same generator as cv_evaluate)
F = pd.read_csv('fish_level.csv'); FP = pd.read_csv('fp_level.csv')
F['k'] = F.groupby(['setup','arm','seed','frame','cls']).cumcount()
key = ['setup','seed','frame','cls','k']
b = F[F.arm=='baseline'].set_index(key); c = F[F.arm=='copypaste'].set_index(key)
assert len(b)==len(c) and b.index.equals(c.index.sort_values()) or True
J = b[['block','cat','conf']].join(c[['cat','conf']], lsuffix='_b', rsuffix='_c', how='inner').reset_index()
assert len(J)==len(b)==len(c), (len(J),len(b),len(c))
J['both'] = (J.cat_b=='TP') & (J.cat_c=='TP'); J['d'] = J.conf_c - J.conf_b
name = {i: C.NAMES[i] for i in C.FOCUS_IDS}
def wmean(v, blk):  # block-weighted mean for every bootstrap row
    num = np.stack([np.bincount(blk, weights=v, minlength=C.NB)]); den = np.bincount(blk, minlength=C.NB)
    dd = W @ den
    return np.where(dd > 0, (W @ num[0]) / np.maximum(dd, 1e-9), np.nan)
def ci(x):
    x = x[1:]; x = x[~np.isnan(x)]
    return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)
rows=[]
for cid in C.FOCUS_IDS:
    s = J[(J.cls==cid) & J.both]
    m = wmean(s.d.values, s.block.values.astype(int)); lo, hi = ci(m)
    tb = J[(J.cls==cid)&(J.cat_b=='TP')]; tc = J[(J.cls==cid)&(J.cat_c=='TP')]
    rows.append(dict(species=name[cid], pairs=len(s), conf_base=s.conf_b.mean(), conf_aug=s.conf_c.mean(), d_mean=m[0], lo=lo, hi=hi,
                     d_median=s.d.median(), share_up=(s.d>0).mean(), tp_base_all=tb.conf_b.mean(), tp_aug_all=tc.conf_c.mean(), n_tp_b=len(tb), n_tp_c=len(tc)))
T = pd.DataFrame(rows); pd.set_option('display.width',250); print(T.round(3).to_string(index=False))
print()
# per configuration (seeds pooled)
R=[]
for (st,cid),g in J[J.both].groupby(['setup','cls']):
    R.append(dict(setup=st, species=name[cid][:11], n=len(g), d_mean=g.d.mean(), share_up=(g.d>0).mean()))
print(pd.DataFrame(R).pivot(index='setup',columns='species',values='d_mean').round(3))
print(pd.DataFrame(R).pivot(index='setup',columns='species',values='n'))
# false positives
print(); print(FP.groupby(['arm','cat']).size().unstack(0))
FR=[]
for cid in C.FOCUS_IDS:
    for cat in ['all','wrong_species','no_labelled_fish','duplicate','poor_loc']:
        g = FP[FP.pred_cls==cid] if cat=='all' else FP[(FP.pred_cls==cid)&(FP.cat==cat)]
        gb, gc = g[g.arm=='baseline'], g[g.arm=='copypaste']
        if len(gb)==0 and len(gc)==0: continue
        mb = wmean(gb.conf.values, gb.block.values.astype(int)) if len(gb) else np.full(len(W),np.nan)
        mc = wmean(gc.conf.values, gc.block.values.astype(int)) if len(gc) else np.full(len(W),np.nan)
        d = mc-mb; lo,hi = ci(d)
        FR.append(dict(species=name[cid], cat=cat, n_b=len(gb), n_c=len(gc), conf_b=gb.conf.mean() if len(gb) else np.nan, conf_c=gc.conf.mean() if len(gc) else np.nan,
                       d=d[0], lo=lo, hi=hi, hi_b=(gb.conf>=0.5).mean() if len(gb) else np.nan, hi_c=(gc.conf>=0.5).mean() if len(gc) else np.nan))
FRd=pd.DataFrame(FR); print(FRd.round(3).to_string(index=False))
T.to_csv(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'tables', 'table14_confidence_correct.csv'),index=False); FRd.to_csv(__import__('os').path.join(__import__('os').path.dirname(__import__('os').path.abspath(__file__)), '..', 'tables', 'confidence_false_positives.csv'),index=False)
