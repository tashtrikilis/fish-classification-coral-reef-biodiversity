import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore'); sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import common as C
F = pd.read_csv('fish_level.csv'); Q = pd.read_csv('fp_level.csv')
N = C.NAMES
out = []
# --- confusion: FN wrong species (true focus -> predicted other) and FP wrong species (true other -> predicted focus)
for c in C.FOCUS_IDS:
    for a in C.ARMS:
        x = F[(F.cls == c) & (F.arm == a) & (F.cat == 'wrong_species')].pred_cls.value_counts()
        y = Q[(Q.pred_cls == c) & (Q.arm == a) & (Q.cat == 'wrong_species')].true_cls.value_counts()
        for k, v in x.head(4).items(): out.append(dict(species=N[c], arm=a, direction='focus fish labelled as', other=N[k], per_model=v / 12))
        for k, v in y.head(4).items(): out.append(dict(species=N[c], arm=a, direction='other fish labelled as focus', other=N[k], per_model=v / 12))
pd.DataFrame(out).to_csv('confusions.csv', index=False)
print(pd.DataFrame(out).round(2).to_string())
# --- confidence
rows = []
for c in C.FOCUS_IDS:
    for a in C.ARMS:
        f = F[(F.cls == c) & (F.arm == a)]; q = Q[(Q.pred_cls == c) & (Q.arm == a)]
        rows.append(dict(species=N[c], arm=a,
            conf_TP_median=f[f.cat == 'TP'].conf.median(), conf_TP_n=(f.cat == 'TP').sum(),
            conf_wrongFN_median=f[f.cat == 'wrong_species'].conf.median(),
            conf_FPwrong_median=q[q.cat == 'wrong_species'].conf.median(),
            conf_FPbg_median=q[q.cat == 'no_labelled_fish'].conf.median(),
            conf_missed_sub_same_median=f[f['sub'] == 'below_tau_same'].conf.median()))
print(pd.DataFrame(rows).round(3).to_string()); pd.DataFrame(rows).to_csv('confidence.csv', index=False)
