"""Baseline macro mAP50-95 (20 classes with >= 20 instances) and any-fish AP50-95 for Table 2."""
import sys, pickle, numpy as np; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import engine as G, common as C
B = pickle.load(open('boot.pkl','rb')); REL = B['reliable']
W1 = G.W[:1]; out = {}
for s in C.SETUPS:
  for iou in (['box'] if s.startswith('detect') else ['box','mask']):
    for sd in C.SEEDS:
      cd = G.classdata(s, 'baseline', sd, iou)
      m5095 = np.nanmean([np.mean([G.ap_rows(cd.get(c), t, W1)[0] for t in range(10)]) for c in REL])
      a5095 = np.mean([G.ap_rows(cd.get(-1), t, W1)[0] for t in range(10)])
      out[(s, iou, sd)] = (m5095, a5095); print(s, iou, sd, round(m5095,3), round(a5095,3), flush=True)
import pandas as pd, os
T = pd.DataFrame([dict(setup=s, iou=i, seed=sd, macro_mAP50_95=v[0], anyfish_AP50_95=v[1]) for (s, i, sd), v in out.items()])
T = T.groupby(['setup', 'iou'], as_index=False)[['macro_mAP50_95', 'anyfish_AP50_95']].mean()   # average of the two seeds
T.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tables', 'table2_baseline_map50_95.csv'), index=False)
print(T.round(3).to_string(index=False))
