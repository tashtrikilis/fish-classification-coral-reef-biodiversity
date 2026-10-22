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
pickle.dump(out, open('t61.pkl','wb'))
