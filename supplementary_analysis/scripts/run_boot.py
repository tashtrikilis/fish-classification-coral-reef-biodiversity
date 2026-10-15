import sys, pickle, time, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import engine as G, common as C
gtcount = {}
for f in C.FOLDS:
    for r in C.GT[f]: gtcount[int(r[0])] = gtcount.get(int(r[0]), 0) + 1
RELIABLE = sorted(c for c, n in gtcount.items() if n >= 20)
res = {'W': G.W, 'reliable': RELIABLE, 'gtcount': gtcount}
jobs = [(s, a, sd, 'box') for s in C.SETUPS for a in C.ARMS for sd in C.SEEDS] + \
       [(s, a, sd, 'mask') for s in C.SETUPS if s.startswith('segment') for a in C.ARMS for sd in C.SEEDS]
for (s, a, sd, iou) in jobs:
    t = time.time()
    cd = G.classdata(s, a, sd, iou)
    d = {}
    for c in RELIABLE:
        d[('AP50', c)] = G.ap_rows(cd.get(c), 0, G.W)
    for c in C.FOCUS_IDS:
        d[('AP50-95', c)] = np.mean([G.ap_rows(cd.get(c), t_, G.W) for t_ in range(10)], 0)
    d[('AP50', -1)] = G.ap_rows(cd.get(-1), 0, G.W)
    res[(s, a, sd, iou)] = d
    print(s, a, sd, iou, f'{time.time()-t:.0f}s', flush=True)
    pickle.dump(res, open('boot.pkl', 'wb'))
print('done')
