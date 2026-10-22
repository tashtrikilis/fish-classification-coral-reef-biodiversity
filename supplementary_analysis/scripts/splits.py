import numpy as np, csv, json
from collections import Counter
from common import FOLDS, GT, NAMES, FOCUS, FOCUS_IDS
frames = sorted(FOLDS); fold_of = {f: FOLDS[f][0] for f in frames}; block_of = {f: FOLDS[f][1] for f in frames}
blocks = sorted(set(block_of.values())); assign = {block_of[f]: fold_of[f] for f in frames}
rows = []
for k in range(5):
    test = [f for f in frames if fold_of[f] == k]
    nxt = sorted(b for b in blocks if assign[b] == (k+1) % 5); vb = set(nxt[::2])
    val = [f for f in frames if block_of[f] in vb]
    held = np.array(test); train, drop = [], []
    for f in frames:
        if fold_of[f] == k or block_of[f] in vb: continue
        (drop if np.min(np.abs(held-f)) <= 15 else train).append(f)
    unused = [f for f in frames if block_of[f] in set(nxt[1::2]) and f not in train and f not in drop]
    sets = dict(train=train, val=val, test=test, buffer=drop)
    for s, fs in sets.items():
        c = Counter(int(x) for f in fs for x in GT[f][:, 0])
        rows.append(dict(fold=k, split=s, frames=len(fs), fish_frames=sum(len(GT[f]) > 0 for f in fs),
                         instances=sum(c.values()), classes=len(c), **{n: c[i] for n, i in zip(FOCUS, FOCUS_IDS)}))
    assert len(train)+len(val)+len(test)+len(drop) == len(frames)
import pandas as pd
D = pd.DataFrame(rows); D.to_csv('splits.csv', index=False)
pd.set_option('display.width', 200); print(D.to_string(index=False))
