import numpy as np, pandas as pd
from collections import Counter
from common import FOLDS, GT, NAMES, FOCUS
frames = sorted(FOLDS); fold_of = {f: FOLDS[f][0] for f in frames}; block_of = {f: FOLDS[f][1] for f in frames}
blocks = sorted(set(block_of.values())); assign = {block_of[f]: fold_of[f] for f in frames}
rows = []
for k in range(5):
    test = [f for f in frames if fold_of[f] == k]
    nxt = sorted(b for b in blocks if assign[b] == (k+1) % 5); vb = set(nxt[::2])
    val = [f for f in frames if block_of[f] in vb]; held = np.array(test); train, drop = [], []
    for f in frames:
        if fold_of[f] == k or block_of[f] in vb: continue
        (drop if np.min(np.abs(held-f)) <= 15 else train).append(f)
    for s, fs in dict(train=train, val=val, test=test, buffer=drop).items():
        c = Counter(int(x) for f in fs for x in GT[f][:, 0])
        for cid, n in c.items(): rows.append(dict(fold=k, split=s, cls=cid, species=NAMES[cid], n=n))
L = pd.DataFrame(rows)
W = L.pivot_table(index='species', columns=['split', 'fold'], values='n', aggfunc='sum', fill_value=0)
tot = W['test'].sum(1)
W = W.loc[tot.sort_values(ascending=False).index]
out = pd.DataFrame({'total': W['test'].sum(1)})
for s in ['test', 'train', 'val', 'buffer']:
    for k in range(5): out[f'{s}_f{k}'] = W[s][k] if k in W[s] else 0
out['folds_tested'] = (W['test'] > 0).sum(1)
out['min_train'] = W['train'].min(1)
out.to_csv('splits_by_species.csv')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 100)
print(out[['total'] + [f'test_f{k}' for k in range(5)] + [f'train_f{k}' for k in range(5)] + ['folds_tested', 'min_train']].to_string())
print(len(out), out.total.sum())
