"""Dataset-description tables of the report, computed from the labels and the donor lists.

  Table 1  : instances, encounters and donors of the three focus species
  Table 12 : size of the copy-paste donors vs the labelled fish (shorter side of the box, px)
  Table 13 : instances, encounters, median size and crowding for the 20 classes with >= 20 instances
             (AP50, P and R in Table 13 come from results/detect_m/seeds0-1/per_class_pooled.csv)

Usage (from this folder, with the dataset zip extracted to ../../dataset/):
  python dataset_tables.py --dataset ../../dataset/kona_reef_fish_dataset --experiment ../../copy_paste_experiment
"""
import argparse, csv, os, re
import numpy as np, pandas as pd

W, H = 1280, 720
FOCUS = ['Forcepsfish', 'Orangespine Unicornfish', 'Convict Tang']

ap = argparse.ArgumentParser()
ap.add_argument('--dataset', default='../../dataset/kona_reef_fish_dataset')
ap.add_argument('--experiment', default='../../copy_paste_experiment')
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tables'))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

names = {int(k): v.strip() for k, v in re.findall(r'^\s+(\d+):\s*(.+)$', open(f'{a.dataset}/data.yaml').read(), re.M)}
inst = []                                            # frame, class, x0, y0, x1, y1 (px)
for fr in range(1000, 2462):
    for line in open(f'{a.dataset}/labels/frame_{fr}.txt'):
        p = line.split()
        if len(p) < 7: continue
        xy = np.array(p[1:], float).reshape(-1, 2) * [W, H]
        inst.append((fr, int(p[0]), *xy.min(0), *xy.max(0)))
df = pd.DataFrame(inst, columns=['frame', 'cls', 'x0', 'y0', 'x1', 'y1'])
df['name'] = df.cls.map(names); df['short'] = np.minimum(df.x1 - df.x0, df.y1 - df.y0)


def encounters(frames, gap=15):
    f = np.sort(np.unique(frames)); return int(1 + (np.diff(f) > gap).sum()) if len(f) else 0


def crowded(g):
    """share of instances whose box overlaps the box of another fish of the same class in the same frame"""
    hit = 0
    for _, fg in g.groupby('frame'):
        b = fg[['x0', 'y0', 'x1', 'y1']].values
        for i in range(len(b)):
            o = [(min(b[i, 2], b[j, 2]) > max(b[i, 0], b[j, 0])) and (min(b[i, 3], b[j, 3]) > max(b[i, 1], b[j, 1]))
                 for j in range(len(b)) if j != i]
            hit += any(o)
    return hit / len(g)


# donors (final pool after the visual quality check)
veto = {}
for line in open(f'{a.experiment}/cv_donor_veto.txt'):
    if line.strip() and not line.startswith('#'):
        p = line.strip().split(','); veto[p[0]] = {int(x) for x in p[1:]}
donor = {s: [] for s in FOCUS}
for r in csv.DictReader(open(f'{a.experiment}/experiment_donors.csv')):
    if r['species'] == 'Forcepsfish': donor['Forcepsfish'].append(min(float(r['bbox_w']), float(r['bbox_h'])))
for r in csv.DictReader(open(f'{a.experiment}/cv_donor_candidates.csv')):
    if r['species'] in donor and int(r['idx']) not in veto.get(r['species'], set()):
        donor[r['species']].append(float(r['min_side_px']))

# Table 1
t1 = pd.DataFrame([dict(species=s, instances=int((df.name == s).sum()), encounters=encounters(df.frame[df.name == s]),
                        donors=len(donor[s])) for s in FOCUS])
t1.to_csv(f'{a.out}/table1_focus_species.csv', index=False); print('Table 1\n', t1.to_string(index=False))

# Table 12
rows = []
for s in FOCUS:
    for grp, v in [('Donors', np.array(donor[s])), ('Labelled fish', df.short[df.name == s].values)]:
        rows.append(dict(species=s, group=grp, n=len(v), min=v.min(), q1=np.percentile(v, 25), median=np.median(v),
                         q3=np.percentile(v, 75), max=v.max()))
t12 = pd.DataFrame(rows); t12.to_csv(f'{a.out}/table12_donor_sizes.csv', index=False)
print('\nTable 12\n', t12.round(0).to_string(index=False))

# Table 13 (dataset columns)
cnt = df.name.value_counts(); rel = cnt[cnt >= 20].index
t13 = pd.DataFrame([dict(cls=c, instances=int(cnt[c]), encounters=encounters(df.frame[df.name == c]),
                         size_px=float(np.median(df.short[df.name == c])), crowded=crowded(df[df.name == c])) for c in rel])
t13.to_csv(f'{a.out}/table13_class_characteristics.csv', index=False)
print('\nTable 13 (dataset columns)\n', t13.round(2).to_string(index=False))
print(f'\nAll fish: {len(df)} instances, median shorter side {np.median(df.short):.0f} px, '
      f'{(df.short < 20).mean():.0%} under 20 px')
