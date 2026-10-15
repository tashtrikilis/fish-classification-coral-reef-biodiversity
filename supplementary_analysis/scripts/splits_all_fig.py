import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.patches import Rectangle
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'], 'font.size': 8,
    'axes.edgecolor': '#898781', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
BLUE, ORANGE = '#2a78d6', '#eb6834'; INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
FOCUS = ['Forcepsfish', 'Orangespine Unicornfish', 'Convict Tang']
o = pd.read_csv('splits_by_species.csv', index_col=0); n = len(o)
cmap = LinearSegmentedColormap.from_list('b', ['#e6effb', '#9cc0ef', '#2a78d6', '#123f78'])
norm = LogNorm(1, 800)
fig, axes = plt.subplots(1, 3, figsize=(7.0, 9.6), gridspec_kw=dict(width_ratios=[1.25, 1, 1], wspace=0.08))
y = np.arange(n)
# (a) total
ax = axes[0]
ax.barh(y, o.total, color=[ORANGE if s in FOCUS else BLUE for s in o.index], height=0.7, edgecolor='white', lw=0.6)
for i, v in enumerate(o.total): ax.text(v * 1.12, i, str(v), va='center', fontsize=6.5, color=INK2)
ax.set_xscale('log'); ax.set_xlim(0.8, 6000); ax.set_ylim(n - 0.5, -0.5)
ax.set_yticks(y); ax.set_yticklabels(o.index, fontsize=6.8)
for t in ax.get_yticklabels():
    if t.get_text() in FOCUS: t.set_fontweight('bold'); t.set_color(INK)
ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(['1', '10', '100', '1000'])
ax.minorticks_off(); ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.set_title('Total instances\n(log scale)', fontsize=8.5, color=INK)
# (b),(c) heatmaps
for ax, split, title in [(axes[1], 'test', 'Test fish per fold\n(each fish tested once)'), (axes[2], 'train', 'Train fish per fold\n(real, before copy-paste)')]:
    M = o[[f'{split}_f{k}' for k in range(5)]].values.astype(float)
    ax.imshow(np.where(M > 0, M, np.nan), cmap=cmap, norm=norm, aspect='auto')
    ax.set_facecolor('#ffffff')
    for i in range(n):
        for k in range(5):
            v = int(M[i, k])
            if v > 0: ax.text(k, i, str(v), ha='center', va='center', fontsize=5.8, color='white' if v >= 60 else INK)
            else: ax.text(k, i, '·', ha='center', va='center', fontsize=6, color=MUTED)
            if split == 'train' and v == 0 and o.iloc[i][f'test_f{k}'] > 0:
                ax.add_patch(Rectangle((k - 0.46, i - 0.44), 0.92, 0.88, fill=False, ec=ORANGE, lw=1.2))
    ax.set_xticks(range(5)); ax.set_xticklabels([str(k) for k in range(5)]); ax.set_xlabel('Fold', fontsize=8)
    ax.xaxis.set_ticks_position('top'); ax.xaxis.set_label_position('top')
    ax.set_yticks([]); ax.set_title(title, fontsize=8.5, color=INK, pad=22)
    ax.set_xticks(np.arange(-0.5, 5), minor=True); ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which='minor', color='white', lw=1.0); ax.tick_params(which='minor', length=0)
    for s in ax.spines.values(): s.set_visible(False)
axes[0].set_title('Total instances\n(log scale)', fontsize=8.5, color=INK, pad=22)
cut = int((o.total >= 20).sum()) - 0.5
for ax in axes: ax.axhline(cut, color=INK2, lw=0.9, ls=(0, (3, 2)))
fig.text(0.5, 0.012, 'Orange box: species tested in that fold but with no real training examples there (cannot be identified in that fold).\n'
         'Dashed line: classes above it have ≥ 20 instances (the macro mAP50 set).   Orange bars: focus species.   · = none.',
         ha='center', va='bottom', fontsize=6.6, color=INK2, linespacing=1.4)
fig.subplots_adjust(left=0.24, right=0.99, top=0.905, bottom=0.07)
for ext in ('png', 'pdf'): fig.savefig(f'cv_splits_by_species.{ext}', dpi=220)
