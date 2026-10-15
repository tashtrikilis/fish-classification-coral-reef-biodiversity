import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
o = pd.read_csv('../../supplementary_analysis/tables/cv_splits_by_species.csv', index_col=0)['total']
o = o.rename(index={'Bullethead Unicornfish': 'Bullethead Parrotfish', 'Lognnose Butterflyfish': 'Longnose Butterflyfish', 'Ornate Butterfly Fish': 'Ornate Butterflyfish'})
o = o.sort_values(ascending=False)
FOCUS = ['Forcepsfish', 'Orangespine Unicornfish', 'Convict Tang']
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e1e0d9'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#898781', 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, ax = plt.subplots(figsize=(6.6, 6.4))
y = range(len(o))
ax.barh(y, o.values, color=[ORANGE if s in FOCUS else BLUE for s in o.index], height=0.72, edgecolor='white', lw=0.5)
for i, v in enumerate(o.values): ax.text(v * 1.08, i, str(v), va='center', fontsize=6.8, color=MUTED, zorder=4, bbox=dict(facecolor='white', edgecolor='none', pad=0.6))
ax.set_xscale('log'); ax.set_xlim(0.85, 3000); ax.set_ylim(len(o) - 0.5, -0.5)
ax.set_yticks(list(y)); ax.set_yticklabels(o.index, fontsize=7.2)
for t in ax.get_yticklabels():
    if t.get_text() in FOCUS: t.set_fontweight('bold'); t.set_color(INK)
ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(['1', '10', '100', '1,000']); ax.minorticks_off()
ax.axvline(20, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.text(21, len(o) - 1.2, '20 instances', fontsize=6.8, color=MUTED, va='center')
ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.set_xlabel('Labelled instances (log scale)', color=MUTED)
fig.tight_layout()
for e in ('pdf', 'png'): fig.savefig(f'class_distribution.{e}', dpi=220)
print(len(o), o.sum())
