"""Class distribution of the re-annotated dataset as a wide column chart (species on the x-axis)."""
import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
o = pd.read_csv('../../supplementary_analysis/tables/cv_splits_by_species.csv', index_col=0)['total']
o = o.rename(index={'Bullethead Unicornfish': 'Bullethead Parrotfish', 'Lognnose Butterflyfish': 'Longnose Butterflyfish', 'Ornate Butterfly Fish': 'Ornate Butterflyfish'})
o = o.sort_values(ascending=False)
FOCUS = ['Forcepsfish', 'Orangespine Unicornfish', 'Convict Tang']
BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e1e0d9'
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#898781', 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, ax = plt.subplots(figsize=(7.2, 3.9))
x = range(len(o))
ax.bar(x, o.values, color=[ORANGE if s in FOCUS else BLUE for s in o.index], width=0.72, edgecolor='white', lw=0.5)
for i, v in enumerate(o.values):
    ax.text(i, v * 1.12, f'{v:,}', ha='center', va='bottom', rotation=90, fontsize=6.3, color=MUTED, zorder=4, bbox=dict(facecolor='white', edgecolor='none', pad=0.4))
ax.set_yscale('log'); ax.set_ylim(0.85, 6000); ax.set_xlim(-0.6, len(o) - 0.4)
ax.set_xticks(list(x)); ax.set_xticklabels(o.index, rotation=60, ha='right', rotation_mode='anchor', fontsize=6.6)
for t in ax.get_xticklabels():
    if t.get_text() in FOCUS: t.set_fontweight('bold'); t.set_color(INK)
ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(['1', '10', '100', '1,000']); ax.minorticks_off()
ax.axhline(20, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
ax.text(len(o) - 0.5, 23, '20 instances', fontsize=6.8, color=MUTED, ha='right', va='bottom')
ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.set_ylabel('Labelled instances (log scale)', color=MUTED)
fig.tight_layout()
for e in ('pdf', 'png'): fig.savefig(f'class_distribution_horizontal.{e}', dpi=220)
print(len(o), o.sum())
