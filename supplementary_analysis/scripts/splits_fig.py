import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'], 'font.size': 8.5,
    'axes.edgecolor': '#898781', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
    'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE, YELLOW = '#2a78d6', '#eb6834', '#eda100'; INK, MUTED, GRID = '#0b0b0b', '#898781', '#e1e0d9'
D = pd.read_csv('splits.csv')
SPL = [('train', 'Train', BLUE), ('val', 'Val (epoch selection)', YELLOW), ('test', 'Test', ORANGE), ('buffer', 'Buffer (unused)', '#cfcdc4')]
fig = plt.figure(figsize=(7.0, 5.2))
gs = fig.add_gridspec(2, 6, hspace=0.62, wspace=1.3)
panels = [(gs[0, 0:3], 'frames', 'Frames'), (gs[0, 3:6], 'instances', 'Labelled fish (all classes)'),
          (gs[1, 0:2], 'Forcepsfish', 'Forcepsfish'), (gs[1, 2:4], 'Orangespine Unicornfish', 'Orangespine Unicornfish'),
          (gs[1, 4:6], 'Convict Tang', 'Convict Tang')]
for i, (g, col, title) in enumerate(panels):
    ax = fig.add_subplot(g); bottom = [0]*5
    for s, lab, c in SPL:
        v = [int(D[(D.fold == k) & (D.split == s)][col].iloc[0]) for k in range(5)]
        ax.bar(range(5), v, bottom=bottom, color=c, edgecolor='white', lw=0.8, width=0.72, label=lab)
        if s == 'test':
            for k in range(5):
                if v[k] > 0 and (col in ('frames', 'instances') or v[k] >= 3):
                    ax.text(k, bottom[k] + v[k]/2, str(v[k]), ha='center', va='center', fontsize=6.5 if i > 1 else 7, color='white')
        bottom = [b + x for b, x in zip(bottom, v)]
    ax.set_xticks(range(5)); ax.set_xticklabels([str(k) for k in range(5)], fontsize=7.5); ax.set_xlabel('Fold', fontsize=8)
    ax.set_title(title, fontsize=9 if i < 2 else 8.5, color=INK)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    if i in (0, 2): ax.set_ylabel('Count')
    if i == 0: h, l = ax.get_legend_handles_labels()
fig.text(0.5, 0.50, 'Focus species instances', ha='center', fontsize=9, color=INK, style='italic')
fig.legend(h, l, ncol=4, frameon=False, fontsize=7.5, loc='lower center', bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(top=0.94, bottom=0.12, left=0.08, right=0.99)
for ext in ('png', 'pdf'): fig.savefig(f'cv_splits_hist.{ext}', dpi=220)
