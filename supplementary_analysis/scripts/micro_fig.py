import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'], 'font.size': 8.5,
    'axes.edgecolor': '#898781', 'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.spines.top': False, 'axes.spines.right': False})
BLUE, ORANGE = '#2a78d6', '#eb6834'; INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
T = pd.read_csv('micro_results.csv')
T = T[((T.setup.str.startswith('detect')) & (T.iou == 'box')) | ((T.setup.str.startswith('segment')) & (T.iou == 'mask'))]
S = ['detect_s', 'detect_m', 'detect_l', 'segment_s', 'segment_m', 'segment_l']
LBL = {'detect_s': 'Det small', 'detect_m': 'Det medium', 'detect_l': 'Det large', 'segment_s': 'Seg small', 'segment_m': 'Seg medium', 'segment_l': 'Seg large'}
MET = [('Focus mean AP50', 'Focus species\n(mean of 3)'), ('Macro AP50 (20 classes)', 'Macro AP50\n(mean of 20 species)'),
       ('Micro AP50', 'Micro AP50\n(every fish equal)'), ('Any-fish AP50', 'Any-fish AP50\n(species ignored)')]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.7), sharey=True, sharex=True)
for ax, (m, title) in zip(axes, MET):
    for i, s in enumerate(S):
        r = T[(T.setup == s) & (T.metric == m)].iloc[0]
        col = ORANGE if r.ci_lo > 0 else (BLUE if r.ci_hi >= 0 else MUTED)
        ax.plot([r.ci_lo, r.ci_hi], [i, i], color=col, lw=2, solid_capstyle='round')
        ax.plot(r.delta, i, 'o', color=col, ms=5, mec='white', mew=0.8, zorder=3)
    ax.axvline(0, color=INK2, lw=0.8); ax.set_title(title, fontsize=8.2, color=INK)
    ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlim(-0.06, 0.24)
    ax.set_xticks([0, 0.1, 0.2]); ax.set_xticklabels(['0', '+0.1', '+0.2'])
axes[0].set_yticks(range(6)); axes[0].set_yticklabels([LBL[s] for s in S]); axes[0].set_ylim(5.6, -0.6)
fig.text(0.145, 0.975, 'Orange: 95% CI above zero.   Blue: CI includes zero.   Det = box AP50, Seg = mask AP50.', fontsize=7, color=INK2, va='top')
fig.supxlabel('Copy-paste − baseline (AP50, seeds 0+1, 95% block-bootstrap CI)', fontsize=8, color=INK2, y=0.02)
fig.subplots_adjust(left=0.145, right=0.99, top=0.80, bottom=0.2, wspace=0.12)
for ext in ('png', 'pdf'): fig.savefig(f'micro_vs_macro.{ext}', dpi=220)
