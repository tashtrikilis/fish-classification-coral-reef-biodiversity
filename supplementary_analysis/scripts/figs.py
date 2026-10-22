import sys, warnings, numpy as np, pandas as pd, matplotlib
warnings.filterwarnings('ignore'); matplotlib.use('Agg'); sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import matplotlib.pyplot as plt, matplotlib.ticker
import common as C
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif'], 'font.size': 8.5,
                     'axes.edgecolor': '#898781', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e1e0d9'
ARMC = {'baseline': BLUE, 'copypaste': ORANGE}; ARML = {'baseline': 'Baseline', 'copypaste': 'Copy-paste'}
SPN = [C.NAMES[c] for c in C.FOCUS_IDS]
SHORT = {'Forcepsfish': 'Forcepsfish', 'Orangespine Unicornfish': 'Orangespine Unicornfish', 'Convict Tang': 'Convict Tang'}
F = pd.read_csv('fish_level.csv')
def save(fig, name):
    fig.savefig(f'figs/{name}.pdf', bbox_inches='tight'); fig.savefig(f'figs/{name}.png', bbox_inches='tight', dpi=200); plt.close(fig)

# ---------- E1 fate of every focus fish
cats = [('TP', 'Correct species', BLUE), ('wrong_species', 'Wrong species', ORANGE), ('poor_loc', 'Poorly localised (IoU 0.1–0.5)', AQUA),
        ('miss_same', 'Missed; correct-species score < 0.25', YELLOW), ('miss_other', 'Missed; no / other-species score', MAGENTA)]
def fate(df):
    n = len(df)
    return {'TP': (df.cat == 'TP').sum() / n, 'wrong_species': (df.cat == 'wrong_species').sum() / n,
            'poor_loc': df.cat.isin(['poor_loc_same', 'poor_loc_other']).sum() / n,
            'miss_same': (df['sub'] == 'below_tau_same').sum() / n,
            'miss_other': df['sub'].isin(['below_tau_other', 'no_prediction']).sum() / n}
fig, ax = plt.subplots(figsize=(6.9, 2.9))
ylab, y = [], 0
for sp in SPN:
    for arm in ['baseline', 'copypaste']:
        fr = fate(F[(F.cls == C.FOCUS_IDS[SPN.index(sp)]) & (F.arm == arm)])
        left = 0
        for k, lab, col in cats:
            v = fr[k]
            ax.barh(y, v, left=left, color=col, height=0.72, edgecolor='white', linewidth=1.0, label=lab if y == 0 else None)
            if v >= 0.06: ax.text(left + v / 2, y, f'{100*v:.0f}', ha='center', va='center', fontsize=7, color='white' if col in (BLUE, ORANGE) else INK)
            left += v
        ylab.append(f"{SHORT[sp]} – {ARML[arm]}"); y += 1
    y += 0.5
ypos = [i + 0.5 * (i // 2) for i in range(6)]
ax.set_yticks(ypos); ax.set_yticklabels(ylab); ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.set_xlabel('Share of labelled fish (pooled over 6 set-ups × 2 seeds); confidence ≥ 0.25, box IoU ≥ 0.5')
ax.tick_params(axis='y', length=0); ax.spines['left'].set_visible(False)
ax.legend(ncol=3, frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.45, -0.22))
save(fig, 'fig_err_fate')

# ---------- E2 recall by size
S = pd.read_csv('size_bins.csv')
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3), sharey=True)
for ax, sp in zip(axes, SPN):
    s = S[S.species == sp].reset_index(drop=True); x = np.arange(len(s))
    for j, (arm, col, off) in enumerate([('base', BLUE, -0.12), ('cp', ORANGE, 0.12)]):
        ax.plot(x + off, s[f'recall_{arm}'], 'o', color=col, ms=5, label=['Baseline', 'Copy-paste'][j], zorder=3)
    for i, r in s.iterrows():
        ax.text(i, -0.12, f'n={r.n_fish:.0f}', ha='center', fontsize=6.5, color=MUTED)
    ax.set_xticks(x); ax.set_xticklabels(s.size_bin); ax.set_title(SHORT[sp], fontsize=8.5, color=INK)
    ax.set_ylim(-0.17, 1); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel('Shorter box side (px)')
axes[0].set_ylabel('Recall (conf ≥ 0.25)'); axes[0].legend(frameon=False, fontsize=7, loc='upper left')
save(fig, 'fig_err_size')

# ---------- E3 encounters
E = pd.read_csv('encounters.csv')
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5), sharey=True)
for ax, sp in zip(axes, SPN):
    e = E[E.species == sp].reset_index(drop=True); x = np.arange(len(e))
    for i, r in e.iterrows():
        ax.plot([i, i], [r.recall_base, r.recall_cp], color=GRID, lw=1.6, zorder=1)
    ax.scatter(x, e.recall_base, s=10 + 4 * e.n_fish, color=BLUE, zorder=2, label='Baseline', edgecolor='white', lw=0.6)
    ax.scatter(x, e.recall_cp, s=10 + 4 * e.n_fish, color=ORANGE, zorder=3, label='Copy-paste', edgecolor='white', lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels([str(i + 1) for i in x], fontsize=7)
    ax.set_title(SHORT[sp], fontsize=8.5, color=INK); ax.set_xlabel('Encounter (in time order)')
    ax.set_ylim(-0.03, 1.03); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel('Recall per encounter'); axes[0].legend(frameon=False, fontsize=7, loc='upper left', markerscale=0.6)
save(fig, 'fig_err_encounters')

# ---------- E4 confidence
Q = pd.read_csv('fp_level.csv')
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.4), sharey=True)
groups = [('TP', lambda f, q, a: f[(f.cat == 'TP') & (f.arm == a)].conf),
          ('FN:\nwrong sp.', lambda f, q, a: f[(f.cat == 'wrong_species') & (f.arm == a)].conf),
          ('FP:\nother sp.', lambda f, q, a: q[(q.cat == 'wrong_species') & (q.arm == a)].conf),
          ('FP: no\nlabel', lambda f, q, a: q[(q.cat == 'no_labelled_fish') & (q.arm == a)].conf)]
for ax, sp in zip(axes, SPN):
    c = C.FOCUS_IDS[SPN.index(sp)]; f = F[F.cls == c]; q = Q[Q.pred_cls == c]
    for gi, (lab, fn) in enumerate(groups):
        for j, arm in enumerate(['baseline', 'copypaste']):
            v = fn(f, q, arm).dropna().values
            if len(v) == 0: continue
            b = ax.boxplot(v, positions=[gi + (j - 0.5) * 0.34], widths=0.28, patch_artist=True, showfliers=False,
                           medianprops=dict(color=INK, lw=1), whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
            b['boxes'][0].set(facecolor=ARMC[arm], edgecolor='white', alpha=0.9)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], fontsize=5.8)
    ax.set_title(SHORT[sp], fontsize=8.5, color=INK); ax.set_ylim(0.2, 1.0)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel('Prediction confidence')
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=BLUE, label='Baseline'), Patch(color=ORANGE, label='Copy-paste')], frameon=False, ncol=2,
           fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.16))
save(fig, 'fig_err_confidence')

# ---------- T1 task comparison
B = pd.read_csv('task_comparison.csv')
targets = ['Focus-species mean'] + SPN
fig, axes = plt.subplots(2, 4, figsize=(6.9, 4.1), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1]))
MK = [('det_box', 'Detection, box AP50', BLUE, 'o', -0.2), ('seg_box', 'Segmentation, box AP50', ORANGE, 's', 0.0),
      ('seg_mask', 'Segmentation, mask AP50', AQUA, 'D', 0.2)]
for col, t in enumerate(targets):
    b = B[(B.metric == 'AP50') & (B.target == t)].set_index('size').loc[['s', 'm', 'l']]
    ax = axes[0, col]; ax.axhline(0, color=MUTED, lw=0.8)
    for k, lab, c, m, off in MK:
        for i, (sz, r) in enumerate(b.iterrows()):
            sig = r[f'{k}_lo'] > 0 or r[f'{k}_hi'] < 0
            ax.plot([i + off] * 2, [r[f'{k}_lo'], r[f'{k}_hi']], color=c, lw=1.3)
            ax.plot(i + off, r[f'{k}_effect'], m, color=c, mfc=c if sig else 'white', mew=1.2, ms=4.5, label=lab if i == 0 and col == 0 else None)
    ax.set_title(t if t != 'Orangespine Unicornfish' else 'Orangespine\nUnicornfish', fontsize=8, color=INK)
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax2 = axes[1, col]; ax2.axhline(0, color=MUTED, lw=0.8)
    for i, (sz, r) in enumerate(b.iterrows()):
        for lab, key, c, off in [('seg box − det box', 'DiD_segbox_minus_detbox', ORANGE, -0.1), ('seg mask − det box', 'DiD_segmask_minus_detbox', AQUA, 0.1)]:
            sig = r[f'{key}_lo'] > 0 or r[f'{key}_hi'] < 0
            ax2.plot([i + off] * 2, [r[f'{key}_lo'], r[f'{key}_hi']], color=c, lw=1.3)
            ax2.plot(i + off, r[key], 'o', color=c, mfc=c if sig else 'white', mew=1.2, ms=4.5, label=lab if i == 0 and col == 0 else None)
    ax2.set_xticks([0, 1, 2]); ax2.set_xticklabels(['S', 'M', 'L']); ax2.set_xlabel('Model size', fontsize=7.5)
    ax2.grid(axis='y', color=GRID, lw=0.6); ax2.set_axisbelow(True)
axes[0, 0].set_ylabel('Copy-paste effect\n(Δ AP50, 95% CI)'); axes[1, 0].set_ylabel('Difference in effect,\nseg − det (95% CI)')
h1, l1 = axes[0, 0].get_legend_handles_labels(); h2, l2 = axes[1, 0].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, ncol=3, frameon=False, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.07))
fig.tight_layout(rect=(0, 0.05, 1, 1))
save(fig, 'fig_task_effects')

# ---------- T2 components
K = pd.read_csv('components_by_setup.csv'); K = K[K.target == 'Focus species (pooled)']
SET = ['detect_s', 'detect_m', 'detect_l', 'segment_s', 'segment_m', 'segment_l']
SL = ['Det s', 'Det m', 'Det l', 'Seg s', 'Seg m', 'Seg l']
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.2), sharey=True)
for ax, comp, title in zip(axes, ['recall', 'localisation', 'identification'],
                           ['Recall (conf ≥ 0.25)', 'Localisation: any-species box\nIoU ≥ 0.5', 'Identification: correct species\namong localised fish']):
    k = K[K.component == comp].set_index('setup').loc[SET]
    ax.axvline(0, color=MUTED, lw=0.8)
    for i, (s, r) in enumerate(k.iterrows()):
        c = INK2 if s.startswith('detect') else AQUA; sig = r.lo > 0 or r.hi < 0
        ax.plot([r.lo, r.hi], [i, i], color=c, lw=1.3); ax.plot(r.effect, i, 'o', color=c, mfc=c if sig else 'white', mew=1.2, ms=4.5)
    ax.set_yticks(range(6)); ax.set_yticklabels(SL); ax.invert_yaxis(); ax.set_title(title, fontsize=8, color=INK)
    ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlabel('Copy-paste − baseline')
save(fig, 'fig_task_components')

# ---------- C1 trade-off
P = pd.read_csv('tradeoff_performance.csv'); CT = pd.read_csv('cost_table.csv')
fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
for ax, met, lab in zip(axes, ['focus_AP50_box', 'nonfocus17_AP50', 'anyfish_AP50'],
                        ['Focus-species mean AP50', 'Mean AP50, 17 non-target classes', 'Any-fish AP50']):
    for task, mk in [('det', 'o'), ('seg', 's')]:
        st = {'det': 'detect', 'seg': 'segment'}[task]
        for size in 'sml':
            g = CT[(CT.setup == f'{st}_{size}')].gflops_1280.iloc[0]
            b = P[(P.task == task) & (P['size'] == size) & (P.arm == 'baseline')].iloc[0]
            c = P[(P.task == task) & (P['size'] == size) & (P.arm == 'copypaste')].iloc[0]
            ax.annotate('', xy=(g, c[met]), xytext=(g, b[met]), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=0.8, shrinkA=3, shrinkB=3))
            ax.plot(g, b[met], mk, color=BLUE, ms=5, mec='white', mew=0.6)
            ax.plot(g, c[met], mk, color=ORANGE, ms=5, mec='white', mew=0.6)
            if met == 'focus_AP50_box':
                ax.text(g * 1.06, b[met] - 0.008, size, fontsize=6.5, color=INK2, va='top')
    ax.set_xscale('log'); ax.set_xticks([100, 200, 400]); ax.set_xticklabels(['100', '200', '400']); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('GFLOPs per 1280-px image'); ax.set_title(lab, fontsize=8, color=INK)
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', color=MUTED, label='Detection'), Line2D([], [], marker='s', ls='', color=MUTED, label='Segmentation'),
     Line2D([], [], marker='o', ls='', color=BLUE, label='Baseline'), Line2D([], [], marker='o', ls='', color=ORANGE, label='Copy-paste')]
fig.legend(handles=h, ncol=4, frameon=False, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.08))
fig.tight_layout(rect=(0, 0.06, 1, 1))
save(fig, 'fig_tradeoff')
print('ok')
