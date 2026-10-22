import json, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Nimbus Roman','DejaVu Serif'],'font.size':8.5})
d=json.load(open('r6.json'))
setups=['Detection s','Detection m','Detection l','Segmentation s','Segmentation m','Segmentation l']
labels=['Detection s','Detection m','Detection l','Segmentation s','Segmentation m','Segmentation l']
panels=[('Focus species (mean)','Focus-species mean\n(3 species)'),('Forcepsfish','Forcepsfish\n(93 fish)'),('Orangespine Unicornfish','Orangespine Unicornfish\n(77 fish)'),('Convict Tang','Convict Tang\n(33 fish)')]
BLUE='#2a78d6'; INK='#2b2a27'; MUTED='#8a887f'; GRID='#e4e3de'
fig,axes=plt.subplots(1,4,figsize=(6.9,2.7),sharey=True)
for ax,(key,title) in zip(axes,panels):
    ax.axvline(0,color=MUTED,lw=0.9,zorder=1)
    for i,s in enumerate(setups):
        y=len(setups)-1-i
        r=d.get(s,{}).get(key+'|AP50')
        if r is None:
            ax.text(0.0,y,'pending',ha='center',va='center',color=MUTED,fontsize=7.5,style='italic'); continue
        sig=r['lo']>0 or r['hi']<0
        ax.plot([r['lo'],r['hi']],[y,y],color=BLUE,lw=1.6,solid_capstyle='round',zorder=2)
        ax.plot(r['diff'],y,'o',ms=5.5,mfc=BLUE if sig else 'white',mec=BLUE,mew=1.4,zorder=3)
    ax.set_title(title,fontsize=8.5,color=INK)
    ax.set_xlim(-0.25,0.42); ax.set_xticks([-0.2,0,0.2,0.4])
    ax.grid(axis='x',color=GRID,lw=0.6); ax.set_axisbelow(True)
    for sp in ['top','right','left']: ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color(MUTED); ax.tick_params(colors=INK,length=2)
    ax.set_ylim(-0.6,len(setups)-0.4); ax.tick_params(axis='y',left=False)
axes[0].set_yticks(range(len(setups))); axes[0].set_yticklabels(labels[::-1])
axes[0].tick_params(axis='y',length=0)
from matplotlib.lines import Line2D
h=[Line2D([],[],marker='o',ls='-',color=BLUE,mfc=BLUE,ms=5,label='95% CI excludes 0'),Line2D([],[],marker='o',ls='-',color=BLUE,mfc='white',ms=5,label='95% CI includes 0')]
fig.legend(handles=h,loc='lower center',ncol=2,frameon=False,fontsize=7.5,bbox_to_anchor=(0.55,-0.02))
fig.supxlabel('Change in AP50, copy-paste minus baseline (pooled test folds, mean of seeds 0 and 1)',fontsize=7.5,color=INK,y=0.09)
fig.tight_layout(rect=(0,0.1,1,1))
fig.savefig('cp_grid_forest.pdf'); fig.savefig('cp_grid_forest.png',dpi=200)
