import sys, numpy as np, pickle; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import engine as G, common as C
TAU=0.25
out={}
for s in C.SETUPS:
  for iou in (['box'] if s.startswith('detect') else ['box','mask']):
    for a in C.ARMS:
      for sd in C.SEEDS:
        cd=G.classdata(s,a,sd,iou)
        for c in C.FOCUS_IDS:
          d=cd.get(c); sel=d.conf>=TAU; t=d.tp[0][sel].astype(float); b=d.block[sel]
          out[(s,iou,a,sd,c)]=dict(tpb=np.bincount(b,weights=t,minlength=C.NB), fpb=np.bincount(b,weights=1-t,minlength=C.NB), gtb=d.gt.astype(float))
        print(s,iou,a,sd,flush=True)
pickle.dump(out,open('prf.pkl','wb'))
