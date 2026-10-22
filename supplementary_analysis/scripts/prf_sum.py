import sys, numpy as np, pickle; sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import engine as G, common as C
D=pickle.load(open('prf.pkl','rb')); W=G.W  # row0 = point estimate, rows1.. = bootstrap
def runs(scheme):
    for s in C.SETUPS:
        iou = 'box' if (scheme=='box' or s.startswith('detect')) else 'mask'
        for sd in C.SEEDS: yield s, iou, sd
def pooled(c, a, scheme):
    tp=fp=gt=0
    for s,iou,sd in runs(scheme):
        r=D[(s,iou,a,sd,c)]; tp=tp+W@r['tpb']; fp=fp+W@r['fpb']; gt=gt+W@r['gtb']
    P=tp/np.maximum(tp+fp,1e-9); R=tp/gt; F=2*P*R/np.maximum(P+R,1e-9)
    return P,R,F,tp[0],fp[0],gt[0]
def mean_runs(c,a,scheme):
    Ps,Rs,Fs=[],[],[]
    for s,iou,sd in runs(scheme):
        r=D[(s,iou,a,sd,c)]; tp,fp,gt=r['tpb'].sum(),r['fpb'].sum(),r['gtb'].sum()
        P=tp/(tp+fp) if tp+fp>0 else np.nan; R=tp/gt; Ps.append(P); Rs.append(R); Fs.append(2*P*R/(P+R) if P+R>0 else 0)
    return np.nanmean(Ps),np.mean(Rs),np.nanmean(Fs)
def ci(x): x=x[~np.isnan(x)]; return np.percentile(x,2.5),np.percentile(x,97.5)
for scheme in ['primary','box']:
    print('=== scheme', scheme)
    for c in C.FOCUS_IDS:
        b=pooled(c,'baseline',scheme); a=pooled(c,'copypaste',scheme)
        dP=(a[0]-b[0]); dR=(a[1]-b[1])
        print(f"{C.NAMES[c]:24s} P {b[0][0]:.3f}->{a[0][0]:.3f} dP {100*dP[0]:+.1f}pp [{100*ci(dP[1:])[0]:+.1f},{100*ci(dP[1:])[1]:+.1f}] | R {b[1][0]:.3f}->{a[1][0]:.3f} dR {100*dR[0]:+.1f}pp [{100*ci(dR[1:])[0]:+.1f},{100*ci(dR[1:])[1]:+.1f}] | F1 {b[2][0]:.3f}->{a[2][0]:.3f} | TP {b[3]:.0f}->{a[3]:.0f} FP {b[4]:.0f}->{a[4]:.0f} GT {b[5]:.0f}")
        mb=mean_runs(c,'baseline',scheme); ma=mean_runs(c,'copypaste',scheme)
        print(f"   mean-of-runs: P {mb[0]:.3f}->{ma[0]:.3f} R {mb[1]:.3f}->{ma[1]:.3f} F1 {mb[2]:.3f}->{ma[2]:.3f}")
