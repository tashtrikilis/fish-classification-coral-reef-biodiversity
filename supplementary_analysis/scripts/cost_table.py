"""Computational cost of each configuration (report Table 7), from the analysis bundle.

Reads analysis_bundle/logs (PBS job logs), run_meta.csv and model_complexity.csv and writes
../tables/log_meta.csv (one row per training run) and ../tables/cost_table.csv (per configuration and condition).
  inference time : the Ultralytics "Speed:" line of the final test-fold evaluation (H100, batch 8), median over runs
  training time  : seconds per epoch = mean training time / mean epochs (runs stop early at different epochs)
  GPU-hours      : sum of PBS walltime over all training jobs
"""
import os, re, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.join(HERE, 'analysis_bundle')
OUT = os.path.join(HERE, '..', 'tables'); os.makedirs(OUT, exist_ok=True)
meta = pd.read_csv(f'{B}/run_meta.csv'); runs = set(meta.run)

rows = []
for f in sorted(glob.glob(f'{B}/logs/*')):
    t = open(f, errors='replace').read()
    m = re.search(r'[,\s]NAME=([\w-]+)', t)
    if not m or m.group(1) not in runs: continue
    sp = re.findall(r'Speed: ([\d.]+)ms preprocess, ([\d.]+)ms inference, [\d.]+ms loss, ([\d.]+)ms postprocess', t)
    wt = re.search(r'walltime=(\d+):(\d+):(\d+)', t); ab = re.findall(r'Using batch-size (\d+)', t)
    ex = re.search(r'Exit_status\s*:\s*(-?\d+)', t)
    rows.append(dict(log=os.path.basename(f), run=m.group(1),
                     walltime_s=int(wt.group(1)) * 3600 + int(wt.group(2)) * 60 + int(wt.group(3)) if wt else np.nan,
                     n_speed=len(sp), test_pre=float(sp[-1][0]) if sp else np.nan, test_inf=float(sp[-1][1]) if sp else np.nan,
                     test_post=float(sp[-1][2]) if sp else np.nan, autobatch=int(ab[-1]) if ab else np.nan,
                     exit=int(ex.group(1)) if ex else np.nan))
L = pd.DataFrame(rows); L.to_csv(f'{OUT}/log_meta.csv', index=False)
print(f'{len(L)} training logs, {int((L.exit == 0).sum())} finished with exit status 0')

df = meta.merge(L, on='run'); cx = pd.read_csv(f'{B}/model_complexity.csv').set_index('setup')
cost = []
for (s, arm), g in df.groupby(['setup', 'arm']):
    tm = g.train_seconds / 60
    cost.append(dict(setup=s, arm=arm, runs=len(g), epochs_mean=g.epochs_run.mean(), epochs_min=g.epochs_run.min(),
                     epochs_max=g.epochs_run.max(), train_min_mean=tm.mean(), train_min_sd=tm.std(),
                     wall_min_mean=g.walltime_s.mean() / 60, gpu_h_total=g.walltime_s.sum() / 3600,
                     infer_ms_median=g.test_inf.median(), infer_ms_iqr_lo=g.test_inf.quantile(.25),
                     infer_ms_iqr_hi=g.test_inf.quantile(.75), autobatch=g.autobatch.median(),
                     ckpt_MB=g.best_pt_bytes.mean() / 1e6, params=cx.loc[s, 'params'], gflops_640=cx.loc[s, 'gflops_640'],
                     gflops_1280=cx.loc[s, 'gflops_1280'], sec_per_epoch=g.train_seconds.mean() / g.epochs_run.mean()))
C = pd.DataFrame(cost); C.to_csv(f'{OUT}/table7_cost.csv', index=False)
print(C[['setup', 'arm', 'params', 'gflops_1280', 'infer_ms_median', 'sec_per_epoch']].round(2).to_string(index=False))
print(f'Total: {df.walltime_s.sum() / 3600:.1f} GPU-hours over {len(df)} runs')
