import re, json
U='../../results/'   # run from report_figures/scripts/
src={'Detection s':U+'cv_results_round6/cv_results_detect_s_box_seeds0-1/cv_results.md',
     'Detection m':U+'cv_results_round5/cv_results_seeds0-1/cv_results.md',
     'Detection l':U+'cv_results_round6/cv_results_detect_l_box_seeds0-1/cv_results.md',
     'Segmentation s':U+'cv_results_round6/cv_results_segment_s_mask_seeds0-1/cv_results.md',
     'Segmentation m':U+'cv_results_round6/cv_results_segment_m_mask_seeds0-1/cv_results.md',
     'Segmentation l':U+'cv_results_round6/cv_results_segment_l_mask_seeds0-1/cv_results.md',
     'Segmentation l (box)':U+'cv_results_round6/cv_results_segment_l_box_seeds0-1/cv_results.md',
     'Segmentation s (box)':U+'cv_results_round6/cv_results_segment_s_box_seeds0-1/cv_results.md',
     'Segmentation m (box)':U+'cv_results_round6/cv_results_segment_m_box_seeds0-1/cv_results.md'}
num=r'([+-]?\d+\.\d+)'
out={}
for k,f in src.items():
    d={}
    for line in open(f):
        m=re.match(r'\| (\*\*)?([^|*]+?)(\*\*)? \| (\d*) *\| (AP50|R) \| '+num+r' \[([\d.]+)–([\d.]+)\] \| '+num+r' \[([\d.]+)–([\d.]+)\] \| '+num+r' \['+num+' to '+num+r'\] \| ([\d.]+)',line)
        if m:
            name=m.group(2).strip(); met=m.group(5)
            d[f'{name}|{met}']=dict(base=float(m.group(6)),blo=float(m.group(7)),bhi=float(m.group(8)),cp=float(m.group(9)),diff=float(m.group(12)),lo=float(m.group(13)),hi=float(m.group(14)),p=float(m.group(15)))
    out[k]=d
    print(k,len(d))
json.dump(out,open('r6.json','w'),indent=1)
