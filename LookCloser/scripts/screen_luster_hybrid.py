"""Sequential, resumable native recipe screen; no training and no GT-dependent masks."""
import argparse
import os
from pathlib import Path
import time

from luster_hybrid_common import TARGETS,log,read,write
from transfer_luster_hybrid import DEFAULT,combine
from evaluate_luster_hybrid import compute


def summarize(root,name):
    baseline=read(root/'evaluation/baseline_metrics.json')
    result=read(root/'evaluation'/f'{name}_metrics.json')
    ref={(r['camera'],r['region']):r for r in baseline['rows'] if r['status']=='measured'}
    bad=[dict(camera=r['camera'],region=r['region'],lpips_delta=r['lpips']-ref[(r['camera'],r['region'])]['lpips'])
         for r in result['rows'] if r['status']=='measured' and r['lpips']>ref[(r['camera'],r['region'])]['lpips']+1e-6]
    value=result['validation_mean']['lpips'];ratio=value/baseline['validation_mean']['lpips']
    return dict(candidate=name,lpips=value,reduction_percent=100*(1-ratio),regressions=bad,
                numeric_pass=not bad and ratio<1,goal_numeric_pass=not bad and ratio<=.6,
                visual_review='pending')


def run(root,wait_pid=None):
    if wait_pid:
        while Path(f'/proc/{wait_pid}').exists():time.sleep(5)
    # This runner owns the GPU after the source renderer has exited.
    from audit_luster_hybrid import audit
    if not (root/'preflight/numerical_audit.json').exists():audit(root)
    for repeat in [1,2]:
        tag=f'baseline_repeat{repeat}'
        if not (root/'evaluation'/f'{tag}_metrics.json').exists():compute(root,'baseline',list(TARGETS),output_tag=tag)
    selected=read(root/'preflight/sources.json')['sources']
    while any(not (root/'warps'/f'cam{int(c):03d}'/f'src{s:03d}'/'receipt.json').exists()
              for c,ids in selected.items() for s in ids[:8]):time.sleep(5)
    configs=[]
    for mode,sigma in [('direct',8.),('detail',4.),('detail',8.),('detail',16.),('detail',32.),('multiband',4.),('multiband',8.)]:
        configs.append((f's8_{mode}_s{int(sigma)}',{**DEFAULT,'count':8,'mode':mode,'sigma':sigma}))
    configs.extend([
       ('s8_detail_strict',{**DEFAULT,'count':8,'mode':'detail','sigma':8.,'roundtrip':.5,'depth_error':.0025,'uncertainty':1.,'agreement':.04}),
       ('s8_detail_wide',{**DEFAULT,'count':8,'mode':'detail','sigma':8.,'roundtrip':2.,'depth_error':.01,'uncertainty':4.,'alpha':.95}),
       ('s8_detail_half',{**DEFAULT,'count':8,'mode':'detail','sigma':8.,'gain':.5}),
    ])
    summaries=[]
    for name,config in configs:
        write(root/'preflight/screen_state.json',dict(state='running',candidate=name,completed=summaries,time=time.time()))
        for c in TARGETS:
            receipt=root/'candidates'/name/f'cam{c:03d}_receipt.json'
            if receipt.exists():assert read(receipt)['config']==config
            else:combine(root,c,name,config)
        if not (root/'evaluation'/f'{name}_metrics.json').exists():compute(root,name,list(TARGETS))
        summary=summarize(root,name);summaries.append(summary)
        write(root/'evaluation/screen8.json',summaries)
        log(root,'candidate_measured',**summary)
    write(root/'preflight/screen_state.json',dict(state='complete_pending_visual_review',completed=summaries,time=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pid',type=int)
    a=p.parse_args();run(a.root,a.wait_pid)
