"""Expanded native-source ablations; one GPU evaluator after the renderer exits."""
import argparse
from pathlib import Path
import time
from luster_hybrid_common import TARGETS,read,write,log
from bracket_adaptive_luster_hybrid import combine,CONFIG
from evaluate_luster_hybrid import compute
from screen_luster_hybrid import summarize


def run(root,wait_pid):
    selected=read(root/'preflight/sources.json')['sources']
    deadline=read(root/'preflight/manifest.json')['deadline']
    cases=[
      ('bracket32_s8',root,dict(local_sources=4,full_strength=False)),
      ('bracket32_k8_s8',root,dict(local_sources=8,full_strength=False)),
      ('bracket32_k8_full_s8',root,dict(local_sources=8,full_strength=True)),
      ('area32_k8_s8',root/'variants/area4',dict(local_sources=8,full_strength=False)),
      ('foreground_noise32_k8_s8',root/'variants/foreground_noise',dict(local_sources=8,full_strength=False)),
    ]
    summaries=[]
    for name,location,options in cases:
        config={**CONFIG,'adaptive_strength':False,'count':32,**options}
        for c in TARGETS:
            while any(not (location/'warps'/f'cam{c:03d}'/f'src{s:03d}'/'receipt.json').exists() for s in selected[str(c)][:32]):
                if time.time()>deadline:raise TimeoutError('Campaign deadline')
                time.sleep(5)
            receipt=root/'candidates'/name/f'cam{c:03d}_receipt.json'
            if receipt.exists():assert read(receipt)['config']==config
            else:combine(location,c,name,config)
        while wait_pid and Path(f'/proc/{wait_pid}').exists():time.sleep(5)
        compute(root,name,list(TARGETS),panels=True)
        summary=summarize(root,name);summaries.append(summary)
        write(root/'evaluation/screen_expanded.json',summaries)
        log(root,'expanded_measured',**summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pid',type=int)
    a=p.parse_args();run(a.root,a.wait_pid)
