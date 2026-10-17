"""Sequential GPU evaluation of independent resampling and source-choice ablations."""
import argparse
import os
from pathlib import Path
import signal
import time
from luster_hybrid_common import TARGETS,read,write,log
from transfer_luster_hybrid import combine,DEFAULT
from bracket_luster_hybrid import combine as bracket,CONFIG as BRACKET
from evaluate_luster_hybrid import compute
from screen_luster_hybrid import summarize


def run(root,wait_pid,resume_pid):
    try:
        while wait_pid and Path(f'/proc/{wait_pid}').exists():time.sleep(5)
        cases=[
          ('align2_detail_s8',root/'variants/align2',combine,{**DEFAULT,'count':8,'mode':'detail','sigma':8.}),
          ('bracket8_s8',root,bracket,BRACKET),
          ('bracket8_full_s8',root,bracket,{**BRACKET,'full_strength':True}),
          ('area8_detail_s8',root/'variants/area4',combine,{**DEFAULT,'count':8,'mode':'detail','sigma':8.}),
          ('s4_detail_s8',root,combine,{**DEFAULT,'count':4,'mode':'detail','sigma':8.}),
        ]
        summaries=[];selected=read(root/'preflight/sources.json')['sources']
        for name,location,fn,config in cases:
            for c in TARGETS:
                while any(not (location/'warps'/f'cam{c:03d}'/f'src{s:03d}'/'receipt.json').exists() for s in selected[str(c)][:config['count']]):time.sleep(5)
                receipt=root/'candidates'/name/f'cam{c:03d}_receipt.json'
                if receipt.exists():assert read(receipt)['config']==config
                else:fn(location,c,name,config)
            compute(root,name,list(TARGETS),panels=True)
            summary=summarize(root,name);summaries.append(summary)
            write(root/'evaluation/screen_refinement.json',summaries)
            log(root,'refinement_measured',**summary)
    finally:
        if resume_pid and Path(f'/proc/{resume_pid}').exists():os.kill(resume_pid,signal.SIGCONT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--wait-pid',type=int);p.add_argument('--resume-pid',type=int)
    a=p.parse_args();run(a.root,a.wait_pid,a.resume_pid)
