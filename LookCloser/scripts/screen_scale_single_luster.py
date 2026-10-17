"""Controlled scale-compatible single-source recipe ablation, sequential CUDA."""
import argparse,time
from pathlib import Path
from scale_single_luster_hybrid import DEFAULT,combine
from evaluate_luster_hybrid import compute
from screen_luster_hybrid import summarize
from luster_hybrid_common import read,write,log
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--wait-pid',type=int);a=p.parse_args();root=a.root
if a.wait_pid:
 while Path(f'/proc/{a.wait_pid}').exists():time.sleep(2)
common={**DEFAULT,'count':16,'match_weight':.25,'match_limit':.06,'gain':1.}
cases=[('single_scale16_s4',{'sigma':4.}),('single_scale16_s6',{'sigma':6.}),('single_scale16_s12',{'sigma':12.}),('single_scale16_support6',{'support_mass':.6}),('single_scale16_geometry2',{'roundtrip':2.,'uncertainty':4.}),('single_scale32_s8',{'count':32}),('single_scale16_match4',{'match_limit':.04}),('single_scale16_match10',{'match_limit':.10})]
results=[]
for name,change in cases:
 if time.time()>read(root/'preflight/manifest.json')['deadline']:raise TimeoutError('Campaign deadline')
 config={**common,**change}
 for c in [7,12,95,150]:
  if not (root/f'candidates/{name}/cam{c:03d}_receipt.json').exists():combine(root,c,name,config)
 if not (root/f'evaluation/{name}_metrics.json').exists():compute(root,name,[7,12,95,150])
 result=summarize(root,name);results.append(result);write(root/'evaluation/single_scale_ablation.json',results);log(root,'single_scale_ablation',**result)
