"""CPU ablations: geometric footprint and avoiding softer source patches."""
import argparse,time
from pathlib import Path
from quality_risk_luster_hybrid import DEFAULT,combine
from luster_hybrid_common import read
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
common={**DEFAULT,**read(root/'candidates/single_risk_patch/config.json')}
cases=[('single_risk_surface_gate',{'surface_gate':True}),('single_risk_surface_cost',{'surface_gate':True,'surface_weight':.25}),('single_risk_sharp_cost',{'sharpness_weight':.25}),('single_risk_sharp_gate',{'sharpness_weight':.25,'sharpness_min':.75})]
for name,change in cases:
 if time.time()>read(root/'preflight/manifest.json')['deadline']:raise TimeoutError('Campaign deadline')
 for c in [7,12,95,150]:
  if not (root/f'candidates/{name}/cam{c:03d}_receipt.json').exists():combine(root,c,name,{**common,**change})
