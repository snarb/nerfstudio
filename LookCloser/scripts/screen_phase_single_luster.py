"""CPU-only structural-phase compatibility ablation on frozen geometry."""
import argparse,time
from pathlib import Path
from phase_single_luster_hybrid import DEFAULT,combine
from luster_hybrid_common import read
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
common={**DEFAULT,'count':16,'match_weight':.25,'match_limit':.06,'gain':1.,'sigma':6.,'support_mass':.6,'roundtrip_weight':.5,'uncertainty_weight':.2,'depth_weight':.2,'cost_sigma':3.}
cases=[('single_phase25',{'phase_weight':.25}),('single_phase75',{'phase_weight':.75}),('single_phase_guard',{'phase_weight':.25,'phase_min':.3})]
for name,change in cases:
 if time.time()>read(root/'preflight/manifest.json')['deadline']:raise TimeoutError('Campaign deadline')
 for c in [7,12,95,150]:
  if not (root/f'candidates/{name}/cam{c:03d}_receipt.json').exists():combine(root,c,name,{**common,**change})
