"""CPU ablation isolating spatial source-cost coherence from risk weights."""
import argparse,time
from pathlib import Path
from risk_single_luster_hybrid import DEFAULT,combine
from luster_hybrid_common import read
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
common={**DEFAULT,'count':16,'match_weight':.25,'match_limit':.06,'gain':1.,'sigma':6.,'support_mass':.6,'roundtrip_weight':.5,'uncertainty_weight':.2,'depth_weight':.2,'cost_sigma':3.}
cases=[('single_patch_cost3',{'roundtrip_weight':.1,'uncertainty_weight':0.,'depth_weight':0.}),('single_risk_patch6',{'cost_sigma':6.}),('single_risk_patch12',{'cost_sigma':12.}),('single_risk_patch_match10',{'match_limit':.10})]
for name,change in cases:
 if time.time()>read(root/'preflight/manifest.json')['deadline']:raise TimeoutError('Campaign deadline')
 for c in [7,12,95,150]:
  if not (root/f'candidates/{name}/cam{c:03d}_receipt.json').exists():combine(root,c,name,{**common,**change})
