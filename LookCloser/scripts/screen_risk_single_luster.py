"""Controlled automatic source-ranking ablation; predictions CPU, evaluation queued."""
import argparse,time
from pathlib import Path
from risk_single_luster_hybrid import DEFAULT,combine
from luster_hybrid_common import read,write
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
common={**DEFAULT,'count':16,'match_weight':.25,'match_limit':.06,'gain':1.,'sigma':6.,'support_mass':.6}
cases=[('single_scale16_s6_support6',{}),('single_risk_soft',{'roundtrip_weight':.5,'uncertainty_weight':.2,'depth_weight':.2}),('single_risk_patch',{'roundtrip_weight':.5,'uncertainty_weight':.2,'depth_weight':.2,'cost_sigma':3.})]
for name,change in cases:
 if time.time()>read(root/'preflight/manifest.json')['deadline']:raise TimeoutError('Campaign deadline')
 for c in [7,12,95,150]:
  if not (root/f'candidates/{name}/cam{c:03d}_receipt.json').exists():combine(root,c,name,{**common,**change})
