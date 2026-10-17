"""CPU-only controls for the small hand-region regression after cubic sampling."""
import argparse,time
from pathlib import Path
from risk_single_luster_hybrid import combine
from quality_risk_luster_hybrid import combine as quality_combine,DEFAULT
from luster_hybrid_common import read
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root;variant=root/'variants/cubic4';config=read(root/'candidates/single_risk_patch/config.json')
for name,change in [('single_cubic_s4',{'sigma':4.}),('single_cubic_match5',{'match_limit':.05})]:
 for c in [7,12,95,150]:
  if time.time()>read(root/'preflight/manifest.json')['deadline']:raise TimeoutError('Campaign deadline')
  combine(variant,c,name,{**config,**change})
for c in [7,12,95,150]:quality_combine(variant,c,'single_cubic_sharp_cost',{**DEFAULT,**config,'sharpness_weight':.25})
