"""CPU native cubic sampling and unchanged automatic hard source selection."""
import argparse,time
from pathlib import Path
from luster_hybrid_common import read
from cubic_luster_hybrid import variant_root,warp
from risk_single_luster_hybrid import combine
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
config=read(root/'candidates/single_risk_patch/config.json');variant=variant_root(root)
for c,sources in read(root/'preflight/sources.json')['sources'].items():
 if time.time()>read(root/'preflight/manifest.json')['deadline']:raise TimeoutError('Campaign deadline')
 for source in sources[:config['count']]:warp(root,int(c),source)
 combine(variant,int(c),'single_cubic_risk_patch',config)
