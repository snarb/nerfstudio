"""CPU-only controlled single-source seam ablations; GPU evaluation is queued."""
import argparse,time
from pathlib import Path
from seam_single_luster_hybrid import combine
from luster_hybrid_common import read
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
settings=dict(filter_support='labels',seam_fade=3.,adaptive_seam=False,seam_tolerance=.02)
cases=[('single_seam_fade1',dict(seam_fade=1.)),('single_seam_fade2',dict(seam_fade=2.)),('single_seam_support3',dict(filter_support='geometry')),('single_seam_adaptive',dict(filter_support='geometry',adaptive_seam=True))]
for name,change in cases:
 if time.time()>read(a.root/'preflight/manifest.json')['deadline']:raise TimeoutError('Campaign deadline')
 combine(a.root,'single_scale16_match10',name,{**settings,**change})
