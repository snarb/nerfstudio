"""Sequential single-source texture screen, frozen geometry and exact lookups."""
from pathlib import Path
from single_source_luster_hybrid import DEFAULT,combine
from evaluate_luster_hybrid import compute
from screen_luster_hybrid import summarize
from luster_hybrid_common import read,write,log
root=Path('/fsx/tmp/luster_000540_hybrid_20261017T104440Z')
cases=[('single_patch_s8',{}),('single_patch_s16',{'sigma':16.}),('single_patch_direct',{'mode':'direct'}),('single_patch_strict_s8',{'match_limit':.04}),('single_patch_s32',{'sigma':32.})]
results=[]
for name,change in cases:
    config={**DEFAULT,**change}
    for camera in [7,12,95,150]:
        if not (root/f'candidates/{name}/cam{camera:03d}_receipt.json').exists():combine(root,camera,name,config)
    if not (root/f'evaluation/{name}_metrics.json').exists():compute(root,name,[7,12,95,150])
    result=summarize(root,name);results.append(result);write(root/'evaluation/single_source_screen.json',results);log(root,'single_source_measured',**result)
