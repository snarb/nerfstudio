"""Ablate source neighborhood and transfer band with one native source."""
from pathlib import Path
from single_source_luster_hybrid import DEFAULT,combine
from evaluate_luster_hybrid import compute
from screen_luster_hybrid import summarize
from luster_hybrid_common import write,log
root=Path('/fsx/tmp/luster_000540_hybrid_20261017T104440Z')
common={**DEFAULT,'count':4,'match_weight':.25,'match_limit':.06,'gain':.75}
cases=[('single_near4_full_s8',{'gain':1.}),('single_near4_s16',{'sigma':16.}),('single_near4_s32',{'sigma':32.}),('single_near4_direct',{'mode':'direct'}),('single_near8_s8',{'count':8}),('single_near16_s8',{'count':16}),('single_near4_full_s16',{'gain':1.,'sigma':16.})]
results=[]
for name,change in cases:
 config={**common,**change}
 for c in [7,12,95,150]:
  if not (root/f'candidates/{name}/cam{c:03d}_receipt.json').exists():combine(root,c,name,config)
 if not (root/f'evaluation/{name}_metrics.json').exists():compute(root,name,[7,12,95,150])
 result=summarize(root,name);results.append(result);write(root/'evaluation/single_near_screen.json',results);log(root,'single_near_evaluated',**result)
