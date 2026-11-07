"""Finite current-native ROI queue, respecting actual field claims every tick."""
from pathlib import Path
import json,sys,time,os
V=Path('/fsx/tmp/marshawn_video_1660_2015_24fps_20261107');sys.path[:0]=[str(V/'operations/sam3_assist/code'),str(V/'code/LookCloser/scripts')]
import marshawn_tuning_controller as fleet
import marshawn_native_pool_v6 as pool

def main(root,planpath,request):
 held=fleet.lock(root/'controller.lock')
 if held is None:return
 plan,state=fleet.prepare(root,planpath);fleet.atomic(root/'controller_identity.json',fleet.identity());original_request=fleet.read(request);original_nonce=original_request['nonce']
 while True:
  if fleet.digest(plan)!=state['manifest_sha256']:raise ValueError('ROI plan changed')
  allowed,proof=pool.available(V,plan['hosts'],state);fleet.atomic(root/'pool_snapshot.json',proof);filtered=dict(plan,hosts={h:dict(s,gpus=allowed[h])for h,s in plan['hosts'].items()})
  complete=fleet.tick(filtered,state,root)
  current=fleet.read(request)
  if current['nonce']!=original_nonce:raise ValueError('ROI request owner changed')
  current.update(status='complete'if complete else'running',actual_state=str(root/'state.json'),actual_state_sha256=__import__('hashlib').sha256((root/'state.json').read_bytes()).hexdigest(),updated=time.time());fleet.atomic(request,current)
  if complete:return
  time.sleep(5)
if __name__=='__main__':main(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
