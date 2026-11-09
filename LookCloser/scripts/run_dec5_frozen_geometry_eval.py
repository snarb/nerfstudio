#!/usr/bin/env python3
"""Frozen matched-source depth alternatives and excluded-camera RGB evaluation."""
from pathlib import Path
import argparse,json,subprocess,sys
from run_dec5_hair_campaign import atomic_json,sha


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in ['root','training','checkpoint','output']:p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--data',choices=['crop_hd','crop_native'],required=True)
 p.add_argument('--empty-space-grid',type=Path);p.add_argument('--min-background-views',type=int,choices=[3,8],default=3)
 p.add_argument('--depth-keys',nargs='+',default=['median','expected'])
 p.add_argument('--checkpoint-role',choices=['native_rgb_best','auxiliary_final_geometry'],required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);n=a.root;s=Path(__file__).parent
 def run(script,*args):subprocess.run([sys.executable,str(s/script),*map(str,args)],check=True)
 qs=[]
 for key in a.depth_keys:
  if key in ['median','expected']:continue
  if len(key)!=5 or key[0]!='q' or not key[1:].isdigit() or not 0<int(key[1:])<10000:raise ValueError('Unknown native depth statistic')
  qs.append(str(int(key[1:])/10000))
 extras=['--extra-quantiles',*qs] if qs else []
 if a.empty_space_grid:extras+=['--empty-space-grid',str(a.empty_space_grid),'--min-background-views',str(a.min_background_views)]
 common=['--data',n/a.data,'--training',a.training,'--checkpoint',a.checkpoint,'--include-median','--uint8-cpu-cache']
 sources=a.output/'sources';targets=a.output/'targets'
 run('export_dec5_native_depth.py',*common,'--source-semantics',n/'semantics_hd','--camera-manifest',n/'source_hd_cameras.json','--output',sources,*extras)
 run('export_dec5_native_depth.py',*common,'--parent',n/'roi_4k_hull_v2','--camera-manifest',n/'validation_4k_cameras.json','--output',targets,*extras)
 scores={}
 for key in a.depth_keys:
  prediction=a.output/(key+'_prediction');metrics=a.output/(key+'_metrics.json')
  run('render_dec5_native_head_benchmark.py','--source-rgb-data',n/'crop_native','--source-cameras',n/'source_hd_cameras.json','--source-depth',sources,'--semantics',n/'semantics_hd','--target-depth',targets,'--fallback',n/'ray_bg_fallback_uhd_hull_v2','--cameras',n/'validation_4k_cameras.json','--depth-key',key,'--output',prediction)
  run('score_dec5_hair.py','--root',n/'benchmark_scene','--prediction',prediction,'--output',metrics,'--boundary-radius','5')
  scores[key]=json.load(open(metrics))['aggregate'];atomic_json(a.output/'progress.json',dict(completed=len(scores),total=len(a.depth_keys),depth_key=key))
 atomic_json(a.output/'complete.json',dict(checkpoint_sha256=sha(a.checkpoint),checkpoint_role=a.checkpoint_role,training_request_sha256=sha(a.training/'request.json'),scores=scores,fitting=False,uses_validation_rgb_for_fitting=False,uses_test_rgb=False,same_field_source_target=True,common_rgb_transfer=True,artifact_kind='UHD held-out head ROI; native JPEG-derived RGB; no independent depth GT'))
if __name__=='__main__':main()
